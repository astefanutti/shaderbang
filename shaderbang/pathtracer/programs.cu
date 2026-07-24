// Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
// SPDX-License-Identifier: MIT
//
// OptiX device programs for the shaderbang path tracer (milestone M1).
//
// M1 is the "first live frame": one primary ray per pixel against the cloth GAS
// plus analytic sphere and ground-plane intersection, single-bounce Lambert
// direct lighting from one directional light (no shadows yet), a gradient sky on
// miss, and progressive HDR accumulation. Shadows / Disney BSDF / MIS / NEE /
// env sampling arrive in later milestones (see docs/pathtracer.md).
//
// The Params struct below MUST match PARAMS_DTYPE in renderer.py field for
// field. Self-contained: only <optix.h>, with minimal inline float3 math (no
// vec_math.h) so NVRTC needs just the OptiX + CUDA include dirs.

#include <optix.h>

// --------------------------------------------------------------------------- //
// Minimal float3 math
// --------------------------------------------------------------------------- //
static __forceinline__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __forceinline__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __forceinline__ __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __forceinline__ __device__ float3 operator*(float3 a, float s)  { return make_float3(a.x * s, a.y * s, a.z * s); }
static __forceinline__ __device__ float3 operator*(float s, float3 a)  { return make_float3(a.x * s, a.y * s, a.z * s); }
static __forceinline__ __device__ float  dot(float3 a, float3 b)       { return a.x * b.x + a.y * b.y + a.z * b.z; }
static __forceinline__ __device__ float3 cross(float3 a, float3 b)     { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static __forceinline__ __device__ float3 normalize(float3 v)          { float inv = rsqrtf(dot(v, v)); return v * inv; }
static __forceinline__ __device__ float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
static __forceinline__ __device__ float  clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

// --------------------------------------------------------------------------- //
// Launch parameters (mirror renderer.PARAMS_DTYPE exactly)
// --------------------------------------------------------------------------- //
struct Params
{
    // --- 8-byte members first (pointers + handle) --- //
    float4*                accum;         // HDR accumulator, width*height (input res)
    float4*                output;        // per-frame HDR (= accum / (subframe+1)), denoiser input
    float4*                albedo;        // guide AOV: per-pixel surface albedo
    float4*                normal;        // guide AOV: per-pixel view-space normal (+z toward camera)
    float3*                prev_vertices; // previous-frame cloth vertex positions (for motion vectors)
    uint3*                 tri_indices;   // triangle vertex-index triplets (prev-vertex lookup)
    float2*                flow;          // output motion-vector AOV (input res, curr -> prev in pixels)
    OptixTraversableHandle handle;        // cloth GAS

    // --- 4-byte scalars --- //
    unsigned int           width;         // input (render) width
    unsigned int           height;        // input (render) height
    unsigned int           subframe;      // 0 resets the accumulator
    float                  exposure;      // unused on device (tonemap is a Warp kernel)

    // --- float3 basis / colors (float3 has 4-byte alignment) --- //
    float3                 cam_eye;
    float3                 cam_u;
    float3                 cam_v;
    float3                 cam_w;

    float3                 prev_cam_eye;  // previous-frame camera (for motion-vector reprojection)
    float3                 prev_cam_u;
    float3                 prev_cam_v;
    float3                 prev_cam_w;

    float3                 light_dir;     // normalized, points toward the light
    float3                 light_color;
    float3                 sky_top;
    float3                 sky_bottom;

    float3                 sphere_center;
    float3                 sphere_albedo;
    float                  sphere_radius;

    float3                 sphere_center_prev; // previous-frame sphere center (rigid motion)

    float3                 ground_albedo;
    float                  ground_y;

    float3                 cloth_albedo_front;
    float3                 cloth_albedo_back;
};

// Firefly clamp on per-sample radiance luminance. Direct-only shading (M3) is
// already bounded, so this is a no-op today; it becomes load-bearing once GI /
// NEE add stochastic bounces in M4.
#define PT_MAX_RADIANCE 64.0f

extern "C" {
__constant__ Params params;
}

// Payloads: p0 = hit t (float bits), p1..p3 = geometric normal, p4..p6 =
// previous-frame world position of the hit (for motion vectors).
static __forceinline__ __device__ void setHitPayload(float t, float3 n, float3 pPrev)
{
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(__float_as_uint(n.x));
    optixSetPayload_2(__float_as_uint(n.y));
    optixSetPayload_3(__float_as_uint(n.z));
    optixSetPayload_4(__float_as_uint(pPrev.x));
    optixSetPayload_5(__float_as_uint(pPrev.y));
    optixSetPayload_6(__float_as_uint(pPrev.z));
}

// Invert the pin-hole ray generation: solve D = a*cam_u + b*cam_v + c*cam_w for
// (a, b, c) via Cramer's rule (scalar triple products). The NDC coordinates are
// (a/c, b/c) -- exactly the (dx, dy) the raygen program maps to a pixel -- and c
// is the depth along the (non-normalized) forward axis. Returns (a/c, b/c, c).
static __forceinline__ __device__ float3 solveCameraNDC(float3 D, float3 u, float3 v, float3 w)
{
    float denom = dot(u, cross(v, w));
    float inv = 1.0f / denom;
    float a = dot(D, cross(v, w)) * inv;
    float b = dot(u, cross(D, w)) * inv;
    float c = dot(u, cross(v, D)) * inv;
    return make_float3(a, b, c);
}

// Pixel coordinates (buffer convention: x right, y = launch-index y, both at the
// pixel-center offset the raygen uses) for a world point seen by the given
// camera. Returns false if the point projects behind the camera.
static __forceinline__ __device__ bool projectToPixel(
        float3 P, float3 eye, float3 u, float3 v, float3 w,
        unsigned int W, unsigned int H, float2& outPix)
{
    float3 abc = solveCameraNDC(P - eye, u, v, w);
    if (abc.z <= 1e-6f)
        return false;
    outPix.x = (abc.x / abc.z + 1.0f) * 0.5f * (float)W;
    outPix.y = (abc.y / abc.z + 1.0f) * 0.5f * (float)H;
    return true;
}

// Same, for a pure direction (a point at infinity, e.g. the sky): the eye
// translation is irrelevant, only the camera basis (rotation) matters.
static __forceinline__ __device__ bool projectDirToPixel(
        float3 Dd, float3 u, float3 v, float3 w,
        unsigned int W, unsigned int H, float2& outPix)
{
    float3 abc = solveCameraNDC(Dd, u, v, w);
    if (abc.z <= 1e-6f)
        return false;
    outPix.x = (abc.x / abc.z + 1.0f) * 0.5f * (float)W;
    outPix.y = (abc.y / abc.z + 1.0f) * 0.5f * (float)H;
    return true;
}

// Deterministic per-sample hash for subpixel jitter (PCG-ish).
static __forceinline__ __device__ unsigned int pcg(unsigned int v)
{
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static __forceinline__ __device__ float uintToUnitFloat(unsigned int x)
{
    return (x >> 8) * (1.0f / 16777216.0f);  // [0, 1)
}

// Nearest positive ray-sphere hit distance, or -1.
static __forceinline__ __device__ float intersectSphere(float3 o, float3 d, float3 c, float r)
{
    float3 oc = o - c;
    float b = dot(oc, d);
    float cc = dot(oc, oc) - r * r;
    float disc = b * b - cc;
    if (disc < 0.0f)
        return -1.0f;
    float s = sqrtf(disc);
    float t0 = -b - s;
    if (t0 > 1e-4f)
        return t0;
    float t1 = -b + s;
    if (t1 > 1e-4f)
        return t1;
    return -1.0f;
}

// Ray vs. infinite horizontal plane y = ground_y, or -1.
static __forceinline__ __device__ float intersectGround(float3 o, float3 d, float y)
{
    if (fabsf(d.y) < 1e-6f)
        return -1.0f;
    float t = (y - o.y) / d.y;
    return (t > 1e-4f) ? t : -1.0f;
}

static __forceinline__ __device__ float3 skyColor(float3 dir)
{
    float t = clampf(0.5f * (dir.y + 1.0f), 0.0f, 1.0f);
    return lerp(params.sky_bottom, params.sky_top, t);
}

// Lambert direct lighting with a sky-colored ambient fill. ``nf`` must already
// face the viewer (see the raygen program). Returns the un-clamped HDR radiance.
static __forceinline__ __device__ float3 shade(float3 nf, float3 albedo)
{
    float ndl = fmaxf(dot(nf, params.light_dir), 0.0f);
    float3 ambient = skyColor(nf) * 0.25f;
    float3 direct = params.light_color * ndl;
    return albedo * (ambient + direct);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int pixel = idx.y * params.width + idx.x;

    // Subpixel jitter for progressive antialiasing across accumulated frames.
    unsigned int seed = pcg(pixel ^ pcg(params.subframe + 1u));
    float jx = uintToUnitFloat(seed);
    float jy = uintToUnitFloat(pcg(seed));

    float dx = 2.0f * ((float)idx.x + jx) / (float)dim.x - 1.0f;
    float dy = 2.0f * ((float)idx.y + jy) / (float)dim.y - 1.0f;

    float3 origin = params.cam_eye;
    float3 dir = normalize(params.cam_u * dx + params.cam_v * dy + params.cam_w);

    // Trace the cloth GAS.
    unsigned int p0 = __float_as_uint(1e30f);  // t (1e30 == miss)
    unsigned int p1 = 0u, p2 = 0u, p3 = 0u;    // normal
    unsigned int p4 = 0u, p5 = 0u, p6 = 0u;    // previous-frame hit position
    optixTrace(
            params.handle, origin, dir,
            0.0f, 1e16f, 0.0f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2, p3, p4, p5, p6);
    float t_cloth = __uint_as_float(p0);
    float3 n_cloth = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
    float3 prev_hit_cloth = make_float3(__uint_as_float(p4), __uint_as_float(p5), __uint_as_float(p6));

    // Analytic colliders.
    float t_sphere = intersectSphere(origin, dir, params.sphere_center, params.sphere_radius);
    float t_ground = intersectGround(origin, dir, params.ground_y);

    // Nearest of {cloth, sphere, ground, miss}.
    const float T_MISS = 1e29f;
    float best = T_MISS;
    int which = 0;  // 0 miss, 1 cloth, 2 sphere, 3 ground
    if (t_cloth < best) { best = t_cloth; which = 1; }
    if (t_sphere > 0.0f && t_sphere < best) { best = t_sphere; which = 2; }
    if (t_ground > 0.0f && t_ground < best) { best = t_ground; which = 3; }

    float3 color;
    float3 aov_albedo;              // denoiser albedo guide (background := sky)
    float3 aov_normal;             // denoiser normal guide, view space, +z -> camera
    if (which == 0)
    {
        color = skyColor(dir);
        aov_albedo = color;
        aov_normal = make_float3(0.0f, 0.0f, 0.0f);
    }
    else
    {
        float3 hit = origin + dir * best;
        float3 n;
        float3 albedo;
        if (which == 1)
        {
            n = normalize(n_cloth);
            // Front faces the camera hemisphere; pick front/back albedo.
            albedo = (dot(n_cloth, dir) < 0.0f) ? params.cloth_albedo_front
                                                : params.cloth_albedo_back;
        }
        else if (which == 2)
        {
            n = normalize(hit - params.sphere_center);
            albedo = params.sphere_albedo;
        }
        else
        {
            n = make_float3(0.0f, 1.0f, 0.0f);
            albedo = params.ground_albedo;
        }
        // Face the normal toward the viewer, then shade + emit guides with it.
        float3 nf = (dot(n, dir) > 0.0f) ? (n * -1.0f) : n;
        color = shade(nf, albedo);
        aov_albedo = albedo;
        // View-space normal: project onto the (normalized) camera basis, with the
        // forward axis negated so +z points back toward the camera.
        float3 uh = normalize(params.cam_u);
        float3 vh = normalize(params.cam_v);
        float3 wh = normalize(params.cam_w);
        aov_normal = make_float3(dot(nf, uh), dot(nf, vh), dot(nf, wh * -1.0f));
    }

    // Firefly clamp: cap the per-sample radiance luminance before it enters the
    // accumulator so a single bright outlier can't dominate the mean.
    float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
    if (lum > PT_MAX_RADIANCE)
        color = color * (PT_MAX_RADIANCE / lum);

    // Progressive accumulation.
    float4 prev = (params.subframe == 0u) ? make_float4(0.0f, 0.0f, 0.0f, 0.0f)
                                          : params.accum[pixel];
    float4 acc = make_float4(prev.x + color.x, prev.y + color.y, prev.z + color.z, 1.0f);
    params.accum[pixel] = acc;
    float inv = 1.0f / (float)(params.subframe + 1u);
    params.output[pixel] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 1.0f);

    // Guide AOVs are deterministic per pixel; overwrite each subframe (the small
    // sub-pixel jitter only perturbs them at silhouette edges, which is fine for
    // a denoiser guide). No accumulation needed.
    params.albedo[pixel] = make_float4(aov_albedo.x, aov_albedo.y, aov_albedo.z, 1.0f);
    params.normal[pixel] = make_float4(aov_normal.x, aov_normal.y, aov_normal.z, 0.0f);

    // Motion-vector (flow) AOV: the OptiX temporal denoiser wants, at the current
    // pixel, the vector (current - previous) in input-resolution pixels, so it can
    // recover the source pixel as (current - flow). We find where the surface seen
    // here *was* one frame ago (prev_pix), in the same pixel convention the raygen
    // uses (x right, y up from the bottom row -- self-consistent with the beauty /
    // previousOutput buffers, so the denoiser's reprojection lands correctly).
    // The host zeroes history on the first frame (temporalModeUsePreviousLayers =
    // 0), so a bogus flow then is harmless; we still emit a sane value.
    float2 curr_pix = make_float2((float)idx.x + 0.5f, (float)idx.y + 0.5f);
    float2 prev_pix = curr_pix;  // default: zero flow (static / reprojection failed)
    float2 pp;
    if (which == 0)
    {
        // Sky: reproject the ray direction through the previous camera basis.
        if (projectDirToPixel(dir, params.prev_cam_u, params.prev_cam_v,
                              params.prev_cam_w, params.width, params.height, pp))
            prev_pix = pp;
    }
    else
    {
        float3 P_curr = origin + dir * best;
        float3 P_prev;
        if (which == 1)
            P_prev = prev_hit_cloth;                       // barycentric on prev verts
        else if (which == 2)
            P_prev = P_curr + (params.sphere_center_prev - params.sphere_center);  // rigid
        else
            P_prev = P_curr;                               // static ground
        if (projectToPixel(P_prev, params.prev_cam_eye, params.prev_cam_u,
                           params.prev_cam_v, params.prev_cam_w,
                           params.width, params.height, pp))
            prev_pix = pp;
    }
    params.flow[pixel] = make_float2(curr_pix.x - prev_pix.x, curr_pix.y - prev_pix.y);
}

extern "C" __global__ void __miss__ms()
{
    // Leave t = 1e30 (set in raygen) so the miss is resolved there against sky.
    optixSetPayload_0(__float_as_uint(1e30f));
}

extern "C" __global__ void __closesthit__ch()
{
    // Geometric normal from the triangle vertices (needs the GAS built with
    // ALLOW_RANDOM_VERTEX_ACCESS). Smooth shading normals arrive in M4.
    const unsigned int prim = optixGetPrimitiveIndex();
    const unsigned int sbtIdx = optixGetSbtGASIndex();
    float3 v[3];
    optixGetTriangleVertexData(params.handle, prim, sbtIdx, 0.0f, v);
    float3 n = normalize(cross(v[1] - v[0], v[2] - v[0]));

    // Previous-frame world position of this exact surface point: reuse the hit's
    // barycentrics against the *previous* frame's vertex positions (same topology,
    // same triangle index -- only the vertices moved). This is the cloth's true
    // per-vertex motion, the term a rigid/camera reprojection cannot capture.
    float2 bc = optixGetTriangleBarycentrics();
    uint3 tri = params.tri_indices[prim];
    float3 p0 = params.prev_vertices[tri.x];
    float3 p1 = params.prev_vertices[tri.y];
    float3 p2 = params.prev_vertices[tri.z];
    float3 pPrev = p0 * (1.0f - bc.x - bc.y) + p1 * bc.x + p2 * bc.y;

    setHitPayload(optixGetRayTmax(), n, pPrev);
}
