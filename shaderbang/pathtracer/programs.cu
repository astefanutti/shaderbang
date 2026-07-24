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
    float4*                accum;     // HDR accumulator, width*height
    float4*                output;    // per-frame HDR (= accum / (subframe+1)), denoiser input
    float4*                albedo;    // guide AOV: per-pixel surface albedo
    float4*                normal;    // guide AOV: per-pixel view-space normal (+z toward camera)
    OptixTraversableHandle handle;    // cloth GAS

    unsigned int           width;
    unsigned int           height;
    unsigned int           subframe;  // 0 resets the accumulator
    float                  exposure;  // unused on device (tonemap is a Warp kernel)

    float3                 cam_eye;
    float3                 cam_u;
    float3                 cam_v;
    float3                 cam_w;

    float3                 light_dir;    // normalized, points toward the light
    float3                 light_color;
    float3                 sky_top;
    float3                 sky_bottom;

    float3                 sphere_center;
    float3                 sphere_albedo;
    float                  sphere_radius;

    float3                 ground_albedo;
    float                  ground_y;

    float3                 cloth_albedo_front;
    float3                 cloth_albedo_back;
};

extern "C" {
__constant__ Params params;
}

// Payloads: p0 = hit t (float bits), p1..p3 = geometric normal.
static __forceinline__ __device__ void setHitPayload(float t, float3 n)
{
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(__float_as_uint(n.x));
    optixSetPayload_2(__float_as_uint(n.y));
    optixSetPayload_3(__float_as_uint(n.z));
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
    optixTrace(
            params.handle, origin, dir,
            0.0f, 1e16f, 0.0f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2, p3);
    float t_cloth = __uint_as_float(p0);
    float3 n_cloth = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));

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
    setHitPayload(optixGetRayTmax(), n);
}
