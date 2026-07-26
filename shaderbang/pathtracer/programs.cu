// Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
// SPDX-License-Identifier: MIT
//
// Portions of the Disney BSDF below are ported from GLSL-PathTracer
// (https://github.com/knightcrawler25/GLSL-PathTracer), MIT License,
// Copyright (c) 2019 Asif Ali. See docs/pathtracer.md.
//
// OptiX device programs for the shaderbang path tracer (milestone M4c).
//
// The single-hit Lambert of M1/M3 is replaced by an iterative multi-bounce path
// tracer in the raygen program. Each bounce intersects the cloth GAS plus the
// analytic sphere/ground, evaluates a ported Disney principled BSDF (smooth
// per-vertex cloth normals, two-sided cloth), importance-samples a new
// direction, applies Russian roulette, and accumulates global illumination.
//
// M4b added next-event estimation for a shadow-tested directional sun (a delta
// light, so NEE-only -- no MIS, and a BSDF-sampled ray can never hit it). M4c
// adds an optional HDR lat-long environment map with importance sampling and
// multiple importance sampling (MIS, power heuristic) between the env-sampling
// and BSDF-sampling strategies: directLight() gathers the env-NEE half
// (PowerHeuristic(envPdf, bsdfPdf)) and the raygen miss handler gathers the
// BSDF-sampling half (PowerHeuristic(bsdfPdf, envPdf)), with the BSDF pdf of the
// last bounce carried across as a scalar. When no env map is bound the analytic
// gradient sky stays the default and is BSDF-sampling only (env pdf = 0), exactly
// as in M4b. The env CDF is sin(theta)-weighted on the host, so the device pdf
// carries no 1/sin(theta) pole singularity. Shadow rays reuse a second miss
// program (__miss__shadow) with closesthit disabled and terminate-on-first-hit,
// plus cheap analytic occluder tests. The albedo/normal/flow guide AOVs and
// motion-vector reprojection are preserved byte-compatibly for the temporal
// denoiser -- they are driven by the *first* (primary) hit exactly as before,
// with the normal guide using the smooth shading normal.
//
// The Params struct below MUST match PARAMS_DTYPE in renderer.py field for
// field (a -D PARAMS_EXPECTED_SIZE static_assert guards the ABI). Self-contained:
// only <optix.h>, with hand-rolled float3 math (no vec_math.h) so NVRTC needs
// just the OptiX + CUDA include dirs.

#include <optix.h>

// --------------------------------------------------------------------------- //
// float3 math (hand-rolled -- no vec_math.h)
// --------------------------------------------------------------------------- //
static __forceinline__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __forceinline__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __forceinline__ __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __forceinline__ __device__ float3 operator*(float3 a, float s)  { return make_float3(a.x * s, a.y * s, a.z * s); }
static __forceinline__ __device__ float3 operator*(float s, float3 a)  { return make_float3(a.x * s, a.y * s, a.z * s); }
static __forceinline__ __device__ float3 operator-(float3 a)           { return make_float3(-a.x, -a.y, -a.z); }
static __forceinline__ __device__ float3 operator/(float3 a, float s)  { float inv = 1.0f / s; return make_float3(a.x * inv, a.y * inv, a.z * inv); }
static __forceinline__ __device__ void   operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
static __forceinline__ __device__ void   operator*=(float3& a, float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; }
static __forceinline__ __device__ void   operator*=(float3& a, float s)  { a.x *= s; a.y *= s; a.z *= s; }
static __forceinline__ __device__ float  dot(float3 a, float3 b)       { return a.x * b.x + a.y * b.y + a.z * b.z; }
static __forceinline__ __device__ float3 cross(float3 a, float3 b)     { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static __forceinline__ __device__ float3 normalize(float3 v)          { float inv = rsqrtf(dot(v, v)); return v * inv; }
// Zero-safe normalize: a degenerate (near-zero-length) input would give
// rsqrtf(0)=+Inf and 0*Inf=NaN, so fall back to a supplied unit vector instead.
static __forceinline__ __device__ float3 safeNormalize(float3 v, float3 fallback)
{
    float l2 = dot(v, v);
    return (l2 > 1e-20f) ? v * rsqrtf(l2) : fallback;
}
static __forceinline__ __device__ bool finite3(float3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}
static __forceinline__ __device__ float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
static __forceinline__ __device__ float  clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
static __forceinline__ __device__ float  mixf(float a, float b, float t) { return a + t * (b - a); }
static __forceinline__ __device__ float3 splat(float s)               { return make_float3(s, s, s); }
static __forceinline__ __device__ float3 powf3(float3 a, float e)     { return make_float3(powf(a.x, e), powf(a.y, e), powf(a.z, e)); }
static __forceinline__ __device__ float  luminance(float3 c)          { return 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z; }
static __forceinline__ __device__ float3 reflectf(float3 i, float3 n) { return i - n * (2.0f * dot(n, i)); }
// GLSL refract semantics, including the k<0 total-internal-reflection case that
// returns the zero vector (disneySample normalizes the result, so this matters).
static __forceinline__ __device__ float3 refractf(float3 i, float3 n, float eta)
{
    float ni = dot(n, i);
    float k = 1.0f - eta * eta * (1.0f - ni * ni);
    if (k < 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    return i * eta - n * (eta * ni + sqrtf(k));
}

#define PT_PI          3.14159265358979323f
#define PT_INV_PI      0.31830988618379067f
#define PT_TWO_PI      6.28318530717958648f
#define PT_INV_TWO_PI  0.15915494309189533f

// Orthonormal basis around a unit normal (X=T, Y=B, Z=N), matching GLSL Onb.
static __forceinline__ __device__ void onb(float3 n, float3& t, float3& b)
{
    float3 up = fabsf(n.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f)
                                        : make_float3(1.0f, 0.0f, 0.0f);
    t = normalize(cross(up, n));
    b = cross(n, t);
}
static __forceinline__ __device__ float3 toLocal(float3 x, float3 y, float3 z, float3 v)
{
    return make_float3(dot(v, x), dot(v, y), dot(v, z));
}
static __forceinline__ __device__ float3 toWorld(float3 x, float3 y, float3 z, float3 v)
{
    return x * v.x + y * v.y + z * v.z;
}

// --------------------------------------------------------------------------- //
// Per-object material table entry (M7a). The scene's materials live in a device
// buffer indexed by material_id; makeMaterial reads this and fills the full
// Disney `Material` (below) with constant defaults for the rarely-varied
// parameters. 8 floats == 32 bytes, no internal padding (float3 has 4-byte
// alignment), so it packs identically on host (see renderer._materials_to_device).
// --------------------------------------------------------------------------- //
struct GpuMaterial
{
    float3 baseColor;      // front-face albedo
    float3 baseColorBack;  // back-face albedo (== baseColor for one-sided)
    float  roughness;      // used directly as the GGX alpha (Disney linear roughness)
    float  metallic;
};

// --------------------------------------------------------------------------- //
// Analytic light table entry (M7b). Both light types are *delta* lights:
// next-event-estimated only (a BSDF-sampled ray has zero probability of hitting
// an infinitesimal light), so directLight gathers them with misWeight = 1 and no
// /pdf -- exactly the M4b sun treatment, now generalized to a loop over a device
// buffer. 8 floats == 32 bytes, no padding (float3 has 4-byte alignment); the
// host packs the uint ``type`` into the first 4 bytes bit-for-bit (see
// renderer._lights_to_device).
// --------------------------------------------------------------------------- //
struct Light
{
    unsigned int type;       // 0 = directional, 1 = point
    float3       dir_or_pos; // directional: normalized dir *toward* the light; point: world position
    float3       color;      // radiance (directional) / intensity (point, before 1/dist^2)
    float        radius;     // reserved for soft/area falloff (unused; delta lights for now)
};

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
    uint3*                 tri_indices;   // triangle vertex-index triplets (prev-vertex / normal lookup)
    float2*                flow;          // output motion-vector AOV (input res, curr -> prev in pixels)
    OptixTraversableHandle handle;        // cloth GAS
    float3*                cloth_normals; // per-vertex smooth normals (0 => fall back to geometric normal)
    float4*                env_data;      // HDR lat-long env, row-major (v*W+u); 0/env_enabled=0 => analytic sky
    float*                 env_cdf;       // flat sin(theta)-weighted running-sum CDF (W*H)
    GpuMaterial*           materials;     // material table indexed by material_id (M7a)
    Light*                 lights;        // analytic delta-light table (M7b)

    // --- 4-byte scalars --- //
    unsigned int           width;         // input (render) width
    unsigned int           height;        // input (render) height
    unsigned int           subframe;      // 0 resets the accumulator
    unsigned int           max_depth;     // maximum number of bounces
    unsigned int           rr_depth;      // Russian roulette starts at this bounce
    float                  exposure;      // unused on device (tonemap is a Warp kernel)
    unsigned int           env_width;     // env-map width in texels (0 when no env)
    unsigned int           env_height;    // env-map height in texels
    unsigned int           env_enabled;   // 1 => importance-sample the env map + MIS
    unsigned int           num_materials; // number of entries in materials[] (M7a)
    unsigned int           cloth_material;  // material_id of the cloth mesh (transient; removed in M7d)
    unsigned int           sphere_material; // material_id of the analytic sphere (transient; removed in M7c)
    unsigned int           ground_material; // material_id of the analytic ground (transient; removed in M7c)
    unsigned int           num_lights;    // number of entries in lights[] (M7b)

    // --- float3 basis / colors (float3 has 4-byte alignment) --- //
    float3                 cam_eye;
    float3                 cam_u;
    float3                 cam_v;
    float3                 cam_w;

    float3                 prev_cam_eye;  // previous-frame camera (for motion-vector reprojection)
    float3                 prev_cam_u;
    float3                 prev_cam_v;
    float3                 prev_cam_w;

    float3                 sky_top;
    float3                 sky_bottom;

    float3                 sphere_center;
    float                  sphere_radius;

    float3                 sphere_center_prev; // previous-frame sphere center (rigid motion)
    float                  ground_y;

    // --- environment map scalars (M4c) --- //
    float                  env_total_sum; // sum of the sin(theta)-weighted CDF weights
    float                  env_intensity; // multiplier applied to env radiance
    float                  env_rotation;  // azimuth offset in uv (u += env_rotation)
};

#ifdef PARAMS_EXPECTED_SIZE
// ABI guard: the host builds PARAMS_DTYPE and passes its itemsize as a -D define.
// A mismatch here is silent memory corruption at launch, so fail the NVRTC
// compile instead. Keep struct Params and _PARAMS_NAMES/_PARAMS_FORMATS in sync.
static_assert(sizeof(Params) == PARAMS_EXPECTED_SIZE,
              "Params size != PARAMS_DTYPE.itemsize (renderer.py ABI drift)");
#endif

// Firefly clamp on whole-path radiance luminance (applied once, after the bounce
// loop, so it does not bias per-bounce GI): caps a single bright outlier before
// it enters the accumulator.
#define PT_MAX_RADIANCE 64.0f

extern "C" {
__constant__ Params params;
}

// --------------------------------------------------------------------------- //
// RNG: per-pixel PCG stream, one draw per random. Seeded from (pixel, subframe)
// only, so every subframe is a deterministic, decorrelated sample -> the
// progressive mean is an unbiased estimator and the temporal denoiser sees a
// stable jitter distribution.
// --------------------------------------------------------------------------- //
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

struct RNG { unsigned int state; };

static __forceinline__ __device__ float rng_next(RNG& r)
{
    r.state = r.state * 747796405u + 2891336453u;
    unsigned int w = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    w = (w >> 22u) ^ w;
    return uintToUnitFloat(w);
}

// --------------------------------------------------------------------------- //
// Camera reprojection helpers (for the motion-vector AOV) -- unchanged from M3.
// --------------------------------------------------------------------------- //
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

// --------------------------------------------------------------------------- //
// Analytic intersections + sky
// --------------------------------------------------------------------------- //
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

// --------------------------------------------------------------------------- //
// Disney principled BSDF -- ported from GLSL-PathTracer (MIT (c) 2019 Asif Ali).
// The lobe helpers operate in the local shading frame (Z = normal), so cosines
// are the .z components and anisotropic tangent projections are .x/.y.
// --------------------------------------------------------------------------- //
static __forceinline__ __device__ float schlickW(float u)
{
    float m = clampf(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m;
}

static __forceinline__ __device__ float dielectricFresnel(float cosThetaI, float eta)
{
    float s = eta * eta * (1.0f - cosThetaI * cosThetaI);
    if (s > 1.0f)
        return 1.0f;                               // total internal reflection
    float cosThetaT = sqrtf(fmaxf(1.0f - s, 0.0f));
    float rs = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);
    float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    return 0.5f * (rs * rs + rp * rp);
}

static __forceinline__ __device__ float gtr1(float NDotH, float a)
{
    if (a >= 1.0f)
        return PT_INV_PI;
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return (a2 - 1.0f) / (PT_PI * logf(a2) * t);
}

static __forceinline__ __device__ float3 sampleGTR1(float rgh, float r1, float r2)
{
    float a = fmaxf(0.001f, rgh);
    float a2 = a * a;
    float phi = r1 * PT_TWO_PI;
    float cosT = sqrtf((1.0f - powf(a2, 1.0f - r2)) / (1.0f - a2));
    float sinT = clampf(sqrtf(1.0f - cosT * cosT), 0.0f, 1.0f);
    return make_float3(sinT * cosf(phi), sinT * sinf(phi), cosT);
}

static __forceinline__ __device__ float3 sampleGGXVNDF(float3 V, float ax, float ay, float r1, float r2)
{
    float3 Vh = normalize(make_float3(ax * V.x, ay * V.y, V.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(lensq)
                             : make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);
    float r = sqrtf(r1);
    float phi = PT_TWO_PI * r2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
    float3 Nh = T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
    return normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)));
}

static __forceinline__ __device__ float gtr2Aniso(float NDotH, float HDotX, float HDotY, float ax, float ay)
{
    float a = HDotX / ax;
    float b = HDotY / ay;
    float c = a * a + b * b + NDotH * NDotH;
    return 1.0f / (PT_PI * ax * ay * c * c);
}

static __forceinline__ __device__ float smithG(float NDotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotV * NDotV;
    return (2.0f * NDotV) / (NDotV + sqrtf(a + b - a * b));
}

static __forceinline__ __device__ float smithGAniso(float NDotV, float VDotX, float VDotY, float ax, float ay)
{
    float a = VDotX * ax;
    float b = VDotY * ay;
    float c = NDotV;
    return (2.0f * NDotV) / (NDotV + sqrtf(a * a + b * b + c * c));
}

static __forceinline__ __device__ float3 cosineSampleHemisphere(float r1, float r2)
{
    float r = sqrtf(r1);
    float phi = PT_TWO_PI * r2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return make_float3(x, y, sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y)));
}

struct Material
{
    float3 baseColor;
    float  metallic, roughness, subsurface;
    float  specularTint, sheen, sheenTint;
    float  clearcoat, clearcoatRoughness, specTrans, ior, anisotropic;
    float  ax, ay;
};

static __forceinline__ __device__ void tintColors(const Material& m, float eta,
        float& F0, float3& Csheen, float3& Cspec0)
{
    float lum = luminance(m.baseColor);
    float3 ctint = lum > 0.0f ? m.baseColor / lum : splat(1.0f);
    F0 = (1.0f - eta) / (1.0f + eta);
    F0 *= F0;
    Cspec0 = lerp(splat(1.0f), ctint, m.specularTint) * F0;
    Csheen = lerp(splat(1.0f), ctint, m.sheenTint);
}

static __forceinline__ __device__ float3 evalDiffuse(const Material& mat, float3 Csheen,
        float3 V, float3 L, float3 H, float& pdf)
{
    pdf = 0.0f;
    if (L.z <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    float LDotH = dot(L, H);
    float Rr = 2.0f * mat.roughness * LDotH * LDotH;

    float FL = schlickW(L.z);
    float FV = schlickW(V.z);
    float Fretro = Rr * (FL + FV + FL * FV * (Rr - 1.0f));
    float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);

    float Fss90 = 0.5f * Rr;
    float Fss = mixf(1.0f, Fss90, FL) * mixf(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.0f / (L.z + V.z) - 0.5f) + 0.5f);

    float FH = schlickW(LDotH);
    float3 Fsheen = Csheen * (FH * mat.sheen);

    pdf = L.z * PT_INV_PI;
    return mat.baseColor * (PT_INV_PI * mixf(Fd + Fretro, ss, mat.subsurface)) + Fsheen;
}

static __forceinline__ __device__ float3 evalMicroReflect(const Material& mat,
        float3 V, float3 L, float3 H, float3 F, float& pdf)
{
    pdf = 0.0f;
    if (L.z <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    float D = gtr2Aniso(H.z, H.x, H.y, mat.ax, mat.ay);
    float G1 = smithGAniso(fabsf(V.z), V.x, V.y, mat.ax, mat.ay);
    float G2 = G1 * smithGAniso(fabsf(L.z), L.x, L.y, mat.ax, mat.ay);
    pdf = G1 * D / (4.0f * V.z);
    return F * (D * G2 / (4.0f * L.z * V.z));
}

static __forceinline__ __device__ float3 evalMicroRefract(const Material& mat, float eta,
        float3 V, float3 L, float3 H, float3 F, float& pdf)
{
    pdf = 0.0f;
    if (L.z >= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    float LDotH = dot(L, H);
    float VDotH = dot(V, H);
    float D = gtr2Aniso(H.z, H.x, H.y, mat.ax, mat.ay);
    float G1 = smithGAniso(fabsf(V.z), V.x, V.y, mat.ax, mat.ay);
    float G2 = G1 * smithGAniso(fabsf(L.z), L.x, L.y, mat.ax, mat.ay);
    float denom = LDotH + VDotH * eta;
    denom *= denom;
    float eta2 = eta * eta;
    float jacobian = fabsf(LDotH) / denom;
    pdf = G1 * fmaxf(0.0f, VDotH) * D * jacobian / V.z;
    float3 oneMinusF = make_float3(1.0f - F.x, 1.0f - F.y, 1.0f - F.z);
    return powf3(mat.baseColor, 0.5f) * oneMinusF
           * (D * G2 * fabsf(VDotH) * jacobian * eta2 / fabsf(L.z * V.z));
}

static __forceinline__ __device__ float3 evalClearcoat(const Material& mat,
        float3 V, float3 L, float3 H, float& pdf)
{
    pdf = 0.0f;
    if (L.z <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    float VDotH = dot(V, H);
    float F = mixf(0.04f, 1.0f, schlickW(VDotH));
    float D = gtr1(H.z, mat.clearcoatRoughness);
    float G = smithG(L.z, 0.25f) * smithG(V.z, 0.25f);
    float jacobian = 1.0f / (4.0f * VDotH);
    pdf = D * H.z * jacobian;
    float fdg = F * D * G;
    return make_float3(fdg, fdg, fdg);
}

// Evaluate the BSDF value f (cosine folded in) and its solid-angle pdf for a
// world-space (V, N, L) triple pointing away from the surface.
static __device__ float3 disneyEval(const Material& mat, float eta,
        float3 V, float3 N, float3 L, float& pdf)
{
    pdf = 0.0f;
    float3 f = make_float3(0.0f, 0.0f, 0.0f);

    float3 T, B;
    onb(N, T, B);
    V = toLocal(T, B, N, V);
    L = toLocal(T, B, N, L);

    float3 H = (L.z > 0.0f) ? normalize(L + V) : normalize(L + V * eta);
    if (H.z < 0.0f)
        H = -H;

    float F0;
    float3 Csheen, Cspec0;
    tintColors(mat, eta, F0, Csheen, Cspec0);

    float dielectricWt = (1.0f - mat.metallic) * (1.0f - mat.specTrans);
    float metalWt = mat.metallic;
    float glassWt = (1.0f - mat.metallic) * mat.specTrans;

    float schlickWt = schlickW(V.z);
    float diffPr = dielectricWt * luminance(mat.baseColor);
    float dielectricPr = dielectricWt * luminance(lerp(Cspec0, splat(1.0f), schlickWt));
    float metalPr = metalWt * luminance(lerp(mat.baseColor, splat(1.0f), schlickWt));
    float glassPr = glassWt;
    float clearCtPr = 0.25f * mat.clearcoat;

    float inv = 1.0f / (diffPr + dielectricPr + metalPr + glassPr + clearCtPr);
    diffPr *= inv; dielectricPr *= inv; metalPr *= inv; glassPr *= inv; clearCtPr *= inv;

    bool isReflect = L.z * V.z > 0.0f;
    float tmp;
    float VDotH = fabsf(dot(V, H));

    if (diffPr > 0.0f && isReflect)
    {
        f += evalDiffuse(mat, Csheen, V, L, H, tmp) * dielectricWt;
        pdf += tmp * diffPr;
    }
    if (dielectricPr > 0.0f && isReflect)
    {
        float Fr = (dielectricFresnel(VDotH, 1.0f / mat.ior) - F0) / (1.0f - F0);
        f += evalMicroReflect(mat, V, L, H, lerp(Cspec0, splat(1.0f), Fr), tmp) * dielectricWt;
        pdf += tmp * dielectricPr;
    }
    if (metalPr > 0.0f && isReflect)
    {
        float3 Fr = lerp(mat.baseColor, splat(1.0f), schlickW(VDotH));
        f += evalMicroReflect(mat, V, L, H, Fr, tmp) * metalWt;
        pdf += tmp * metalPr;
    }
    if (glassPr > 0.0f)
    {
        float Fr = dielectricFresnel(VDotH, eta);
        if (isReflect)
        {
            f += evalMicroReflect(mat, V, L, H, splat(Fr), tmp) * glassWt;
            pdf += tmp * glassPr * Fr;
        }
        else
        {
            f += evalMicroRefract(mat, eta, V, L, H, splat(Fr), tmp) * glassWt;
            pdf += tmp * glassPr * (1.0f - Fr);
        }
    }
    if (clearCtPr > 0.0f && isReflect)
    {
        f += evalClearcoat(mat, V, L, H, tmp) * (0.25f * mat.clearcoat);
        pdf += tmp * clearCtPr;
    }

    return f * fabsf(L.z);
}

// Importance-sample a new direction L (world space) and return f + pdf via a
// final disneyEval. Draws three uniforms from the per-pixel RNG stream.
static __device__ float3 disneySample(const Material& mat, float eta,
        float3 V, float3 N, float3& L, float& pdf, RNG& rng)
{
    pdf = 0.0f;
    float r1 = rng_next(rng);
    float r2 = rng_next(rng);

    float3 T, B;
    onb(N, T, B);
    float3 Vl = toLocal(T, B, N, V);

    float F0;
    float3 Csheen, Cspec0;
    tintColors(mat, eta, F0, Csheen, Cspec0);

    float dielectricWt = (1.0f - mat.metallic) * (1.0f - mat.specTrans);
    float metalWt = mat.metallic;
    float glassWt = (1.0f - mat.metallic) * mat.specTrans;

    float schlickWt = schlickW(Vl.z);
    float diffPr = dielectricWt * luminance(mat.baseColor);
    float dielectricPr = dielectricWt * luminance(lerp(Cspec0, splat(1.0f), schlickWt));
    float metalPr = metalWt * luminance(lerp(mat.baseColor, splat(1.0f), schlickWt));
    float glassPr = glassWt;
    float clearCtPr = 0.25f * mat.clearcoat;

    float inv = 1.0f / (diffPr + dielectricPr + metalPr + glassPr + clearCtPr);
    diffPr *= inv; dielectricPr *= inv; metalPr *= inv; glassPr *= inv; clearCtPr *= inv;

    float cdf0 = diffPr;
    float cdf1 = cdf0 + dielectricPr;
    float cdf2 = cdf1 + metalPr;
    float cdf3 = cdf2 + glassPr;

    float r3 = rng_next(rng);
    float3 Ll;
    if (r3 < cdf0)
    {
        Ll = cosineSampleHemisphere(r1, r2);
    }
    else if (r3 < cdf2)
    {
        float3 H = sampleGGXVNDF(Vl, mat.ax, mat.ay, r1, r2);
        if (H.z < 0.0f)
            H = -H;
        Ll = normalize(reflectf(-Vl, H));
    }
    else if (r3 < cdf3)
    {
        float3 H = sampleGGXVNDF(Vl, mat.ax, mat.ay, r1, r2);
        float Fr = dielectricFresnel(fabsf(dot(Vl, H)), eta);
        if (H.z < 0.0f)
            H = -H;
        r3 = (r3 - cdf2) / (cdf3 - cdf2);
        Ll = (r3 < Fr) ? normalize(reflectf(-Vl, H))
                       : normalize(refractf(-Vl, H, eta));
    }
    else
    {
        float3 H = sampleGTR1(mat.clearcoatRoughness, r1, r2);
        if (H.z < 0.0f)
            H = -H;
        Ll = normalize(reflectf(-Vl, H));
    }

    L = toWorld(T, B, N, Ll);
    return disneyEval(mat, eta, V, N, L, pdf);
}

// Build the material for an intersected object from the device material table
// (M7a): ``material_id`` indexes params.materials, and the rarely-varied Disney
// parameters keep constant defaults. ``roughness`` is used directly as the GGX
// alpha (Disney "linear roughness == alpha"), NOT alpha=roughness^2. ``front``
// selects the two-sided front/back albedo.
static __forceinline__ __device__ Material makeMaterial(int material_id, bool front)
{
    Material m;
    m.subsurface = 0.0f;
    m.specularTint = 0.0f;
    m.sheen = 0.0f;
    m.sheenTint = 0.5f;
    m.clearcoat = 0.0f;
    m.clearcoatRoughness = 0.03f;
    m.specTrans = 0.0f;
    m.ior = 1.5f;
    m.anisotropic = 0.0f;

    GpuMaterial g = params.materials[material_id];
    m.baseColor = front ? g.baseColor : g.baseColorBack;
    m.roughness = g.roughness;
    m.metallic = g.metallic;

    float aspect = sqrtf(1.0f - m.anisotropic * 0.9f);
    m.ax = fmaxf(1e-3f, m.roughness / aspect);
    m.ay = fmaxf(1e-3f, m.roughness * aspect);
    return m;
}

// --------------------------------------------------------------------------- //
// Hit payloads: p0 = t (float bits, 1e30 = miss); p1..p3 = geometric normal Ng;
// p4..p6 = smooth shading normal Ns; p7..p9 = previous-frame world position.
// --------------------------------------------------------------------------- //
static __forceinline__ __device__ void setHitPayload(float t, float3 ng, float3 ns, float3 pPrev)
{
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(__float_as_uint(ng.x));
    optixSetPayload_2(__float_as_uint(ng.y));
    optixSetPayload_3(__float_as_uint(ng.z));
    optixSetPayload_4(__float_as_uint(ns.x));
    optixSetPayload_5(__float_as_uint(ns.y));
    optixSetPayload_6(__float_as_uint(ns.z));
    optixSetPayload_7(__float_as_uint(pPrev.x));
    optixSetPayload_8(__float_as_uint(pPrev.y));
    optixSetPayload_9(__float_as_uint(pPrev.z));
}

struct Hit
{
    int    which;       // 0 miss, 1 cloth, 2 sphere, 3 ground (geometry dispatch)
    int    material_id; // index into params.materials (shading)
    float  t;
    float3 p;       // world hit position
    float3 ng;      // geometric normal (raw orientation)
    float3 ns;      // shading normal (smooth; == ng for analytic objects)
    float3 prevP;   // previous-frame world position (motion vectors)
};

// Trace the cloth GAS and the analytic sphere/ground, return the nearest hit.
static __forceinline__ __device__ Hit sceneIntersect(float3 o, float3 d)
{
    unsigned int p0 = __float_as_uint(1e30f);
    unsigned int p1 = 0u, p2 = 0u, p3 = 0u;
    unsigned int p4 = 0u, p5 = 0u, p6 = 0u;
    unsigned int p7 = 0u, p8 = 0u, p9 = 0u;
    optixTrace(
            params.handle, o, d,
            0.0f, 1e16f, 0.0f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    float t_cloth = __uint_as_float(p0);
    float t_s = intersectSphere(o, d, params.sphere_center, params.sphere_radius);
    float t_g = intersectGround(o, d, params.ground_y);

    Hit h;
    h.which = 0;
    float best = 1e29f;
    if (t_cloth < best) { best = t_cloth; h.which = 1; }
    if (t_s > 0.0f && t_s < best) { best = t_s; h.which = 2; }
    if (t_g > 0.0f && t_g < best) { best = t_g; h.which = 3; }
    if (h.which == 0)
        return h;

    h.t = best;
    h.p = o + d * best;
    if (h.which == 1)
    {
        h.material_id = (int)params.cloth_material;
        h.ng = normalize(make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3)));
        h.ns = normalize(make_float3(__uint_as_float(p4), __uint_as_float(p5), __uint_as_float(p6)));
        h.prevP = make_float3(__uint_as_float(p7), __uint_as_float(p8), __uint_as_float(p9));
    }
    else if (h.which == 2)
    {
        h.material_id = (int)params.sphere_material;
        float3 n = normalize(h.p - params.sphere_center);
        h.ng = n;
        h.ns = n;
        h.prevP = h.p + (params.sphere_center_prev - params.sphere_center);  // rigid
    }
    else
    {
        h.material_id = (int)params.ground_material;
        h.ng = make_float3(0.0f, 1.0f, 0.0f);
        h.ns = h.ng;
        h.prevP = h.p;                                                       // static
    }
    return h;
}

// --------------------------------------------------------------------------- //
// Shadow / occlusion query (M4b). A binary "is anything between (o) and the
// light in [eps, tmax]?" test, used by next-event estimation. Analytic occluders
// are cheap closed-form tests; the cloth GAS is traced with closesthit disabled
// and terminate-on-first-hit (anyhit is already disabled by the geometry flag),
// so the shadow trace does no shading work. The payload is pre-seeded "occluded"
// and only __miss__shadow (miss SBT index 1) clears it -- reached solely when the
// ray escapes to tmax unobstructed. This trace still issues from raygen (via
// directLight), so it is not recursive and maxTraceDepth stays 1.
// --------------------------------------------------------------------------- //
static __forceinline__ __device__ bool sceneOcclude(float3 o, float3 d, float tmax)
{
    float t_s = intersectSphere(o, d, params.sphere_center, params.sphere_radius);
    if (t_s > 1e-4f && t_s < tmax)
        return true;
    float t_g = intersectGround(o, d, params.ground_y);
    if (t_g > 1e-4f && t_g < tmax)
        return true;

    unsigned int occluded = 1u;   // assume blocked; __miss__shadow clears to 0
    optixTrace(
            params.handle, o, d,
            1e-4f, tmax, 0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0, 1, 1,   // SBToffset=0, SBTstride=1, missSBTIndex=1 (__miss__shadow)
            occluded);
    return occluded != 0u;
}

// --------------------------------------------------------------------------- //
// Environment map (M4c): optional HDR lat-long image, importance-sampled with a
// sin(theta)-weighted CDF built on the host (EnvironmentMap.cpp ported), plus
// MIS against BSDF sampling. Ported from GLSL-PathTracer envmap.glsl, adapted
// for plain device buffers with manual bilinear filtering (no CUDA texture
// objects) and the sin(theta) row weighting the reference omits (which improves
// importance sampling near the poles). The CDF weights bake sin(theta) at the
// texel-ROW center, so a texel is selected with probability
// P = L*sin(thetaCenter)/totalSum. The solid-angle pdf then divides by the
// lat-long Jacobian's sin(theta) at the SAMPLE's actual polar angle:
//   pdf_w = P * W*H / (2*pi^2 * sin(thetaActual)).
// The two sin(theta)s do NOT cancel (center vs. actual differ within a cell), so
// both are kept -- this is exactly pbrt's InfiniteAreaLight pdf and is unbiased
// (dropping 1/sin(thetaActual) biases the estimate by sin(thetaCenter)/
// sin(thetaActual), O(1) near the poles). The identical pdf-of-direction formula
// is evaluated on both the NEE and BSDF-sampling sides so the MIS weights share
// one measure.
// --------------------------------------------------------------------------- //
static __forceinline__ __device__ float powerHeuristic(float a, float b)
{
    // beta=2 power heuristic, one sample per strategy: a^2 / (a^2 + b^2).
    float t = a * a;
    return t / (b * b + t);
}

static __forceinline__ __device__ float3 envTexel(int x, int y)
{
    float4 c = params.env_data[y * (int)params.env_width + x];
    return make_float3(c.x, c.y, c.z);
}

// Direction -> lat-long uv (y-up, right-handed). u is wrapped to [0,1); v is the
// polar angle in [0,1]. Matches EvalEnvMap's forward map (incl. env_rotation).
static __forceinline__ __device__ void envDirToUV(float3 dir, float& u, float& v)
{
    float theta = acosf(clampf(dir.y, -1.0f, 1.0f));
    u = (PT_PI + atan2f(dir.z, dir.x)) * PT_INV_TWO_PI + params.env_rotation;
    u = u - floorf(u);                 // wrap azimuth into [0,1)
    v = theta * PT_INV_PI;             // [0,1]
}

// uv -> direction, the exact inverse of envDirToUV (SampleEnvMap's inverse map).
static __forceinline__ __device__ float3 envUVToDir(float u, float v)
{
    float phi = (u - params.env_rotation) * PT_TWO_PI;
    float theta = v * PT_PI;
    float st = sinf(theta);
    return make_float3(-st * cosf(phi), cosf(theta), -st * sinf(phi));
}

// Bilinear env color at uv (wrap u, clamp v). Filtering is for the *radiance*
// only; the pdf uses the discrete per-texel luminance (see below).
static __forceinline__ __device__ float3 envBilinear(float u, float v)
{
    int W = (int)params.env_width, H = (int)params.env_height;
    float fx = u * (float)W - 0.5f;
    float fy = v * (float)H - 0.5f;
    float x0f = floorf(fx), y0f = floorf(fy);
    float tx = fx - x0f, ty = fy - y0f;
    int x0 = (int)x0f, y0 = (int)y0f;
    int x0w = ((x0 % W) + W) % W;      // wrap azimuth
    int x1w = (((x0 + 1) % W) + W) % W;
    int y0c = min(max(y0, 0), H - 1);  // clamp poles
    int y1c = min(max(y0 + 1, 0), H - 1);
    float3 c00 = envTexel(x0w, y0c), c10 = envTexel(x1w, y0c);
    float3 c01 = envTexel(x0w, y1c), c11 = envTexel(x1w, y1c);
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

// Nearest-texel index for a uv (wrap u, clamp v) -- the texel whose cell the
// sample fell in, matching the discrete CDF the sampler draws from.
static __forceinline__ __device__ void envTexelIndex(float u, float v, int& x, int& y)
{
    int W = (int)params.env_width, H = (int)params.env_height;
    x = (int)floorf(u * (float)W);
    x = ((x % W) + W) % W;
    y = (int)floorf(v * (float)H);
    y = min(max(y, 0), H - 1);
}

// Solid-angle pdf of the env importance sampler for a sample that fell in texel
// (x,y) at actual polar angle theta (sinThetaActual = sin(theta)). Selection
// probability P = L*sin(thetaCenter)/env_total_sum (the host bakes
// sin(thetaCenter) into the CDF weights + env_total_sum); the uniform uv jitter
// within the cell gives pdf_uv = P*W*H; the lat-long Jacobian
// dw = 2*pi^2*sin(theta)*du*dv converts to solid angle. sin(thetaCenter) (the
// baked selection weight, per row) and 1/sin(theta) (the Jacobian, at the
// sample) do NOT cancel, so both are kept -- matching pbrt's InfiniteAreaLight
// and staying unbiased. Guards env_total_sum and the poles (sinThetaActual==0).
static __forceinline__ __device__ float envTexelPdf(int x, int y, float sinThetaActual)
{
    if (!(params.env_total_sum > 0.0f) || sinThetaActual <= 0.0f)
        return 0.0f;
    float thetaCenter = ((float)y + 0.5f) * PT_PI / (float)params.env_height;
    float L = luminance(envTexel(x, y));
    float wh = (float)params.env_width * (float)params.env_height;
    return L * sinf(thetaCenter) * wh
         / (params.env_total_sum * PT_TWO_PI * PT_PI * sinThetaActual);
}

// Env radiance + solid-angle pdf for an arbitrary direction (the BSDF-sampling
// side of the MIS: called from the raygen miss handler).
static __forceinline__ __device__ float3 envRadiance(float3 dir, float& pdf)
{
    float u, v;
    envDirToUV(dir, u, v);
    int x, y;
    envTexelIndex(u, v, x, y);
    pdf = envTexelPdf(x, y, sinf(v * PT_PI));   // sin(theta) at the query dir
    return envBilinear(u, v);
}

// Importance-sample the env map: pick a texel from the CDF, jitter uniformly
// within its uv cell (a proper continuous sampler, unlike the reference's
// texel-corner point sample), and return radiance + direction + solid-angle pdf.
// Draws exactly three uniforms so the RNG stream is deterministic per launch.
static __forceinline__ __device__ float3 sampleEnv(RNG& rng, float3& dir, float& pdf)
{
    int W = (int)params.env_width, H = (int)params.env_height;
    float value = rng_next(rng) * params.env_total_sum;

    // Hierarchical binary search over the flat sin(theta)-weighted CDF: the last
    // column is the marginal over rows, the running sum within a row is the
    // conditional over columns (envmap.glsl BinarySearch).
    int lower = 0, upper = H - 1;
    while (lower < upper)
    {
        int mid = (lower + upper) >> 1;
        if (value < params.env_cdf[(W - 1) + mid * W]) upper = mid;
        else                                           lower = mid + 1;
    }
    int y = min(max(lower, 0), H - 1);
    lower = 0; upper = W - 1;
    while (lower < upper)
    {
        int mid = (lower + upper) >> 1;
        if (value < params.env_cdf[mid + y * W]) upper = mid;
        else                                     lower = mid + 1;
    }
    int x = min(max(lower, 0), W - 1);

    float ju = rng_next(rng);
    float jv = rng_next(rng);
    float u = ((float)x + ju) / (float)W;
    float v = ((float)y + jv) / (float)H;
    dir = envUVToDir(u, v);
    pdf = envTexelPdf(x, y, sinf(v * PT_PI));   // same measure as envRadiance
    return envBilinear(u, v);
}

// Next-event estimation (M4b + M4c). Two independent, non-double-counting light
// contributions are gathered before the BSDF is sampled at this vertex:
//
//  * the directional sun -- a *delta* light: NEE is its only estimator (a
//    BSDF-sampled ray has zero probability of hitting an infinitesimal light),
//    so there is no MIS, no /pdf, and it is deliberately gathered nowhere else;
//  * the HDR env map (when bound) -- importance-sampled and MIS-weighted with
//    PowerHeuristic(envPdf, bsdfPdf); the BSDF-sampling half is gathered on a
//    miss in raygen. Skipped when no env is bound, leaving the analytic sky
//    BSDF-sampling only (env pdf = 0), so the sun still works with either.
//
// The BSDF value already folds |N.L| in. Draws RNG only for the env sample.
static __forceinline__ __device__ float3 directLight(
        float3 V, float3 N, float3 Ng, const Material& mat, float eta,
        float3 hitP, RNG& rng)
{
    float3 Ld = make_float3(0.0f, 0.0f, 0.0f);

    // --- Analytic delta lights (directional + point): NEE-only, full weight,
    // no MIS. Each is shadow-tested; a directional light is at infinity
    // (tmax = 1e16), a point light attenuates as 1/dist^2 and its shadow ray
    // stops just short of the light itself. ---
    for (unsigned int i = 0u; i < params.num_lights; ++i)
    {
        Light lt = params.lights[i];
        float3 L;      // direction toward the light
        float3 Li;     // incident radiance at the surface
        float  tmax;   // shadow-ray max distance
        if (lt.type == 0u)                 // directional
        {
            L = lt.dir_or_pos;             // normalized, toward the light
            Li = lt.color;
            tmax = 1e16f;
        }
        else                               // point
        {
            float3 toL = lt.dir_or_pos - hitP;
            float dist2 = dot(toL, toL);
            if (!(dist2 > 1e-12f))
                continue;
            float dist = sqrtf(dist2);
            L = toL / dist;
            Li = lt.color / dist2;         // inverse-square falloff
            tmax = dist - 1e-3f;           // stop before the light
        }

        float pdf;
        float3 f = disneyEval(mat, eta, V, N, L, pdf);   // |N.L| folded in
        // Skip the shadow ray if the BSDF vanishes (e.g. light below the horizon).
        if (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)
        {
            float3 offN = (dot(L, Ng) < 0.0f) ? -Ng : Ng;
            float3 so = hitP + offN * 1e-4f;
            if (!sceneOcclude(so, L, tmax))
                Ld += f * Li;
        }
    }

    // --- Environment map NEE (importance sampled + MIS). ---
    if (params.env_enabled)
    {
        float3 Lenv;
        float envPdf;
        float3 Li = sampleEnv(rng, Lenv, envPdf);
        if (envPdf > 0.0f)
        {
            float bsdfPdf;
            float3 f = disneyEval(mat, eta, V, N, Lenv, bsdfPdf);
            if (bsdfPdf > 0.0f && (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f))
            {
                float3 offN = (dot(Lenv, Ng) < 0.0f) ? -Ng : Ng;
                float3 so = hitP + offN * 1e-4f;
                if (!sceneOcclude(so, Lenv, 1e16f))
                {
                    float mis = powerHeuristic(envPdf, bsdfPdf);
                    Ld += (f * Li) * (mis * params.env_intensity / envPdf);
                }
            }
        }
    }

    return Ld;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int pixel = idx.y * params.width + idx.x;

    // Per-pixel RNG stream. The first two draws are the subpixel jitter for
    // progressive antialiasing; the rest feed the BSDF sampling / RR.
    RNG rng;
    rng.state = pcg(pixel ^ pcg(params.subframe + 1u));
    float jx = rng_next(rng);
    float jy = rng_next(rng);

    float dx = 2.0f * ((float)idx.x + jx) / (float)dim.x - 1.0f;
    float dy = 2.0f * ((float)idx.y + jy) / (float)dim.y - 1.0f;

    float3 o = params.cam_eye;
    float3 d = normalize(params.cam_u * dx + params.cam_v * dy + params.cam_w);
    const float3 d0 = d;   // primary direction, kept for the AOV / flow tail

    // ---- Iterative multi-bounce path tracing (all traces issue from raygen, so
    // maxTraceDepth stays 1; nothing traces from closesthit/miss). ----
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float bsdfPdf = 0.0f;   // solid-angle pdf of the last BSDF sample (MIS carry)
    Hit first;
    first.which = -1;

    for (unsigned int depth = 0u; depth <= params.max_depth; ++depth)
    {
        Hit h = sceneIntersect(o, d);
        if (depth == 0u)
            first = h;

        if (h.which == 0)
        {
            // Environment on a miss. With an HDR env bound this is the
            // BSDF-sampling half of the env MIS: weight it against the env-NEE
            // strategy with PowerHeuristic(bsdfPdf, envPdf), using the BSDF pdf
            // carried from the bounce that produced this ray. The primary ray
            // (depth 0) had no NEE competitor, so it takes full weight. With no
            // env bound the analytic sky takes full weight (BSDF-sampling only,
            // as in M4b). The delta sun is never gathered here (no double count).
            if (params.env_enabled)
            {
                float envPdf;
                float3 envCol = envRadiance(d, envPdf);
                float misWeight = (depth == 0u) ? 1.0f
                                                : powerHeuristic(bsdfPdf, envPdf);
                radiance += throughput * envCol * (params.env_intensity * misWeight);
            }
            else
            {
                radiance += throughput * skyColor(d);
            }
            break;
        }

        // Two-sided shading: face-forward both normals toward the viewer, and
        // decide front/back from the unambiguous geometric normal.
        bool front = dot(h.ng, d) < 0.0f;
        float3 Ng = front ? h.ng : -h.ng;
        float3 Ns = (dot(h.ns, d) > 0.0f) ? -h.ns : h.ns;
        Material mat = makeMaterial(h.material_id, front);
        float eta = front ? (1.0f / mat.ior) : mat.ior;   // relative IOR (entering -> 1/ior)

        // Next-event estimation (M4b sun + M4c env): shadow-tested directional
        // sun (NEE-only, delta light) plus importance-sampled env with MIS.
        radiance += throughput * directLight(-d, Ns, Ng, mat, eta, h.p, rng);

        if (depth == params.max_depth)
            break;

        // Sample the BSDF for the next bounce.
        float3 L;
        float pdf;
        float3 f = disneySample(mat, eta, -d, Ns, L, pdf, rng);
        if (!(pdf > 0.0f))
            break;
        throughput *= f * (1.0f / pdf);
        bsdfPdf = pdf;   // carry for the next vertex's env MIS (miss handler)

        // Offset along the geometric normal on the outgoing side to avoid
        // self-intersection, then continue.
        float3 offN = (dot(L, Ng) < 0.0f) ? -Ng : Ng;
        o = h.p + offN * 1e-4f;
        d = L;

        // Russian roulette (unbiased) once past rr_depth.
        if (depth >= params.rr_depth)
        {
            float q = fminf(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)) + 0.001f, 0.95f);
            if (rng_next(rng) > q)
                break;
            throughput *= (1.0f / q);
        }
    }

    // NaN sink (the firefly clamp below cannot catch NaN: NaN > x is false) +
    // whole-path firefly clamp on radiance luminance.
    if (!(isfinite(radiance.x) && isfinite(radiance.y) && isfinite(radiance.z)))
        radiance = make_float3(0.0f, 0.0f, 0.0f);
    float3 color = radiance;
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

    // ---- Guide AOVs + motion-vector (flow) AOV, driven by the primary hit
    // exactly as in M3 (only the normal guide now uses the smooth shading
    // normal, a strict improvement for the denoiser). Deterministic per pixel;
    // overwrite each subframe. ----
    float3 aov_albedo;
    float3 aov_normal;
    float2 curr_pix = make_float2((float)idx.x + 0.5f, (float)idx.y + 0.5f);
    float2 prev_pix = curr_pix;   // default: zero flow (static / reprojection failed)
    float2 pp;
    if (first.which <= 0)
    {
        // Sky (miss): reproject the primary direction through the previous camera.
        // The albedo guide uses the env color when bound, else the analytic sky.
        if (params.env_enabled)
        {
            float envPdfUnused;
            aov_albedo = envRadiance(d0, envPdfUnused);
        }
        else
        {
            aov_albedo = skyColor(d0);
        }
        aov_normal = make_float3(0.0f, 0.0f, 0.0f);
        if (projectDirToPixel(d0, params.prev_cam_u, params.prev_cam_v,
                              params.prev_cam_w, params.width, params.height, pp))
            prev_pix = pp;
    }
    else
    {
        bool front0 = dot(first.ng, d0) < 0.0f;
        aov_albedo = makeMaterial(first.material_id, front0).baseColor;
        // Face the shading normal toward the viewer for the view-space guide.
        float3 nf = (dot(first.ns, d0) > 0.0f) ? -first.ns : first.ns;
        float3 uh = normalize(params.cam_u);
        float3 vh = normalize(params.cam_v);
        float3 wh = normalize(params.cam_w);
        aov_normal = make_float3(dot(nf, uh), dot(nf, vh), dot(nf, -wh));
        if (projectToPixel(first.prevP, params.prev_cam_eye, params.prev_cam_u,
                           params.prev_cam_v, params.prev_cam_w,
                           params.width, params.height, pp))
            prev_pix = pp;
    }
    // A NaN in a guide AOV is undefined for the denoiser network and can smear
    // garbage across neighboring pixels, so sanitize both before the writes
    // (the beauty already has its own isfinite sink above).
    if (!finite3(aov_albedo))
        aov_albedo = make_float3(0.0f, 0.0f, 0.0f);
    if (!finite3(aov_normal))
        aov_normal = make_float3(0.0f, 0.0f, 0.0f);
    params.albedo[pixel] = make_float4(aov_albedo.x, aov_albedo.y, aov_albedo.z, 1.0f);
    params.normal[pixel] = make_float4(aov_normal.x, aov_normal.y, aov_normal.z, 0.0f);
    params.flow[pixel] = make_float2(curr_pix.x - prev_pix.x, curr_pix.y - prev_pix.y);
}

extern "C" __global__ void __miss__ms()
{
    // Leave t = 1e30 (set in sceneIntersect) so the miss is resolved there.
    optixSetPayload_0(__float_as_uint(1e30f));
}

extern "C" __global__ void __miss__shadow()
{
    // Reached only when a shadow ray escapes to tmax with no cloth hit: mark the
    // path to the light as unobstructed (sceneOcclude pre-seeds payload_0 = 1).
    optixSetPayload_0(0u);
}

extern "C" __global__ void __closesthit__ch()
{
    // Geometric normal from the triangle vertices (needs the GAS built with
    // ALLOW_RANDOM_VERTEX_ACCESS).
    const unsigned int prim = optixGetPrimitiveIndex();
    const unsigned int sbtIdx = optixGetSbtGASIndex();
    float3 v[3];
    optixGetTriangleVertexData(params.handle, prim, sbtIdx, 0.0f, v);
    // Zero-safe: a collapsed triangle (cross == 0) would otherwise normalize to
    // NaN and poison the denoiser normal guide (see safeNormalize).
    float3 ng = safeNormalize(cross(v[1] - v[0], v[2] - v[0]),
                              make_float3(0.0f, 1.0f, 0.0f));

    float2 bc = optixGetTriangleBarycentrics();
    float w0 = 1.0f - bc.x - bc.y;
    uint3 tri = params.tri_indices[prim];

    // Smooth shading normal from the cloth's per-vertex normals (barycentric
    // interpolation); fall back to the geometric normal when unavailable or when
    // the interpolated normal cancels to ~0 (opposing normals across a sharp
    // self-collision fold -- Warp's normalize_normals emits exactly (0,0,0)
    // there, which would make an unguarded normalize NaN).
    float3 ns = ng;
    if (params.cloth_normals != 0)
    {
        float3 s = params.cloth_normals[tri.x] * w0
                 + params.cloth_normals[tri.y] * bc.x
                 + params.cloth_normals[tri.z] * bc.y;
        ns = safeNormalize(s, ng);
    }

    // Previous-frame world position of this exact surface point: reuse the hit's
    // barycentrics against the *previous* frame's vertex positions (same
    // topology, same triangle index -- only the vertices moved).
    float3 pPrev = params.prev_vertices[tri.x] * w0
                 + params.prev_vertices[tri.y] * bc.x
                 + params.prev_vertices[tri.z] * bc.y;

    setHitPayload(optixGetRayTmax(), ng, ns, pPrev);
}
