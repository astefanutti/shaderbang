// Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
// SPDX-License-Identifier: MIT
//
// common.glsl -- pieces shared by every plasma-globe shader (GLSL 4.60, included by glutil.load_shader_source;
// it carries no #version of its own).
//
// Attribution
//   * pcg4d / rand, Luminance, DielectricFresnel: ported from GLSL-PathTracer, Copyright (c) 2019 Asif Ali,
//     https://github.com/knightcrawler25/GLSL-PathTracer (MIT).
//   * sphIntersect: Copyright (c) 2014 Inigo Quilez, https://iquilezles.org/articles/intersectors/ (MIT, as
//     published with the Shadertoy "Sphere - intersection" source).
//   * AgXToneMapping / agxDefaultContrastApprox and the Rec.2020 matrices: Copyright (c) 2010-2024 three.js
//     authors, src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js (MIT), itself after Filament
//     (Apache-2.0) and Blender's AgX; `toneMappingExposure` is applied by the caller here.
//   * capsuleEmission3 / capsuleEmission2: derived below.
//
// ---------------------------------------------------------------------------------------------------------------
// MIT License.  The copied works above are each distributed under the MIT License; the permission notice below
// accompanies, and applies to, every one of these copyright notices:
//
//   Copyright (c) 2019 Asif Ali                (GLSL-PathTracer: pcg4d, rand, Luminance, DielectricFresnel)
//   Copyright (c) 2014 Inigo Quilez            (sphIntersect)
//   Copyright (c) 2010-2024 three.js authors   (AgXToneMapping, agxDefaultContrastApprox, Rec.2020 matrices)
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without
//   limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
//   Software, and to permit persons to whom the Software is furnished to do so, subject to the following
//   conditions:
//
//   The above copyright notice and this permission notice shall be included in all copies or substantial portions
//   of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
//   TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
//   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.
// ---------------------------------------------------------------------------------------------------------------

#ifndef PLASMA_COMMON_GLSL
#define PLASMA_COMMON_GLSL

const float PI = 3.14159265358979323846;
const float INV_PI = 0.31830988618379067154;
const float TWO_PI = 6.28318530717958647692;

// Globe geometry (metres) and glass, plan section 4.1
const float GLOBE_R1 = 0.011;      // electrode bulb
const float GLOBE_R2I = 0.075;     // inner glass surface
const float GLOBE_R2O = 0.0775;    // outer glass surface
const float GLASS_IOR = 1.52;

// ---------------------------------------------------------------------------------------------------------------
// Pinhole camera shared with glutil.perspective_camera / camera_rays (numpy references use the same basis)
// ---------------------------------------------------------------------------------------------------------------
uniform ivec2 uRes;
uniform vec3 uEye;
uniform vec3 uFwd;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec2 uTanAspect;   // (tan(fov_y / 2), aspect)

vec3 cameraRay(ivec2 pix, vec2 jitter)
{
    vec2 ndc = (vec2(pix) + jitter) / vec2(uRes) * 2.0 - 1.0;
    return normalize(uFwd + ndc.x * uRight * uTanAspect.x * uTanAspect.y + ndc.y * uUp * uTanAspect.x);
}

// ---------------------------------------------------------------------------------------------------------------
// Random numbers [GLSL-PathTracer]: Jarzynski & Olano, "Hash Functions for GPU Rendering" (pcg4d)
// ---------------------------------------------------------------------------------------------------------------
void pcg4d(inout uvec4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> 16u);
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
}

float rand(inout uvec4 seed)
{
    pcg4d(seed);
    return float(seed.x) / float(0xffffffffu);
}

// ---------------------------------------------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------------------------------------------
// [GLSL-PathTracer]
float Luminance(vec3 c)
{
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

vec3 RGBToYCoCg(vec3 c)
{
    return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
                0.5 * c.r - 0.5 * c.b,
                -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

vec3 YCoCgToRGB(vec3 c)
{
    return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

vec3 LinearToSRGB(vec3 c)
{
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// ---------------------------------------------------------------------------------------------------------------
// Fresnel [GLSL-PathTracer] -- exact dielectric Fresnel (pbrt FrDielectric form); eta = n_incident / n_transmitted,
// the same convention as GLSL refract(); returns 1 under total internal reflection.
// ---------------------------------------------------------------------------------------------------------------
float DielectricFresnel(float cosThetaI, float eta)
{
    float sinThetaTSq = eta * eta * (1.0 - cosThetaI * cosThetaI);
    if (sinThetaTSq > 1.0)
        return 1.0;
    float cosThetaT = sqrt(max(1.0 - sinThetaTSq, 0.0));
    float rs = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);
    float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    return 0.5 * (rs * rs + rp * rp);
}

// ---------------------------------------------------------------------------------------------------------------
// Sphere intersector (iq) -- both roots, (-1,-1) on a miss
// ---------------------------------------------------------------------------------------------------------------
vec2 sphIntersect(vec3 ro, vec3 rd, vec3 ce, float ra)
{
    vec3 oc = ro - ce;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - ra * ra;
    float h = b * b - c;
    if (h < 0.0)
        return vec2(-1.0);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

// ---------------------------------------------------------------------------------------------------------------
// Capsule emission integrals
// ---------------------------------------------------------------------------------------------------------------
// A filament segment [A, B] radiates with a profile f(d) of the distance d from the point to the *segment*
// (hemispherical caps, i.e. a capsule).  Along the ray x(t) = ro + t rd, t in [t0, t1], the accumulated
// radiance is  I = int f(d(t)) dt.  Both profiles below have a closed form on each of the three ranges of t
// where the closest point of the segment is A (cap A), interior (body) or B (cap B).
//
// Body.  With u = (B - A)/L, w = ro - A, s(t) = (w + t rd).u = s0 + m t  (s0 = w.u, m = rd.u) the squared
// distance to the *line* is the perpendicular part of w + t rd:
//     d^2(t) = |w_perp + t rd_perp|^2 = k t^2 + 2 b t + c0,   w_perp = w - s0 u,  rd_perp = rd - m u,
//     k = |rd_perp|^2 = 1 - m^2,   b = w_perp . rd_perp,   c0 = |w_perp|^2.
// Caps.  The distance to the end point A is |w + t rd|^2 = t^2 + 2 (w.rd) t + |w|^2, i.e. k = 1,
//     b = w.rd, c0 = |w|^2 (and w' = ro - B for cap B).
// So in all three ranges q(t) = d^2(t) + eps^2 = k t^2 + 2 b t + c with c = c0 + eps^2 and
//     det = k c - b^2 = k (d_min^2 + eps^2) > 0.
//
// Profile 1/(d^2 + eps^2)^(3/2)  (chosen: algebraic, no transcendental):
//     d/dt [ (k t + b) / (det sqrt(q(t))) ] = [ k det sqrt(q) - (k t + b) det (k t + b)/sqrt(q) ] / (det^2 q)
//                                            = [ k q - (k t + b)^2 ] / (det q^(3/2)) = det / (det q^(3/2))
//                                            = q(t)^(-3/2)                                  (k q - (kt+b)^2 = det)
//     => int_ta^tb q^(-3/2) dt = [ (k t + b) / (det sqrt(q(t))) ]_ta^tb.
// With x = (k t + b)/sqrt(k) and a^2 = det/k this is the textbook x / (a^2 sqrt(x^2 + a^2)) up to 1/sqrt(k).
//
// Profile 1/(d^2 + eps^2)  (Cauchy, needs atan -- kept for the cost comparison):
//     int_ta^tb dt / q(t) = [ atan((k t + b)/sqrt(det)) / sqrt(det) ]_ta^tb.
//
// Ray parallel to the segment (k -> 0): b -> 0 as well (|b| <= |w_perp| sqrt(k)) and both forms become the
// difference of two large nearly-equal terms; below K_MIN the exact limit d = const is used instead:
//     int dt / (c)^(3/2) = (tb - ta) c^(-3/2),   int dt / c = (tb - ta) / c.
//
// Ranges.  s(t) in [0, L] is the body; s < 0 is cap A, s > L is cap B.  For m > 0 the body is
// [-s0/m, (L - s0)/m] and cap A lies below it; for m < 0 the interval flips and cap B lies below.  |m| is
// clamped away from zero so that s = const falls into the right single range without a branch.
//
// Normalisation.  A ray passing an infinite line at distance d receives 2/(d^2+eps^2) with profile 1 and
// pi/sqrt(d^2+eps^2) with profile 2; multiply by eps^2 (resp. eps) for a bounded, dimensionless response
// that peaks at 2 (resp. pi) on the axis with a halo of half-width ~eps.
//
// Precision.  q(t) is evaluated in metres around the segment; keep |w| small (re-base the ray origin at the
// globe entry point) so that eps >= 0.2 mm is resolved: the rounding floor of q is ~6e-8 * max(|w|^2, t^2).
//
// Degenerate segments.  |B - A|^2 is clamped to CAPSULE_L2_MIN before the division that forms u, so a
// zero-length segment (coinciding nodes, a zero-filled unused slot reaching a cell list) gives u = 0,
// s0 = m = 0: the "body" coefficients collapse to those of cap A (k = |rd|^2 = 1, b = w.rd, c = |w|^2 + eps^2),
// the two caps coincide, and the sum over the three ranges is the point-source integral over [t0, t1] --
// instead of the NaN a plain division would produce, which imageStore would hand to the TAAU history for good
// (clamp/mix do not scrub NaN).  Segments shorter than sqrt(CAPSULE_L2_MIN) = 0.1 um get the same treatment
// (their body range is shorter than 1 um, immaterial next to eps >= 0.2 mm).

const float CAPSULE_K_MIN = 1e-6;
const float CAPSULE_L2_MIN = 1e-14;

// int_ta^tb (k t^2 + 2 b t + c)^(-3/2) dt
float quadIntegral3(float k, float b, float c, float ta, float tb)
{
    if (k < CAPSULE_K_MIN)
        return (tb - ta) * inversesqrt(c) / c;
    float det = k * c - b * b;
    float fa = (k * ta + b) * inversesqrt(fma(fma(k, ta, 2.0 * b), ta, c));
    float fb = (k * tb + b) * inversesqrt(fma(fma(k, tb, 2.0 * b), tb, c));
    return (fb - fa) / det;
}

// int_ta^tb dt / (k t^2 + 2 b t + c)
float quadIntegral2(float k, float b, float c, float ta, float tb)
{
    if (k < CAPSULE_K_MIN)
        return (tb - ta) / c;
    float s = inversesqrt(k * c - b * b);
    return (atan((k * tb + b) * s) - atan((k * ta + b) * s)) * s;
}

// Splits [t0, t1] into cap A / body / cap B and returns the coefficient sets; shared by both profiles.
// Returns the body coefficients (k, b, c) and range, the below-body cap coefficients/range and the above-body ones.
struct CapsuleRanges
{
    vec3 body;   // k, b, c
    vec2 bodyT;
    vec3 lo;     // 1, b, c of the cap below the body
    vec2 loT;
    vec3 hi;     // 1, b, c of the cap above the body
    vec2 hiT;
};

CapsuleRanges capsuleRanges(vec3 ro, vec3 rd, float t0, float t1, vec3 a, vec3 b, float eps2)
{
    vec3 ab = b - a;
    float L = sqrt(max(dot(ab, ab), CAPSULE_L2_MIN));   // never 0: see "Degenerate segments" above
    vec3 u = ab / L;
    vec3 w = ro - a;
    float s0 = dot(w, u);
    float m = dot(rd, u);
    vec3 wp = w - s0 * u;
    vec3 rp = rd - m * u;

    CapsuleRanges r;
    r.body = vec3(dot(rp, rp), dot(wp, rp), dot(wp, wp) + eps2);

    vec3 wb = ro - b;
    vec3 capA = vec3(1.0, dot(w, rd), dot(w, w) + eps2);
    vec3 capB = vec3(1.0, dot(wb, rd), dot(wb, wb) + eps2);

    float ms = (m >= 0.0) ? max(m, 1e-7) : min(m, -1e-7);
    float tA = -s0 / ms;
    float tB = (L - s0) / ms;
    float lo = min(tA, tB);
    float hi = max(tA, tB);

    r.bodyT = vec2(max(t0, lo), min(t1, hi));
    r.loT = vec2(t0, min(t1, lo));
    r.hiT = vec2(max(t0, hi), t1);
    r.lo = (ms > 0.0) ? capA : capB;
    r.hi = (ms > 0.0) ? capB : capA;
    return r;
}

// int_t0^t1 dt / (d(t)^2 + eps^2)^(3/2) for the capsule [a, b]; d = distance to the segment.
float capsuleEmission3(vec3 ro, vec3 rd, float t0, float t1, vec3 a, vec3 b, float eps2)
{
    CapsuleRanges r = capsuleRanges(ro, rd, t0, t1, a, b, eps2);
    float acc = 0.0;
    if (r.bodyT.y > r.bodyT.x)
        acc += quadIntegral3(r.body.x, r.body.y, r.body.z, r.bodyT.x, r.bodyT.y);
    if (r.loT.y > r.loT.x)
        acc += quadIntegral3(1.0, r.lo.y, r.lo.z, r.loT.x, r.loT.y);
    if (r.hiT.y > r.hiT.x)
        acc += quadIntegral3(1.0, r.hi.y, r.hi.z, r.hiT.x, r.hiT.y);
    return acc;
}

// int_t0^t1 dt / (d(t)^2 + eps^2) for the capsule [a, b] (Cauchy profile, atan-based).
float capsuleEmission2(vec3 ro, vec3 rd, float t0, float t1, vec3 a, vec3 b, float eps2)
{
    CapsuleRanges r = capsuleRanges(ro, rd, t0, t1, a, b, eps2);
    float acc = 0.0;
    if (r.bodyT.y > r.bodyT.x)
        acc += quadIntegral2(r.body.x, r.body.y, r.body.z, r.bodyT.x, r.bodyT.y);
    if (r.loT.y > r.loT.x)
        acc += quadIntegral2(1.0, r.lo.y, r.lo.z, r.loT.x, r.loT.y);
    if (r.hiT.y > r.hiT.x)
        acc += quadIntegral2(1.0, r.hi.y, r.hi.z, r.hiT.x, r.hiT.y);
    return acc;
}

// Segment record shared by the benchmarks (the app's Segment is 80 B, plan 5.8; this is the 32 B subset the
// emission integral needs): p0 + eps (core half-width, metres), p1 + power.
struct Segment
{
    vec4 p0e;
    vec4 p1w;
};

// ---------------------------------------------------------------------------------------------------------------
// AgX tone mapping (three.js, MIT).  Input: exposed linear sRGB radiance.  Output: display-linear sRGB in [0, 1].
// ---------------------------------------------------------------------------------------------------------------
// Matrices for rec 2020 <> rec 709 color space conversion
// matrix provided in row-major order so it has been transposed
// https://www.itu.int/pub/R-REP-BT.2407-2017
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
    vec3( 1.6605, - 0.1246, - 0.0182 ),
    vec3( - 0.5876, 1.1329, - 0.1006 ),
    vec3( - 0.0728, - 0.0083, 1.1187 )
);

const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
    vec3( 0.6274, 0.0691, 0.0164 ),
    vec3( 0.3293, 0.9195, 0.0880 ),
    vec3( 0.0433, 0.0113, 0.8956 )
);

// https://iolite-engine.com/blog_posts/minimal_agx_implementation
// Mean error^2: 3.6705141e-06
vec3 agxDefaultContrastApprox( vec3 x ) {

    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;

    return + 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232;

}

// AgX Tone Mapping implementation based on Filament, which in turn is based
// on Blender's implementation using rec 2020 primaries
// https://github.com/google/filament/pull/7236
// Inputs and outputs are encoded as Linear-sRGB.

vec3 AgXToneMapping( vec3 color ) {

    // AgX constants
    const mat3 AgXInsetMatrix = mat3(
        vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
        vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
        vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
    );

    // explicit AgXOutsetMatrix generated from Filaments AgXOutsetMatrixInv
    const mat3 AgXOutsetMatrix = mat3(
        vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
        vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
        vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
    );

    // LOG2_MIN      = -10.0
    // LOG2_MAX      =  +6.5
    // MIDDLE_GRAY   =  0.18
    const float AgxMinEv = - 12.47393;  // log2( pow( 2, LOG2_MIN ) * MIDDLE_GRAY )
    const float AgxMaxEv = 4.026069;    // log2( pow( 2, LOG2_MAX ) * MIDDLE_GRAY )

    color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;

    color = AgXInsetMatrix * color;

    // Log2 encoding
    color = max( color, 1e-10 ); // avoid 0 or negative numbers for log2
    color = log2( color );
    color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );

    color = clamp( color, 0.0, 1.0 );

    // Apply sigmoid
    color = agxDefaultContrastApprox( color );

    // Apply AgX look
    // v = agxLook(v, look);

    color = AgXOutsetMatrix * color;

    // Linearize
    color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );

    color = LINEAR_REC2020_TO_LINEAR_SRGB * color;

    // Gamut mapping. Simple clamp for now.
    color = clamp( color, 0.0, 1.0 );

    return color;

}

#endif // PLASMA_COMMON_GLSL
