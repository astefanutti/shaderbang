#version 460
// Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
// SPDX-License-Identifier: MIT
//
// Present: (resolved HDR + glow) * exposure -> AgX -> blue-noise dither -> 8-bit framebuffer.
#include "tonemap.glsl"

in vec2 vUv;
layout(location = 0) out vec4 fragColor;

layout(binding = 0) uniform sampler2D resolvedTex;   // display res, linear HDR
layout(binding = 1) uniform sampler2D glowTex;       // glow pyramid level 1 (linear HDR, bilinear)
layout(binding = 2) uniform sampler2D noiseTex;      // 64x64 blue noise, tiled
layout(std430, binding = 3) readonly buffer ExposureBuf {
    int   total; int unused; float evPrev; float evCur; float exposure; float exposureRatio; float lumP, pad1;
} ex;
uniform float glowGain;        // 0 disables the glow
uniform int   debugView;       // 1 = emissive/glow only
uniform int   tonemapMode;     // 0 AgX, 1 camera (clip + sRGB OETF, like the reference video)
uniform float fixedExposure;   // > 0 overrides the metered exposure

vec3 srgbOETF(vec3 c) {
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// Camera-like tone curve that keeps hue: a per-channel clip turns a bright violet filament pink
// wherever it is brighter (foreshortening, overlaps), so instead the largest channel is rolled
// off with a tanh shoulder above the knee and all channels scaled alike; only extreme highlights
// (the touched filament, ~10x) desaturate towards white as a sensor would.
vec3 toneCamera(vec3 c) {
    const float knee = 0.7;
    float m = max(c.r, max(c.g, c.b));
    if (m > knee) {
        float m2 = knee + (1.0 - knee) * tanh((m - knee) / (1.0 - knee));
        c *= m2 / m;
        float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(c, vec3(lum) / max(lum, 1e-6) * m2, clamp((m - 1.2) / 2.8, 0.0, 1.0));   // white by ~4x
    }
    return clamp(c, 0.0, 1.0);
}

void main() {
    vec3 hdr = textureLod(resolvedTex, vUv, 0.0).rgb;
    ivec2 np = ivec2(gl_FragCoord.xy) & 63;
    float noise = texelFetch(noiseTex, np, 0).r - 0.5;
    vec3 glow = textureLod(glowTex, vUv, 0.0).rgb * glowGain;
    float exposure = fixedExposure > 0.0 ? fixedExposure : ex.exposure;
    vec3 c = (debugView == 1 ? glow : hdr + glow) * exposure;
    vec3 srgb = tonemapMode == 1 ? srgbOETF(toneCamera(c)) : agxToneMap(c);
    fragColor = vec4(srgb + noise / 255.0, 1.0);
}
