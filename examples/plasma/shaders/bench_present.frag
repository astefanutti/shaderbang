#version 460
// Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
// SPDX-License-Identifier: MIT
//
// M1 micro-benchmark 5: the present pass of plan 5.7 -- fullscreen triangle at mode resolution sampling the
// upscaled history, the bilinear 2x glow, exposure, AgX (three.js port in common.glsl), sRGB encoding and a
// 64x64 tiled blue-noise dither into an RGBA8 target.

#include "common.glsl"

layout(binding = 0) uniform sampler2D uHistory;    // mode resolution, rgba16f
layout(binding = 1) uniform sampler2D uGlow;       // internal resolution, rgba16f (bilinear x2)
layout(binding = 2) uniform sampler2D uBlueNoise;  // 64x64 R8

uniform ivec2 uOutRes;
uniform float uExposure;
uniform float uGlowWeight;

in vec2 vUV;
out vec4 fragColor;

void main()
{
    vec2 uv = gl_FragCoord.xy / vec2(uOutRes);
    vec3 hdr = texture(uHistory, uv).rgb + uGlowWeight * texture(uGlow, uv).rgb;
    vec3 c = AgXToneMapping(hdr * uExposure);
    c = LinearToSRGB(c);
    float n = texelFetch(uBlueNoise, ivec2(gl_FragCoord.xy) & 63, 0).r;
    c += (n - 0.5) / 255.0;
    fragColor = vec4(c, 1.0);
}
