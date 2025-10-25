#!/usr/bin/env -S ./shadertoy.py -v iChannel0 presets/vol_grey_noise_3D.bin

// Copied from https://www.shadertoy.com/view/WlGGRV
// Created by https://www.shadertoy.com/user/Klems

uniform highp sampler3D iChannel0;

#define PI 3.14159265359
#define rot(a) mat2(cos(a + PI*0.5*vec4(0,1,3,0)))

vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// https://research.nvidia.com/sites/default/files/pubs/2017-02_Hashed-Alpha-Testing/Wyman2017Hashed.pdf
const float hashScale = 1.0;
float hashedNoise(vec3 p, vec3 dpdx, vec3 dpdy) {
    float maxDeriv = max(length(dpdx), length(dpdy));
    float pixScale = 1.0 / (hashScale*maxDeriv);
    vec2 pixScales = vec2(exp2(floor(log2(pixScale))), exp2(ceil(log2(pixScale))));
    float aa = textureGrad(iChannel0, pixScales.x*p.xyz/32.0, pixScales.x*dpdx/32.0, pixScales.x*dpdy/32.0).r;
    float bb = textureGrad(iChannel0, pixScales.y*p.xyz/32.0, pixScales.y*dpdx/32.0, pixScales.y*dpdy/32.0).r;
    vec2 alpha = vec2(aa, bb);
    //alpha = vec2(hash33(floor(pixScales.x*p.xyz)).r, hash33(floor(pixScales.y*p.xyz)).r);
    float lerpFactor = fract( log2(pixScale) );
    float x = (1.0-lerpFactor)*alpha.x + lerpFactor*alpha.y;
    float a = min( lerpFactor, 1.0-lerpFactor );
    vec3 cases = vec3( x*x/(2.0*a*(1.0-a)),(x-0.5*a)/(1.0-a),1.0-((1.0-x)*(1.0-x)/(2.0*a*(1.0-a))) );
    float alphaRes = (x < (1.0-a)) ? ((x < a) ? cases.x : cases.y) :cases.z;
    alphaRes = clamp(alphaRes, 1.0e-6, 1.0);
    return alphaRes;
}

// https://iquilezles.org/articles/filteringrm
void calcDpDxy( in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy, in float t, in vec3 nor,
out vec3 dpdx, out vec3 dpdy ) {
    dpdx = t*(rdx*dot(rd,nor)/dot(rdx,nor) - rd);
    dpdy = t*(rdy*dot(rd,nor)/dot(rdy,nor) - rd);
}

// noise with smooth derivative
float snoise( in vec3 x, const in float lod ) {
    float dim = 32.0 / exp2(lod);
    x = x * dim;
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    x = (p+f+0.5) / dim;
    return textureLod(iChannel0, x, lod).r;
}

// smoother noise
float noise( in vec2 x ) {
    x *= 32.0;
    const vec2 e = vec2(1, 0);
    vec2 i = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix(mix( hash12(i+e.yy), hash12(i+e.xy),f.x),
    mix( hash12(i+e.yx), hash12(i+e.xx),f.x),f.y);
}

// cascading return to optimize distance function
float height(vec2 p, float y) {
    p /= 32.0;
    float hei = noise(p*0.04);
    hei *= hei*30.0;
    if (y > hei+3.0) return hei;
    hei += snoise(vec3(p*0.5, 0), 0.0)*1.0;
    if (y > hei+2.0) return hei;
    hei += snoise(vec3(p*1.0, 10), 10.0)*0.2;
    if (y > hei+1.0) return hei;
    hei += snoise(vec3(p*2.0, 100), 20.0)*0.1;
    return hei;
}

float de(vec3 p) {
    return p.y - height(p.xz, p.y);
}

vec3 getNormal(vec3 p) {
    vec3 e = vec3(0.0, 0.3, 0.0);
    return normalize(vec3(
    de(p+e.yxx)-de(p-e.yxx),
    de(p+e.xyx)-de(p-e.xyx),
    de(p+e.xxy)-de(p-e.xxy)));
}

bool intSphere( in vec4 sp, in vec3 ro, in vec3 rd, out float t ) {
    vec3  d = ro - sp.xyz;
    float b = dot(rd,d);
    float c = dot(d,d) - sp.w*sp.w;
    float tt = b*b-c;
    if ( tt > 0.0 ) {
        t = -b-sqrt(tt);
        return true;
    }
    return false;
}

float star( in vec3 dir ) {
    dir.yz *= rot(-0.7);
    float base = step(abs(dir.z), 0.007);
    dir.xy *= rot(iTime*0.5);
    float trail = (atan(dir.x, dir.y)+PI)/(2.0*PI);
    trail = pow(trail+0.05, 20.0);
    return base*trail;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {

    vec2 uv = (fragCoord - iResolution.xy * 0.5) / iResolution.y;
    uv += uv*dot(uv,uv)*0.5;

    vec3 from = vec3(0, 2, -5);
    vec3 dir = normalize(vec3(uv, 0.4));

    vec2 mouse=(iMouse.xy - iResolution.xy*0.5) / iResolution.y * 2.5;
    if (iMouse.z < 0.5) mouse = vec2(0);
    mat2 rotxz = rot(-mouse.x+sin(iTime*0.1512)*0.75-PI*0.5);
    mat2 rotxy = rot(mouse.y+sin(iTime*0.12412)*0.25);
    dir.zy *= rotxy;
    dir.xz *= rotxz;

    from.x += iTime*10.0;
    from.y += height(from.xz, 9e9);

    float totdist = 0.0;
    for (int steps = min(iFrame, 0) ; steps < 150 ; steps++) {
        vec3 p = from + totdist * dir;
        float dist = de(p);
        totdist += dist*0.5;
        if (dist < 0.001 || totdist > 400.0) {
            break;
        }
    }

    const vec3 light = normalize(vec3(3, 1, 2));
    float noi = hashedNoise(dir, dFdx(dir), dFdy(dir));
    float lig = 0.0;
    vec3 test = vec3(0);
    if (totdist > 400.0) {

        // add stars
        lig = pow(texture(iChannel0, dir*0.7).r, 40.0)*10.0;

        // add a shooting star
        lig += star(dir);

        // add a planet
        vec3 fromPl = vec3(8, 5, -1);
        float toSphere = 0.0;
        bool sphere = intSphere( vec4(0, 0, 0, 1), fromPl, dir, toSphere);
        if (sphere) {
            vec3 normal = fromPl + dir*toSphere;
            lig = max(0.0, dot(normal, light))*30.0;
        }

    } else {

        vec3 p = from + totdist * dir;
        vec3 n = getNormal(p);
        vec3 dpdx = vec3(0);
        vec3 dpdy = vec3(0);
        calcDpDxy(from, dir, dir+dFdx(dir), dir+dFdy(dir), totdist, n, dpdx, dpdy);

        noi = hashedNoise(p, dpdx, dpdy);

        // diffuse
        lig = max(0.0, dot(n, light));
        lig *= lig*2.0;

        // some fresnel
        float fres = pow(max(0.0, 1.0-dot(n, -dir)), 10.0);
        lig += fres*2.0;
    }

    // vignette, gamma correction
    lig = pow(lig*0.5, 1.0/2.2);
    vec2 uu = (fragCoord.xy-iResolution.xy*0.5)/iResolution.xy;
    lig = mix(lig, 0.0, dot(uu,uu)*1.3);

    fragColor = vec4(step(noi, lig));
}
