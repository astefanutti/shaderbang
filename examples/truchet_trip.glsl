#!/usr/bin/env -S python -m shaderbang.shadertoy

// Copied from https://www.shadertoy.com/view/st3BDf
// Created by https://www.shadertoy.com/user/BigWIngs

// "Truchet Trip"
// by Martijn Steinrucken aka BigWings/The Art of Code - 2021
//
// I recently made a tutorial on how to make a weave of two quad truchet layers.
// Check it out here:
// https://youtu.be/pmS-F6RJhAk
//
// Here is a 'sketch' I had lying around that makes use of this.

#define S smoothstep
#define TAU 6.283185
#define AA 3

mat2 Rot(float a) {
    float s=sin(a),c=cos(a);
    return mat2(c,-s,s,c);
}

vec2 Tile(vec2 p) {
    bool corner = p.x>-p.y;
    p -= corner?.5:-.5;
    float d = length(p);

    float a = atan(p.x,p.y)/TAU +.5;
    a = fract(a*4.);

    float m = cos(d*TAU*2.);

    return vec2(a-.5, d-.5);
}

float Hash21(vec2 p) {
    p = fract(p*vec2(123.489,234.95));
    p += dot(p, p+34.4);
    return fract(p.x*p.y);
}

float Xor(float a, float b) {
    return a*(1.-b) + b*(1.-a);
}

vec3 TileLayer(vec2 p, float w) {
    vec2 gv = fract(p)-.5;

    p.y = mod(p.y, 6.);
    vec2 id = floor(p);
    // id = mod(id, 16.);

    float checker = mod(id.x+id.y, 2.);

    float n = Hash21(id);
    //n = .3;
    float flip = step(n, .5);

    if(flip==1.) gv.x *= -1.;

    vec2 st = Tile(gv);

    st.x = mix(st.x, 1.-st.x, checker);
    st.x = (st.x-.5)/2.+.5;
    st.y = st.y*(Xor(checker, flip)*2.-1.)/(w*2.);

    float z = gv.x>.48 || gv.y>.48 ? 1. : 0.;
    return vec3(st, z);
}

vec3 Render(vec2 fragCoord) {
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
    vec2 M = iMouse.xy/iResolution.xy;

    vec3 col = vec3(0);
    float r = 16., r2=r/2., r4=r/4.;
    float t = iTime+M.x*17.;
    t = mod(t, r);

    vec2 p = uv;
    float cd = length(p), lcd = log(cd);

    p*= Rot(sin(t*TAU/r+lcd)*.2);
    float a = atan(p.x,p.y);
    p = vec2(a/TAU+.5, lcd);
    p*= vec2(4.,1.);

    p.y -= t/r4;

    float scale = 3.;
    float w = .2;
    vec3 st1 = TileLayer(p*scale, w);

    // st.x = fract(st.x+t*.1);
    float m1 = S(.01,.0,abs(st1.y)-w);
    float h1 =  S(.0,.5, .5-abs(st1.x-.5));

    vec3 st2 = TileLayer((p+.5)*scale, .5);
    float m2 = S(.01,.0,abs(st2.y)-w);
    float h2 =  S(0., .5, .5-abs(st2.x-.5));

    float t1 = st1.x+sin(st1.x*TAU)*.0;
    float t2 = st2.x;
    float arrow = st1.x+abs(st1.y);
    arrow = fract(arrow-t/2.+a/TAU+lcd);

    col += vec3(1,.02,.02)*t1*h1*m1 * arrow;

    vec3 ribbon = vec3(.02,1,.02)*t2;
    ribbon += sin(st2.x*TAU*4.+t*TAU/r4-a-lcd)*.1;
    ribbon += sin(st2.y*TAU*20.)*.01;

    col += ribbon*h2*m2;

    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec3 col = vec3(0);

    for(int x=0; x<AA; x++) {
        for(int y=0; y<AA; y++) {
            vec2 offs = vec2(x, y)/float(AA);
            col += Render(fragCoord+offs);
        }
    }
    col /= float(AA*AA);

    col = vec3(sqrt(col.r+col.g));

    fragColor = vec4(col,1.0);
}
