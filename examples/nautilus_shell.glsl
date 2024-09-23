#!/usr/bin/env -S python -m shaderbang.shadertoy

// Copied from https://www.shadertoy.com/view/sdVGWh
// Created by https://www.shadertoy.com/user/harry7557558

// I have always been facinated with seashells.

// Nautilus is probably the most mathematical one
// to get my seashell shader journey started.


#define PI 3.1415926
#define ZERO min(iTime,0.)


vec2 cut;  // when modeling the nautilus, cut z<cut.x and z>cut.y to show its interior


// calculate the signed distance and color of the nautilus shell
// this function calculates color only when @req_color is true
float mapShell(in vec3 p, out vec3 col, bool req_color) {
    p -= vec3(0.7, 0, 0);

    // r=exp(b*θ)
    const float b = 0.17;

    // Catesian to cylindrical
    float r = length(p.xy);  // r
    float a = mix(0.0, 0.45, smoothstep(0.0, 1.0, 0.5*(r-0.6)));  // rotate by this angle
    p.xy = mat2(cos(a),-sin(a),sin(a),cos(a))*p.xy;  // rotation
    float t = atan(p.y, p.x);  // θ

    // shell opening, kill discontinuities of the spiral
    float ro = exp(b*PI);  // center of the "ring"
    float d = length(vec2(length(p.xz-vec2(-ro,0))-ro,p.y));  // distance to the "ring"
    float u = t, dx = r-ro, dy = p.z;  // longitude and two numbers to determine latitude

    // spiral
    // r(n) = exp(b*(2.*PI*n+t)), (x-r)^2+y^2=r^2, solve for n
    float n = (log((r*r+p.z*p.z)/(2.*r))/b-t)/(2.0*PI);  // decimal n
    n = min(n, 0.0);  // clamp to opening
    float n0 = floor(n), n1 = ceil(n);  // test two boundaries
    float r0 = exp(b*(2.*PI*n0+t)), r1 = exp(b*(2.*PI*n1+t));  // two r
    float d0 = abs(length(vec2(r-r0,p.z))-r0);  // distance to inner
    float d1 = abs(length(vec2(r-r1,p.z))-r1);  // distance to outer
    if (d0 < d) d = d0, u = 2.*PI*n0+t, dx = r-r0, dy = p.z;  // update distance
    if (d1 < d) d = d1, u = 2.*PI*n1+t, dx = r-r1, dy = p.z;  // update distance

    // septa/chambers
    const float f = 2.4;  // "frequency" of chambers
    float s0 = t + 2.0*PI*(n0+0.5);  // longitude parameter
    float v = fract(n);  // 0-1, distance from inner circle
    float s = f*s0 + 1.0*pow(0.25-(v-0.5)*(v-0.5), 0.5)+0.5*v;  // curve of septa
    s += pow(min(1.0/(40.0*length(vec2(v-0.5,p.z))+1.0), 0.5), 2.0);  // hole on septa
    float sf = fract(s);  // periodic
    sf = s0>-1.8 ? abs(s+3.25) :  // outer-most septa, possibly cause discontinuities
    min(sf, 1.0-sf);  // inner septa
    float w = sf/f*exp(b*(s0+PI));  // adjust distance field
    if (length(p*vec3(1,1,1.5))<3.0)  // prevent outer discontinuity
    d = min(d, 0.5*w+0.012);  // union chambers

    d += 0.00012*r*sin(200.*u);  // geometric texture
    d = abs(d)-0.8*max(0.02*pow(r,0.4),0.02);  // thickness of shell
    d = max(d, max(cut.x-p.z,p.z-cut.y));  // cut it open
    if (!req_color) return d;  // distance calculation finished

    // color
    v = atan(dy, dx);  // latitude parameter
    w = length(vec2(dx,dy)) / exp(b*u);  // section radius parameter
    for (float i=0.;i<6.;i+=1.) {  // distort the parameters
        float f = pow(2., i);
        float du = 0.15/f*sin(f*u)*cos(f*v);
        float dv = 0.15/f*cos(f*u)*sin(f*v);
        u+=du, v+=dv;
    }
    float f1 = cos(50.*u);  // middle stripes
    float f2 = cos(21.3*u)+0.1;  // side stripes
    float tex = mix(f1, f2, 0.5-0.5*tanh(1.0-3.0*sin(v)*sin(v)))  // blend stripes
    + 0.5-0.6*cos(v);  // fading at sides
    tex += 0.5+0.5*tanh(4.0*(u-2.0));  // fading near opening
    col = n==0.0 ? vec3(0.9,0.85,0.8) : vec3(0.95,0.85,0.7);  // base color, outer and inner
    if (w>1.0 && w<1.1)  // on the surface of the shell
    col = (u-0.3*cos(v)<-2.6 ? 1.0-0.6*min(exp(2.+0.5*u),1.0) : 1.0)  // black inside the opening
    * mix(vec3(0.6,0.3,0.2), col, clamp(8.0*tex+0.5,0.,1.));  // apply stripes

    return d;
}

// calculate the signed distance and color of the scene
float map(in vec3 p, out vec3 col, bool req_color) {
    vec3 shell_col;
    float shell_d = mapShell(p.yzx, shell_col, req_color);  // call mapShell
    float beach_d = p.z+1.5;  // beach surface: z=-1.5
    beach_d += 0.001*sin(20.0*p.x)*sin(20.0*p.y) + 0.0005*(sin(51.0*p.x)+sin(50.0*p.y));  // deform the surface of the beach
    vec3 beach_col = vec3(0.95,0.8,0.5);  // color of the beach
    float d = min(shell_d, beach_d);  // final signed distance
    if (d==shell_d) col = shell_col;  // closer to nautilus shell
    else col = beach_col;  // closer to beach
    return d;
}

// calculate signed distance only
float mapDist(vec3 p) {
    vec3 col;
    return map(p, col, false);
}

// numerical gradient of the SDF
vec3 mapGrad(vec3 p) {
    const float e = 0.001;
    float a = mapDist(p+vec3(e,e,e));
    float b = mapDist(p+vec3(e,-e,-e));
    float c = mapDist(p+vec3(-e,e,-e));
    float d = mapDist(p+vec3(-e,-e,e));
    return (.25/e)*vec3(a+b-c-d,a-b+c-d,a-b-c+d);
}


// "standard" raymarching
bool raymarch(vec3 ro, vec3 rd, inout float t, float t1, float step_size, float eps) {
    for (int i=int(ZERO); i<100; i++) {
        float dt = step_size*mapDist(ro+rd*t);
        t += dt;
        if (abs(dt) < eps) break;
        if (t > t1) return false;
    }
    return true;
}

// soft shadow - https://iquilezles.org/articles/rmshadows
float calcShadow(vec3 ro, vec3 rd) {
    float sh = 1.;
    float t = 0.1;
    for (int i = int(ZERO); i<20; i++){
        float h = mapDist(ro + rd*t);
        sh = min(sh, smoothstep(0., 1., 4.0*h/t));
        t += clamp(h, 0.1, 0.3);
        if (h<0. || t>8.0) break;
    }
    return max(sh, 0.);
}

// AO - from Shane's https://www.shadertoy.com/view/wslcDS
float calcAO(vec3 p, vec3 n){
    float sca = 1.5;
    float occ = 0.;
    for(float i=ZERO+1.; i<=5.; i+=1.){
        float t = 0.07*i;
        float d = mapDist(p+n*t);
        occ += (t-d)*sca;
        sca *= .5;
    }
    return 1.0 - clamp(occ, 0., 1.);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {

    // set cut to view interior
    float at = mod(iTime, 8.0);  // animation time
    cut = vec2(-1.8, 1.8);  // show all
    cut = mix(cut, vec2(-1.8, 0.0), clamp(at-1.0, 0., 1.));  // half
    cut = mix(cut, vec2(-0.1, 0.1), clamp(at-3.0, 0., 1.));  // slice in the middle
    cut = mix(cut, vec2(-1.0, -0.8), clamp(at-5.0, 0., 1.));  // slice at the side
    cut = mix(cut, vec2(-1.8, 1.8), clamp(at-7.0, 0., 1.));  // show all

    // set camera
    float rx = iMouse.z!=0.0 ? 1.8*(iMouse.y/iResolution.y)-0.2 : 0.3;
    float rz = iMouse.z!=0.0 ? -iMouse.x/iResolution.x*4.0*3.14 : -0.3;

    vec3 w = vec3(cos(rx)*vec2(cos(rz),sin(rz)), sin(rx));  // far to near
    vec3 u = vec3(-sin(rz),cos(rz),0);  // left to right
    vec3 v = cross(w,u);  // down to up

    vec3 ro = 12.0*w;  // ray origin
    vec2 uv = 2.0*fragCoord.xy/iResolution.xy - vec2(1.0);
    vec3 rd = mat3(u,v,-w)*vec3(uv*iResolution.xy, 2.0*length(iResolution.xy));
    rd = normalize(rd);  // ray direction

    // ray intersection
    float t0 = 0.01;  // start at t=t0
    float t1 = 3.0*length(ro);  // end distance
    float t = t0;
    if (!raymarch(ro, rd, t, t1, 0.8, 1e-3)) {  // raymarch
        t = 100.;  // miss, set t to a large number so it fades
    }
    vec3 p = ro+rd*t;  // current position

    const vec3 sundir = normalize(vec3(0.5, -0.5, 0.5));  // direction of the sun
    vec3 n = normalize(mapGrad(p));  // get normal
    vec3 col; map(p, col, true);  // get color
    float shadow = calcShadow(p, sundir);  // soft shadow
    float ao = calcAO(p, n);  // ao
    vec3 sunlight = shadow * max(dot(n, sundir), 0.0) * vec3(0.9, 0.8, 0.6);  // sunlight, yellowish
    vec3 skylight = ao * max(n.z, 0.0) * vec3(0.6, 0.7, 0.8);  // skylight, blueish
    vec3 backlit = ao * (vec3(0.2)  // background lighting
    + vec3(0.3)*max(-dot(n,sundir),0.0)  // opposite of sunlight
    + vec3(0.4,0.3,0.2)*max(-n.z,0.0));  // opposite of skylight, warm
    col *= sunlight + skylight + backlit;  // sum three lights
    float fresnel = 0.2+1.4*pow(1.0+dot(rd,n),2.0);  // faked Fresnel reflectance
    vec3 refl = 0.8*col+vec3(0.4,0.3,0.2)*pow(max(dot(rd-2.0*dot(rd,n)*n,sundir),0.0),100.);  // reflection, blend with col
    if (!raymarch(p+0.2*reflect(rd, n), reflect(rd, n), t0, 8.0, 1.0, 0.02))  // not occluded
    col = mix(col, refl, fresnel);  // add reflection
    col = mix(vec3(0.5, 0.6, 0.7)-0.3*max(rd.z, 0.0), col, exp(-0.15*max(t-0.4*t1,0.)));  // sky/fog
    col += 0.6*vec3(0.3,0.2,0.25) * max(dot(rd, sundir), 0.);  // sun haze
    col = 0.9*pow(col, vec3(0.75));  // brightness/gamma
    fragColor = vec4(vec3(col), 1.0);
}
