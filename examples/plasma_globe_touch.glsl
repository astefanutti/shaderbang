// python glsl.py -t iChannel0 presets/tex_RGBA_noise_medium.png --touchscreen iTouch

// Interactive Plasma Globe - Touchscreen
//
// This is a fork of the amazing "Plasma Globe"
// shader made by nimitz, with the addition of
// touchscreen interaction.
//
// Original Shader Header :
//
// Plasma Globe by nimitz (twitter: @stormoid)
// https://www.shadertoy.com/view/XsjXRm
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
// Contact the author for other licensing options

#define ENABLE_TOUCH 1
#define NUM_TOUCH 10

#define NUM_RAYS 10

#define MIN_RAYS min(NUM_RAYS, NUM_TOUCH)

#define VOLUMETRIC_STEPS 19

#define MAX_ITER 35
#define FAR 6.

#define time iTime * 1.1


uniform highp sampler2D iChannel0;
uniform highp vec4[NUM_TOUCH] iTouch;

mat2 mm2(in float a) {float c = cos(a), s = sin(a);return mat2(c, -s, s, c);}

float noise(in float x) {return textureLod(iChannel0, vec2(x * .01, 1.), 0.0).x;}

float hash(float n) {return fract(sin(n) * 43758.5453);}

float noise(in vec3 p)
{
    vec3 ip = floor(p);
    vec3 fp = fract(p);
    fp = fp * fp * (3.0 - 2.0 * fp);

    vec2 tap = (ip.xy + vec2(37.0, 17.0) * ip.z) + fp.xy;
    vec2 rg = textureLod(iChannel0, (tap + 0.5) / 256.0, 0.0).yx;
    return mix(rg.x, rg.y, fp.z);
}

mat3 m3 = mat3(0.00, 0.80, 0.60,
               -0.80, 0.36, -0.48,
               -0.60, -0.48, 0.64);


//See: https://www.shadertoy.com/view/XdfXRj
float flow(in vec3 p, in float t)
{
    float z = 2.;
    float rz = 0.;
    vec3 bp = p;
    for (float i = 1.;i < 5.; i++)
    {
        p += time * .1;
        rz += (sin(noise(p + t * 0.8) * 6.) * 0.5 + 0.5) / z;
        p = mix(bp, p, 0.6);
        z *= 2.;
        p *= 2.01;
        p *= m3;
    }
    return rz;
}

// Could be improved
float sins(in float x)
{
    float rz = 0.;
    float z = 2.;
    for (float i = 0.;i < 3.; i++)
    {
        rz += abs(fract(x * 1.4) - 0.5) / z;
        x *= 1.3;
        z *= 1.15;
        x -= time * .65 * z;
    }
    return rz;
}

float segm(vec3 p, vec3 a, vec3 b)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.);
    return length(pa - ba * h) * .5;
}

vec3 path(in float i, in float d, vec3 hit)
{
    float sns2 = sins(d + i * 0.5) * 0.22;
    float sns = sins(d + i * .6) * 0.21;
    //float sns2 = sins(d + i * 0.5) * 0.;
    //float sns = sins(d + i * .6) * 0.;

    if (dot(hit, hit) > 0.) {
        // mouse interaction
        hit.xz *= mm2(sns2 * .5);
        hit.xy *= mm2(sns * .3);
        return hit;
    }

    vec3 en = vec3(0., 0., 1.);
    en.xz *= mm2((hash(i * 10.569) - .5) * 6.2 + sns2);
    en.xy *= mm2((hash(i * 4.732) - .5) * 6.2 + sns);

    return en;
}

vec2 map(vec3 p, float i, vec3 hit)
{
    float lp = length(p);
    vec3 bg = vec3(0.);
    vec3 en = path(i, lp, hit);

    float ins = smoothstep(0.11, .46, lp);
    float outs = .15 + smoothstep(.0, .15, abs(lp - 1.));
    p *= ins * outs;
    float id = ins * outs;

    float rz = segm(p, bg, en) - 0.011;

    return vec2(rz, id);
}

float march(in vec3 ro, in vec3 rd, in float startf, in float maxd, in float j, vec3 hit)
{
    float precis = 0.001;
    float h = 0.5;
    float d = startf;
    for (int i = 0; i < MAX_ITER; i++)
    {
        if (abs(h) < precis || d > maxd) break;
        d += h * 1.2;
        float res = map(ro + rd * d, j, hit).x;
        h = res;
    }
    return d;
}

// Volumetric marching
vec3 vmarch(in vec3 ro, in vec3 rd, in float j, in vec3 orig, vec3 hit)
{
    vec3 p = ro;
    vec2 r = vec2(0.);
    vec3 sum = vec3(0);
    float w = 0.;
    for (int i = 0; i < VOLUMETRIC_STEPS; i++)
    {
        r = map(p, j, hit);
        p += rd * .03;
        float lp = length(p);

        vec3 col = sin(vec3(1.05, 2.5, 1.52) * 3.94 + r.y) * .85 + 0.4;
        col.rgb *= smoothstep(.0, .015, -r.x);
        col *= smoothstep(0.04, .2, abs(lp - 1.1));
        col *= smoothstep(0.1, .34, lp);

        if (dot(hit, hit) > 0.) {
            col *= 2.5;
        }

        sum += abs(col) * 5. * (1.2 - noise(lp * 2. + j * 13. + time * 5.) * 1.1) / (log(distance(p, orig) - 2.) + .75);
    }
    return sum;
}

// Returns both collision dists of unit sphere
vec2 iSphere2(in vec3 ro, in vec3 rd)
{
    vec3 oc = ro;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - 1.;
    float h = b * b - c;
    if (h < 0.0) return vec2(-1.);
    else return vec2((-b - sqrt(h)), (-b + sqrt(h)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = fragCoord.xy / iResolution.xy - 0.5;
    p.x *= iResolution.x / iResolution.y;

    // Camera
    vec3 ro = vec3(0., 0., 5.);
    vec3 rd = normalize(vec3(p * .7, -1.5));
    mat2 mx = mm2(time*.4);
    mat2 my = mm2(time*0.3);
    //mat2 mx = mm2(time * .0);
    //mat2 my = mm2(time * 0.);
    ro.xz *= mx;rd.xz *= mx;
    ro.xy *= my;rd.xy *= my;

    vec3 bro = ro;
    vec3 brd = rd;

    vec3 col = vec3(0.0125, 0., 0.025);

    vec3 intersect[MIN_RAYS];
    vec3 hits[MIN_RAYS];
    int intersects = 0;

    #if ENABLE_TOUCH
    for (int i = 0; i < NUM_TOUCH; i++)
    {
        if (iTouch[i].z <= 0.) continue;

        vec2 um = iTouch[i].xy / iResolution.xy - .5;
        um.x *= iResolution.x / iResolution.y;

        vec3 hit = vec3(0.);
        vec3 rdm = normalize(vec3(um * .7, -1.5));
        rdm.xz *= mx;
        rdm.xy *= my;

        // Compute intersection point between the surface of the sphere
        // and the ray casted to the touch point
        vec2 res = iSphere2(bro, rdm);
        if (res.x > 0.) hit = bro + res.x * rdm;

        // Cast a ray for the light path of the touch point
        if (dot(hit, hit) == 0.) continue;

        // Find the closest ray from the touch point
        for (float j = 1.; j <= float(NUM_RAYS); j++)
        {
            ro = bro;
            rd = rdm;
            mat2 mm = mm2((time * 0.1 + ((j + 1.) * 5.1)) * j * 0.25);
            //mat2 mm = mm2(0.);
            ro.xy *= mm;
            ro.xz *= mm;
            rd.xy *= mm;
            rd.xz *= mm;
            vec3 h = ro + iSphere2(bro, rdm).x * rd;

            float d = map(h, j, vec3(0.)).x;
            bool exists = false;
            for (int k = 0; k < intersects; k++)
            {
                if (intersect[k].s == float(i)) {
                    exists = true;
                    if (d < intersect[k].p)
                    {
                        intersect[k].t = j;
                        intersect[k].p = d;
                        // hits[k] = hit;
                        break;
                    }
                }
            }
            if (!exists)
            {
                intersect[intersects] = vec3(float(i), j, d);
                hits[intersects] = hit;
                intersects++;
            }
        }
    }
    #endif

    #if 1
    for (float j = 1.; j <= float(NUM_RAYS); j++)
    {
        ro = bro;
        rd = brd;
        mat2 mm = mm2((time * 0.1 + ((j + 1.) * 5.1)) * j * 0.25);
        //mat2 mm = mm2(0.);
        ro.xy *= mm;rd.xy *= mm;
        ro.xz *= mm;rd.xz *= mm;

        vec3 hit = vec3(0.);
        #if ENABLE_TOUCH
        for (int k = 0; k < intersects; k++)
        {
            if (intersect[k].t == j) {
                hit = hits[k];
                ro = bro;
                rd = brd;
                break;
            }
        }
        #endif

        float rz = march(ro, rd, 2.5, FAR, j, hit);
        if (rz >= FAR) continue;
        vec3 pos = ro + rz * rd;
        col = max(col, vmarch(pos, rd, j, bro, hit));
    }
    #endif

    ro = bro;
    rd = brd;
    vec2 sph = iSphere2(ro, rd);

    if (sph.x > 0.)
    {
        vec3 pos = ro + rd * sph.x;
        vec3 pos2 = ro + rd * sph.y;
        vec3 rf = reflect(rd, pos);
        vec3 rf2 = reflect(rd, pos2);
        float nz = (-log(abs(flow(rf * 1.2, time) - .01)));
        float nz2 = (-log(abs(flow(rf2 * 1.2, -time) - .01)));
        col += (0.1 * nz * nz * vec3(0.12, 0.12, .5) + 0.05 * nz2 * nz2 * vec3(0.55, 0.2, .55)) * 0.8;
    }

    fragColor = vec4(col * 1.3, 1.0);
}
