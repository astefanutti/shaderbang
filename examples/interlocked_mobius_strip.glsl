#!/usr/bin/env -S python -m shaderbang.shadertoy -t iChannel0 presets/tex_rusty_metal.jpg -t iChannel1 presets/tex_wood.jpg -c iChannel2 presets/cube_st_peter_basilica_blurred.png

// Copied from https://www.shadertoy.com/view/ctdfRM
// Created by https://www.shadertoy.com/user/Shane

/*

    Interlocked Mobius Strip
    ------------------------

    Interlocked Mobius strips are a pretty common object amongst the 3D geometric
    art crowd, and on the internet in general, so I was surprised not to find one on
    Shadertoy... Well, it's possible that someone like Dr2, Fabrice, etc., has one
    hidden away on here somewhere, but I wasn't able to track one down.

    Either way, they're pretty easy to make: Produce two toroidal strips running
    perpendicular to one another, each with some repeat holes in them. Offset one
    set of holes by an angle that allows them to interweave, which will give you two
    interlaced toroidal objects. The final step is to twist each along the toroidal
    axis (the long circular one) by half a turn. This will, in effect, cause the two
    separate toroidal objects to fuse into one continuous band, which I've always
    thought looked pretty cool... but I'm easily amused, so it's probably not that
    great. :D

    I've had various incarnations of these objects sitting around for ages, so I
    thought I'd dust one off and make it presentable. It originally featured some
    rolling bearings running around the object, but I felt they were too distacting.
    There are so many variations possible. I went for the single twist version with
    bands around the edges because I thought it suited the background more. I happen
    to prefer the more elegant ceramic or metallic versions sitting on a matte plane,
    so I'll post one of those at some stage.



	Related examples:

    // Essentially the same thing, but without the interlocking component.
    // Dr2 has a heap of Mobius related material that's worth the look.
    Twisted Ladder 2 - Dr2
    https://www.shadertoy.com/view/Xsdczl

    // Stylish and mind bending at the same time.
    Eternal Commute - tdhooper
    https://www.shadertoy.com/view/ldKBRt


*/

uniform highp sampler2D iChannel0;
uniform highp sampler2D iChannel1;
uniform highp samplerCube iChannel2;


// Attempting not to unroll loops.
#define ZERO min(0, iFrame)

// Max ray distance.
#define FAR 20.



// Scene object ID to separate the mesh object from the terrain.
int objID;
vec4 vID;


// Standard 2D rotation formula.
mat2 rot2(in float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }


// Fabrice's fork of "Integer Hash - III" by IQ: https://shadertoy.com/view/4tXyWN
float hash21(vec2 f){

    // The first line relates to ensuring that icosahedron vertex identification
    // points snap to the exact same position in order to avoid hash inaccuracies.
    uvec2 p = floatBitsToUint(f);
    p = 1664525U*(p>>1U^p.yx);
    return float(1103515245U*(p.x^(p.y>>3U)))/float(0xffffffffU);
}

/*
// Commutative smooth minimum function. Provided by Tomkh, and taken
// from Alex Evans's (aka Statix) talk:
// http://media.lolrus.mediamolecule.com/AlexEvans_SIGGRAPH-2015.pdf
// Credited to Dave Smith @media molecule.
float smin(float a, float b, float k){

   float f = max(0., 1. - abs(b - a)/k);
   return min(a, b) - k*.25*f*f;
}
*/

// Commutative smooth maximum function. Provided by Tomkh, and taken
// from Alex Evans's (aka Statix) talk:
// http://media.lolrus.mediamolecule.com/AlexEvans_SIGGRAPH-2015.pdf
// Credited to Dave Smith @media molecule.
float smax(float a, float b, float k){

   float f = max(0., 1. - abs(b - a)/k);
   return max(a, b) + k*.25*f*f;
}



// Tri-Planar blending function. Based on an old Nvidia tutorial by Ryan Geiss.
vec3 tex3D(sampler2D t, in vec3 p, in vec3 n){

    n = max(abs(n) - .1, .001); // max(abs(n), 0.001), etc.
    n /= dot(n, vec3(1));
    //n /= length(n);

	vec3 tx = texture(t, p.yz).xyz;
    vec3 ty = texture(t, p.zx).xyz;
    vec3 tz = texture(t, p.xy).xyz;

    // Textures are stored in sRGB (I think), so you have to convert them to linear space
    // (squaring is a rough approximation) prior to working with them... or something like
    // that. :) Once the final color value is gamma corrected, you should see correct
    // looking colors.
    return mat3(tx*tx, ty*ty, tz*tz)*n;
}

// Texture sample.
vec3 getTex(sampler2D iCh, vec2 p){

    vec3 tx = texture(iCh, p).xyz;
    return tx*tx; // Rough sRGB to linear conversion.
}


// Height map value, which is just the pixel's greyscale value.
float hm(sampler2D iCh, in vec2 p){ return dot(getTex(iCh, p), vec3(.299, .587, .114)); }

// IQ's extrusion formula.
float opExtrusion(in float sdf, in float pz, in float h){

    //vec2 w = vec2( sdf, abs(pz) - h );
  	//return min(max(w.x, w.y), 0.) + length(max(w, 0.));

    // Slight rounding. A little nicer, but slower.
    const float sf = .028;
    vec2 w = vec2( sdf, abs(pz) - h ) + sf;
  	return min(max(w.x, w.y), 0.) + length(max(w, 0.)) - sf;

}

// IQ's box routine.
float sBoxS(in vec2 p, in vec2 b, float r){

  vec2 d = abs(p) - b + r;
  return min(max(d.x, d.y), 0.) + length(max(d, 0.)) - r;
}

// Object rotation, with some optional mouse movement.
vec3 objRot(vec3 p){

    // Mouse movement.
    if(iMouse.z>1.){
        p.yz *= rot2(-(iMouse.y - iResolution.y*.5)/iResolution.y*3.1459);
        p.xz *= rot2(-(iMouse.x - iResolution.x*.5)/iResolution.x*3.1459);
    }

    p.xy = rot2(3.14159/5.)*p.xy;
    p.yz = rot2(-3.14159/5.)*p.yz;
    p.xz = rot2(iTime/2.)*p.xz;  //iTime/4.
    return p;

}

// Global scale for the background grid.
vec2 fSc = vec2(1)/2.;


// Subdivided rectangle grid.
vec4 getGrid(vec2 p, inout vec2 sc){

    // Block offsets.
    vec2 ipOffs = vec2(0);
    // Row or column offset. Values like "1/3" would offset more
    // haphazardly, but I wanted to maintain a little symmetry.

    const float offDst = .5;

    if(mod(floor(p.y/sc.y), 2.)<.5){
        p.x -= sc.x*offDst; // Row offset.
        ipOffs.x += offDst;
    }
    /*
    if(mod(floor(p.x/sc.x), 2.)<.5){
        p.y -= sc.y*offDst; // Column offset.
        ipOffs.y += offDst;
    }
    */
    /*
    float ii = floor(p.y/sc.y);
    float offDst = mod(ii, 4.)/4.;
    p.x -= sc.x*offDst; // Row offset.
    ipOffs.x += offDst;
    */


    // Current block ID.
    vec2 ip = floor(p/sc) + .5;

    #define SUBDIV
    #ifdef SUBDIV
    // Random subdivision.

    // Rectangle partitioning.
    if(hash21(ip + .14)<.5){
       sc.x /= 2.;
       ip.x = floor(p.x/sc.x) + .5;

    }
    if(hash21(ip + .42)<.5){
       sc.y /= 2.;
       ip.y = floor(p.y/sc.y) + .5;

    }

    /*
    // Use this for squares only.
    if(hash21(ip + .253)<.5){
       sc /= 2.;
       ip = floor(p/sc) + .5;
    }
    */
    #endif

    // Local coordinates and cell ID.
    return vec4(p - ip*sc, (ip + ipOffs)*sc);

}



// Back wall unit direction ray.
// We need the direction itself for the standard cube
// traversal trickery.
vec3 gDir;
vec3 gRd;
float gCD;

// The block pattern on the back wall.
float getSurf(vec3 q3){


    vec2 sc = fSc; // Surface scale.
    vec4 p4 = getGrid(q3.xy, sc); // Grid.
    vec2 p = p4.xy, ip = p4.zw; // Local coordinates and cell ID.


    vec3 p3 = vec3(p.xy, q3.z - 2.); // 3D position.

    float h = hash21(ip + .31)*.05; // Cell height.

    float sq = sBoxS(p3.xy, sc/2., min(sc.y, sc.x)*.1); // Rounded square.

    float d2 = sq; // 2D distance.

    // Bore out some random holes.
    if(hash21(ip + .21)<.5){
    //if(mod(ip.x + ip.y, 2.)<.5){
        //d2 = sBoxS(p3.xy, sc/2. - fSc.y/5., min(sc.y, sc.x)*.07);
        float d2B = length(p3.xy) - (min(sc.y, sc.x)/2. - fSc.y/5.);
        //float d2B = opExtrusion(d2, p3.z + (h + .2)*2., .5);
        d2 = smax(d2, -d2B, fSc.y*.06);

    }

    // Extrude the 2D field above.
    float d = opExtrusion(d2, p3.z + (h + .2), h + .2);



    // Face curvature.
    float fSph = length(p3 - vec3(0, 0, -(h + .2 -  max(sc.y, sc.x)/6.)*2.)) - sc.y/2.;
    d += fSph*.1;
    // Edge smoothing.
    //d = smax(d, -abs(sq), fSc.y*.06);

    // Directional ray collision with the square cell boundaries.
    vec2 rC = (gDir.xy*sc - p4.xy)/gRd.xy; // For 2D, this will work too.

    // Minimum of all distances, plus not allowing negative distances.
    // Adding a touch to advance to the next cell.
    gCD = max(min(rC.x, rC.y), 0.) + .001;

    // Return the surface distance.
    return d;


}





// Texture coordinates. It's easier to save them in the distance field and
// reuse them later, rather than recalculate them all over again. The downside
// is expense, but it's not really noticeable here.
vec3 txCoord;


// Scene distance function.
float map(vec3 p){

    // Back wall.
    //
    float fl = getSurf(p);



    // Rotate the object.
    vec3 rP = objRot(p);

    // Number of toroidal twists: Only whole numbers work.
    // Odd numbers will produce a continous strip, and even numbers
    // will produce more than one.
    float twists = 1.;


    // Toroidal strip dimensions.
    vec2 dim = vec2(.1, .02);
    float r = .38; // Toroidal radius.

    // Disc coordinates.
    vec3 q = rP;
    vec2 tc = vec2(length(q.xz) - r, rP.y);


    // Disc holes.
    vec3 q2 = rP;
    float aN = 14.;
    float a = mod(atan(q2.z, q2.x), 6.2831);
    float na = (floor(a*aN/6.2831) + .5)/aN;

    // Rotate the repeat cells into position and move them out by the radius.
    q2.xz *= rot2(-na*6.2831);
    q2.x -= r;

    q2.xy *= rot2(a*twists/2. - iTime*.5); // Twisting the toroidal plane objects.
    // Producing the holes.
    //float hole = sBoxS(q2.xz, vec2(1., 1)*r*6.2831/aN/2.*.65, .05); // X-axis holes.
    float hole = length(q2.xz) - r*6.2831/aN/2.*.7; // X-axis holes.


    tc *= rot2(a*twists/2. - iTime*.5); // Twisting the toroidal plane itself.
    float taper = smoothstep(0., 1., abs(tc.x)/dim.x)*.5 + .5; // Holowing out the center.
    float tor = sBoxS(tc, dim*vec2(1, taper), .01); // Creating the central strip.

    // Outer band coordinates.
    vec2 btc = tc; btc.x = abs(btc.x) - dim.x - dim.y;
    float bands = sBoxS(btc, vec2(1, 1.5)*dim.y, .01);
    tor = smax(tor, -hole, .01); // Boring out the wholes.

    // Band ridges... Interesting, but not for this example.
    //bands += smoothstep(0., 1., sin(a*aN*6.))*.001;

    // Saving some coordinates to use for texturing.
    txCoord = vec3(tc.xy, a/6.2831);


    // Doing the same as above, but for a second object running
    // perpendicular to the first. By twisting each object by a
    // half turn, they fuse into one another giving the impression
    // of one continous band.
    q2 = rP;
    q2.xz *= rot2(3.14159/aN);
    a = atan(q2.z, q2.x);
    na = (floor(a*aN/6.2831) + .5)/aN;
    q2.xz *= rot2(-na*6.2831);
    q2.x -= r;


    q2.xy *= rot2(a*twists/2. - iTime*.5); // Twisting the toroidal plane objects.
    //hole = sBoxS(q2.yz, vec2(1., 1)*r*6.2831/aN/2.*.65, .05); // Square holes.
    hole = length(q2.yz) - r*6.2831/aN/2.*.7; // Z-axis holes.

    // Rendering the same as the torus object above, but out of
    // sync by 90 degrees.
    taper = smoothstep(0., 1., abs(tc.y)/dim.x)*.5 + .5; // Holowing out in the center.
    float tor2 = sBoxS(tc, dim.yx*vec2(taper, 1), .01);
    btc = tc; btc.y = abs(btc.y) - dim.x - dim.y;
    float bands2 = sBoxS(btc, vec2(1, 1.5).yx*dim.y, .01);
    //bands2 += smoothstep(0., 1., sin(a*aN*6.))*.001; // Second band ridges.

    // Second strip and second strip coordinates.
    tor2 = smax(tor2, -hole, .02);
    if(min(tor2, bands2)<min(tor, bands)) txCoord = vec3(tc.yx, a/6.2831);


    // Combine the central strips and the outer bands. With odd turn numbers, they'll
    // fuse together as one, and even numbers will produce separate objects.
    tor = min(tor, tor2);
    bands = min(bands, bands2);


    // Overall object ID -- There are two rundundant slots there.
    vID = vec4(fl, bands, 1e5, tor);

    // Shortest distance.
    return  min(min(tor, fl), bands);

}

// Basic raymarcher.
float trace(in vec3 ro, in vec3 rd){

    // Overall ray distance and scene distance.
    float t = 0., d;

    // Back wall unit direction ray.
    // We need the direction itself for the standard cube
    // traversal trickery.
    gDir = step(0., rd) - .5;
    gRd = rd;

    for(int i = ZERO; i<80; i++){

        d = map(ro + rd*t);

        if(abs(d)<.001 || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.), etc.

        t += min(d*.9, gCD);
    }

    return min(t, FAR);
}


// Normal function. It's not as fast as the tetrahedral calculation, but more symmetrical.
vec3 getNormal(in vec3 p, float t){

    //return normalize(vec3(map(p + e.xyy) - map(p - e.xyy),
    //                      map(p + e.yxy) - map(p - e.yxy),
    //                      map(p + e.yyx) - map(p - e.yyx)));

    // This mess is an attempt to speed up compiler time by contriving a break... It's
    // based on a suggestion by IQ. I think it works, but I really couldn't say for sure.
    float sgn = 1.;
    vec3 e = vec3(.001, 0, 0), mp = e.zzz; // Spalmer's clever zeroing.
    for(int i = ZERO; i<6; i++){
		mp.x += map(p + sgn*e)*sgn;
        sgn = -sgn;
        if((i&1)==1){ mp = mp.yzx; e = e.zxy; }
    }

    return normalize(mp);
}


// Cheap shadows are hard. In fact, I'd almost say, shadowing particular scenes with
// limited iterations is impossible... However, I'd be very grateful if someone could
// prove me wrong. :)
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not always affordable.
    const int maxIter = 32;

    // Bumping the ray off the surface to avoid self collisions.
    // The constant coincides with the hit condition in the "trace" function.
    ro += n*.0015;

    vec3 rd = lp - ro; // Unnormalized direction ray.


    float shade = 1.; // Initial shadow value.
    float t = 0.; // Initial distance.
    float end = max(length(rd), 0.0001); // Distance from the jump point to the light.
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down.
    // Obviously, the lowest number to give a decent shadow is the best one to choose.
    for (int i = ZERO; i<maxIter; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Thanks to IQ for this.
        // So many options here, and none are perfect:
        // dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25);


        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break;
    }

    // Return the shadow.
    return max(shade, 0.);
}


// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float calcAO(in vec3 p, in vec3 n)
{
	float sca = 2., occ = 0.;
    for( int i = ZERO; i<5; i++ ){

        float hr = float(i + 1)*.15/5.;
        float d = map(p + n*hr);
        occ += (hr - d)*sca;
        sca *= .7;
        //if(occ>1e5) break; // Faux exit.
    }

    return clamp(1. - occ, 0., 1.);

}



void mainImage( out vec4 fragColor, in vec2 fragCoord ){


    // Screen coordinates.
	vec2 uv = (fragCoord - iResolution.xy*.5)/iResolution.y;

	// Camera Setup.
	vec3 lk = vec3(0, -.1, 0); // "Look At" position.
    vec3 ro = lk + vec3(cos(iTime/2.)*.05, .2, -1.25); // Camera position.
 	vec3 lp = ro + vec3(-1, 1, -.5); // Light position.


    // Using the above to produce the unit ray-direction vector.
    float FOV = 1.; // FOV - Field of view.
    vec3 fwd = normalize(lk - ro);
    vec3 rgt = normalize(vec3(fwd.z, 0., -fwd.x));
    vec3 up = cross(fwd, rgt);

    // rd - Ray direction.
    vec3 rd = normalize(uv.x*rgt + uv.y*up + fwd/FOV);


    // Raymarch to the scene.
    float t = trace(ro, rd);

    // Save the texture coordinates.
    vec3 svTxCoord = txCoord;


    // Obtain the object ID.
    objID = 0;
    float obD = vID[0];

    for(int i = 0; i<4; i++){
        if(vID[i]<obD){ obD = vID[i]; objID = i; }
    }


    // Initiate the scene color to black.
	vec3 col = vec3(0);

	// The ray has effectively hit the surface, so light it up.
	if(t < FAR){


    	// Surface position and surface normal.
	    vec3 sp = ro + rd*t;
	    //vec3 sn = getNormal(sp, edge, crv, ef, t);
        vec3 sn = getNormal(sp, t);


            	// Light direction vector.
	    vec3 ld = lp - sp;

        // Distance from respective light to the surface point.
	    float lDist = max(length(ld), .001);

    	// Normalize the light direction vector.
	    ld /= lDist;


        // Shadows and ambient self shadowing.
    	float sh = softShadow(sp, lp, sn, 8.);
    	float ao = calcAO(sp, sn); // Ambient occlusion.

	    // Light attenuation, based on the distances above.
	    float atten = 1./(1. + lDist*.05);


    	// Diffuse lighting.
	    float diff = max( dot(sn, ld), 0.);

        // Specular lighting.
	    float spec = pow(max(dot(reflect(ld, sn), rd ), 0.), 32.);

        // Obtaining the texel color.
	    vec3 texCol;

        // Object coloring.

        if(objID==0){ // Background.

            // The dark subdivide wall.
            vec3 txP = sp;
            vec3 txN = sn;
            txP.yz *= rot2(3.14159/5.);
            vec4 p4 = getGrid(sp.xy, fSc);


            vec3 tx = tex3D(iChannel1, txP*2. + .5, txN);
            //vec3 tx = texture(iChannel1, txP.xz/3. + .4).xyz; tx *= tx;
            //tx = smoothstep(0., 1., tx);


            texCol = vec3(.4)*(hash21(p4.zw + .1)*.5 + .5);
            texCol *= tx*2. + .1;


        }
        else if(objID==1){ //  Bands.

            // Using the saved coordinates from the distance function
            // to texture the outer bands. The sides aren't technically
            // correct, but no one will notice.

            vec3 tx = getTex(iChannel0, svTxCoord.yz*vec2(4, 4));
            //vec3 tx2 = getTex(iChannel1, svTxCoord.yz*vec2(4, 4));
            //tx = mix(tx, tx2, .5);

            // Gold.
            texCol = vec3(.6, .35, .15)*(tx*2. + .05);
            // Silver -- I almost went with this but decided on gold at
            // the last minute.
            //texCol = vec3(.35)*(tx*2. + .05);

            // Ramping up the diffuse for a more metallic look.
            diff = pow(diff, 8.)*2.;


        }
        else { // Strip.

            // Using the saved coordinates from the distance function
            // to texture the central strip with holes.

            vec3 tx = getTex(iChannel0, svTxCoord.xz*vec2(2, 4));

            // Coloring the individual blocks with the saved ID.
            texCol = tx*.25 + .875;


            //diff = pow(diff, 4.)*2.; // Diffuse ramping.


        }


        // Specular reflection.
        vec3 hv = normalize(ld - rd); // Half vector.
        vec3 ref = reflect(rd, sn); // Surface reflection.
        vec3 refTx = texture(iChannel2, ref).xyz; refTx *= refTx;
        refTx = (texCol*.75 + .25)*refTx;//smoothstep(0., .5, refTx);
        float spRef = pow(max(dot(hv, sn), 0.), 16.); // Specular reflection.
        float rf = objID == 0? .25 : 1.;
        //
        // Adding the specular reflection and glow for the inner light.
        texCol += spRef*refTx*rf*8.;//vec3(1.4, 1, .4)*


        // Combining the above terms to produce the final color.
        col = texCol*(diff*sh + .25 + vec3(1, .97, .92)*spec*1.*sh);


        // Shading.
        col *= ao*atten;



	}

    // Fog -- A bit redundant here, but it does have a minor effect.
    vec3 fog = vec3(1, .925, .85)*.01;
    col = mix(col, fog, smoothstep(0., .99, t/FAR));



    // Rough gamma correction.
	fragColor = vec4(sqrt(max(col, 0.)), 1);

}