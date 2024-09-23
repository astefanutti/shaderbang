#!/usr/bin/env -S python -m shaderbang.shadertoy

// Copied from https://www.shadertoy.com/view/llGcDm
// Created by https://www.shadertoy.com/user/iq

// Created by inigo quilez - iq/2022

// I'm drawing some shapes in a pixel buffer as usual where pixels are ordered
// in "raster" or "linear", that is, left to right one row at a time, and then
// repositioning all pixels back into the buffer following a hilbert curve.
//
// That is the TRANSFORM set to 0 below. You can set TRANSFORM to 1 to see the
// opposite - drawing shapes into a buffer that has a hilbert curve in it just
// to then stretch it into a linear string that we lay again over the plane in
// regular "raster" order.


// 0: linear  -> hilbert
// 1: hilbert -> linear
#define TRANSFORM 0


// adapted from https://en.wikipedia.org/wiki/Hilbert_curve
int hilbert( ivec2 p, int level )
{
    int d = 0;
    for( int k=0; k<level; k++ )
    {
        int n = level-k-1;
        ivec2 r = (p>>n)&1;
        d += ((3*r.x)^r.y) << (2*n);
        if (r.y == 0) { if (r.x == 1) { p = (1<<n)-1-p; } p = p.yx; }
    }
    return d;
}

// adapted from  https://en.wikipedia.org/wiki/Hilbert_curve
ivec2 ihilbert( int i, int level )
{
    ivec2 p = ivec2(0,0);
    for( int k=0; k<level; k++ )
    {
        ivec2 r = ivec2( i>>1, i^(i>>1) ) & 1;
        if (r.y==0) { if(r.x==1) { p = (1<<k) - 1 - p; } p = p.yx; }
        p += r<<k;
        i >>= 2;
    }
    return p;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // work in integer coordinates please
    ivec2 ip = ivec2(fragCoord);
    ivec2 ir = ivec2(iResolution);

    // select hilbert resolution
    int             level =  7;
    if( ir.x> 512 ) level =  8;
    if( ir.x>1024 ) level =  9;
    if( ir.x>2048 ) level = 10;

    // two square's bottom-left corner coordinates
    int res = (1<<level);
    ivec2 c1 = ivec2( (ir.x-2*res)/3, (ir.y-1*res)/2 );
    ivec2 c2 = ivec2( c1.x+res+c1.x, c1.y );

    // distance to two squares
    ivec2 e1 = abs(ip-c1-res/2)-res/2; int d1 = max(e1.x,e1.y);
    ivec2 e2 = abs(ip-c2-res/2)-res/2; int d2 = max(e2.x,e2.y);

    // twitter "dark" mode background
    vec3 col = vec3(20,30,40)/255.0;
    vec2 p;

    // inside left square
    if( d1<0 )
    {
        #if TRANSFORM==1
        int id = (ip.y-c1.y)*res + (ip.x-c1.x);
        p = vec2( ihilbert(id,level) ) / float(res);
        #else
        int i = hilbert(ip-c1,level);
        p = vec2( i%res, i/res ) / float(res);
        #endif
    }
    // inside right square
    else if( d2<0 )
    {
        p = vec2(ip-c2)/float(res);
    }
    // otherwise, exterior
    else
    {
        // border color
        if( min(d1,d2)<8 ) col*=3.5;
        fragColor = vec4(col,1.0);
        return;
    }

    // animate
    int id = int(floor(iTime/4.0)) % 4;
    float t = 0.5 - 0.5*cos(6.283185*iTime/4.0);

    // render
    float f = 0.0;

    // horizontal line
    if( id==0 ) { f = abs(p.y-t); f = 1.0-smoothstep( 0.00, 0.02, f ); }
    // vertical line
    if( id==1 ) { f = abs(p.x-t); f = 1.0-smoothstep( 0.00, 0.10, f ); }
    // circle
    if( id==2 ) { f = abs(length(p-0.5)-0.5*t); f = 1.0-smoothstep( 0.00, 0.08, f ); }
    // circular waves
    if( id==3 )
    {
        t = -6.283185*2.0*fract(iTime/4.0);
        float l = length(p-0.5);
        f  = 1.0*sin(  20.0*l+0.0 + t*1.0);
        f += 0.5*sin(  56.0*l+1.0 + t*1.3);
        f += 0.3*sin(  82.0*l+2.0 + t*2.1);
        f += 0.2*sin( 132.0*l+3.0 + t*2.7);
        f -= 0.2;
        f = smoothstep(0.0,1.0,f);
    }

    // put color
    col = mix( col, vec3(1.0), f );

    fragColor = vec4( col, 1.0 );
}
