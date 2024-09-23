/*
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _COMMON_H
#define _COMMON_H

#ifndef GL_ES_VERSION_2_0
#include <GLES2/gl2.h>
#endif
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <gbm.h>
#include <drm_fourcc.h>
#include <stdbool.h>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/* from mesa's util/macros.h: */
#define MIN2( A, B )   ( (A)<(B) ? (A) : (B) )
#define MAX2( A, B )   ( (A)>(B) ? (A) : (B) )
#define MIN3( A, B, C ) ((A) < (B) ? MIN2(A, C) : MIN2(B, C))
#define MAX3( A, B, C ) ((A) > (B) ? MAX2(A, C) : MAX2(B, C))

static inline unsigned u_minify(unsigned value, unsigned levels) {
	return MAX2(1, value >> levels);
}

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0
#endif

#ifndef DRM_FORMAT_MOD_INVALID
#define DRM_FORMAT_MOD_INVALID ((((__u64)0) << 56) | ((1ULL << 56) - 1))
#endif

#ifndef EGL_KHR_platform_gbm
#define EGL_KHR_platform_gbm 1
#define EGL_PLATFORM_GBM_KHR              0x31D7
#endif /* EGL_KHR_platform_gbm */

#ifndef EGL_EXT_platform_base
#define EGL_EXT_platform_base 1
typedef EGLDisplay (EGLAPIENTRYP PFNEGLGETPLATFORMDISPLAYEXTPROC) (EGLenum platform, void *native_display, const EGLint *attrib_list);
typedef EGLSurface (EGLAPIENTRYP PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC) (EGLDisplay dpy, EGLConfig config, void *native_window, const EGLint *attrib_list);
typedef EGLSurface (EGLAPIENTRYP PFNEGLCREATEPLATFORMPIXMAPSURFACEEXTPROC) (EGLDisplay dpy, EGLConfig config, void *native_pixmap, const EGLint *attrib_list);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLDisplay EGLAPIENTRY eglGetPlatformDisplayEXT (EGLenum platform, void *native_display, const EGLint *attrib_list);
EGLAPI EGLSurface EGLAPIENTRY eglCreatePlatformWindowSurfaceEXT (EGLDisplay dpy, EGLConfig config, void *native_window, const EGLint *attrib_list);
EGLAPI EGLSurface EGLAPIENTRY eglCreatePlatformPixmapSurfaceEXT (EGLDisplay dpy, EGLConfig config, void *native_pixmap, const EGLint *attrib_list);
#endif
#endif /* EGL_EXT_platform_base */

#ifndef EGL_VERSION_1_5
#define EGLImage EGLImageKHR
#endif /* EGL_VERSION_1_5 */

#define WEAK __attribute__((weak))

/* Define tokens and proc types from EGL_EXT_image_dma_buf_import_modifiers */
#ifndef EGL_EXT_image_dma_buf_import_modifiers
#define EGL_EXT_image_dma_buf_import_modifiers 1
#define EGL_DMA_BUF_PLANE3_FD_EXT         0x3440
#define EGL_DMA_BUF_PLANE3_OFFSET_EXT     0x3441
#define EGL_DMA_BUF_PLANE3_PITCH_EXT      0x3442
#define EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT 0x3443
#define EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT 0x3444
#define EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT 0x3445
#define EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT 0x3446
#define EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT 0x3447
#define EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT 0x3448
#define EGL_DMA_BUF_PLANE3_MODIFIER_LO_EXT 0x3449
#define EGL_DMA_BUF_PLANE3_MODIFIER_HI_EXT 0x344A

typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYDMABUFMODIFIERSEXTPROC) (EGLDisplay dpy, EGLint format, EGLint max_modifiers, EGLuint64KHR *modifiers, EGLBoolean *external_only, EGLint *num_modifiers);
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglQueryDmaBufModifiersEXT (EGLDisplay dpy, EGLint format, EGLint max_modifiers, EGLuint64KHR *modifiers, EGLBoolean *external_only, EGLint *num_modifiers);
#endif
#endif /* EGL_EXT_image_dma_buf_import_modifiers */

#define NUM_BUFFERS 2

struct options {
	const char *device;
	char mode[DRM_DISPLAY_MODE_LEN];
	uint32_t format;
	uint64_t modifier;
	int connector;
	bool async_page_flip;
	bool atomic_drm_mode;
	bool surfaceless;
	unsigned int vrefresh;
	unsigned int frames;
};

typedef int (*onInitCallback)(unsigned int width, unsigned int height);
typedef int (*onRenderCallback)(uint64_t frame, float time);

typedef struct {
	void (**callbacks)();
	size_t length;
} Callbacks;

int callInitCallbacks(unsigned int width, unsigned int height);
int callRenderCallbacks(uint64_t frame, float time);

struct gbm {
	const struct drm *drm;
	struct gbm_device *dev;
	struct gbm_surface *surface;
	struct gbm_bo *bos[NUM_BUFFERS];    /* for the surfaceless case */
	uint32_t format;
	int width, height;
};

const struct gbm * init_gbm_device(const struct drm *drm, uint32_t format);

struct framebuffer {
	EGLImageKHR image;
	GLuint tex;
	GLuint fb;
};

struct egl {
	EGLDisplay display;
	EGLConfig config;
	EGLContext context;
	EGLSurface surface;
	struct framebuffer fbs[NUM_BUFFERS];    /* for the surfaceless case */

	PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT;
	PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR;
	PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR;
	PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES;
	PFNEGLQUERYDMABUFMODIFIERSEXTPROC eglQueryDmaBufModifiersEXT;
	PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR;
	PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;
	PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR;
	PFNEGLCLIENTWAITSYNCKHRPROC eglClientWaitSyncKHR;
	PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFDANDROID;

	/* AMD_performance_monitor */
	PFNGLGETPERFMONITORGROUPSAMDPROC         glGetPerfMonitorGroupsAMD;
	PFNGLGETPERFMONITORCOUNTERSAMDPROC       glGetPerfMonitorCountersAMD;
	PFNGLGETPERFMONITORGROUPSTRINGAMDPROC    glGetPerfMonitorGroupStringAMD;
	PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC  glGetPerfMonitorCounterStringAMD;
	PFNGLGETPERFMONITORCOUNTERINFOAMDPROC    glGetPerfMonitorCounterInfoAMD;
	PFNGLGENPERFMONITORSAMDPROC              glGenPerfMonitorsAMD;
	PFNGLDELETEPERFMONITORSAMDPROC           glDeletePerfMonitorsAMD;
	PFNGLSELECTPERFMONITORCOUNTERSAMDPROC    glSelectPerfMonitorCountersAMD;
	PFNGLBEGINPERFMONITORAMDPROC             glBeginPerfMonitorAMD;
	PFNGLENDPERFMONITORAMDPROC               glEndPerfMonitorAMD;
	PFNGLGETPERFMONITORCOUNTERDATAAMDPROC    glGetPerfMonitorCounterDataAMD;

	bool modifiers_supported;

	EGLuint64KHR *modifiers;
	EGLint num_modifiers;

	int (*draw)(uint64_t start_time, unsigned frame);
};

static inline int __egl_check(void *ptr, const char *name) {
	if (!ptr) {
		printf("no %s\n", name);
		return -1;
	}
	return 0;
}

#define egl_check(egl, name) __egl_check((egl)->name, #name)

const struct egl * init_egl(const struct gbm *gbm, uint64_t modifier, bool surfaceless);

int create_program(const char *vs_src, const char *fs_src);
int link_program(unsigned program);

int init_shaderbang(const struct gbm *gbm, struct egl *egl);
int init_shadertoy(const struct gbm *gbm, struct egl *egl, const char *shadertoy);

void init_perfcntrs(const struct egl *egl, const char *perfcntrs);
void start_perfcntrs(void);
void end_perfcntrs(void);
void finish_perfcntrs(void);
void dump_perfcntrs(unsigned nframes, uint64_t elapsed_time_ns);

#define NSEC_PER_SEC (INT64_C(1000) * USEC_PER_SEC)
#define USEC_PER_SEC (INT64_C(1000) * MSEC_PER_SEC)
#define MSEC_PER_SEC INT64_C(1000)

uint64_t get_time_ns(void);

#endif /* _COMMON_H */
