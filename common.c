/*
 * Copyright (c) 2013 Intel Corporation
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
 * Copyright (c) 2019 NVIDIA Corporation
 * Copyright (c) 2024 Antonin Stefanutti <antonin.stefanutti@gmail.com>
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

#include <assert.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "drm-common.h"

static struct gbm gbm;
static struct egl egl;

WEAK struct gbm_surface *
gbm_surface_create_with_modifiers(struct gbm_device *gbm,
				uint32_t width, uint32_t height,
				uint32_t format,
				const uint64_t *modifiers,
				const unsigned int count);
WEAK struct gbm_bo *
gbm_bo_create_with_modifiers(struct gbm_device *gbm,
				uint32_t width, uint32_t height,
				uint32_t format,
				const uint64_t *modifiers,
				const unsigned int count);

const struct gbm *init_gbm_device(const struct drm *drm, uint32_t format)
{
	gbm.drm = drm;

	gbm.dev = gbm_create_device(drm->fd);
	if (!gbm.dev) {
		fprintf(stderr, "Failed to create a GBM device on fd %d\n", drm->fd);
		return NULL;
	}

	gbm.format = format;
	gbm.width = drm->mode->hdisplay;
	gbm.height = drm->mode->vdisplay;
	gbm.surface = NULL;

	return &gbm;
}

static int init_gbm_surface(const uint64_t *modifiers,
                            const unsigned int count)
{
	if (gbm_surface_create_with_modifiers) {
		gbm.surface = gbm_surface_create_with_modifiers(gbm.dev,
		                                                gbm.width,
		                                                gbm.height,
		                                                gbm.format,
		                                                modifiers, count);
	}

	if (!gbm.surface) {
		if (count > 0 && modifiers[0] != DRM_FORMAT_MOD_LINEAR) {
			fprintf(stderr, "Modifiers requested but support isn't available\n");
			return -1;
		}
		gbm.surface = gbm_surface_create(gbm.dev,
		                                 gbm.width,
		                                 gbm.height,
		                                 gbm.format,
		                                 GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
	}

	if (!gbm.surface) {
		printf("Failed to create GBM surface\n");
		return -1;
	}

	return 0;
}

static struct gbm_bo *init_gbm_bo(const uint64_t *modifiers,
                                  const unsigned int count)
{
	struct gbm_bo *bo = NULL;

	if (gbm_bo_create_with_modifiers) {
		bo = gbm_bo_create_with_modifiers(gbm.dev,
		                                  gbm.width, gbm.height,
		                                  gbm.format,
		                                  modifiers, count);
	}

	if (!bo) {
		if (count > 0 && modifiers[0] != DRM_FORMAT_MOD_LINEAR) {
			fprintf(stderr, "Modifiers requested but support isn't available\n");
			return NULL;
		}

		bo = gbm_bo_create(gbm.dev,
		                   gbm.width, gbm.height,
		                   gbm.format,
		                   GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
	}

	if (!bo) {
		printf("Failed to create GBM BO\n");
		return NULL;
	}

	return bo;
}

static int init_gbm_buffer_objects(const uint64_t *modifiers,
                                   const unsigned int count)
{
	for (unsigned i = 0; i < ARRAY_SIZE(gbm.bos); i++) {
		gbm.bos[i] = init_gbm_bo(modifiers, count);
		if (!gbm.bos[i])
			return -1;
	}
	return 0;
}

static bool has_ext(const char *extension_list, const char *ext)
{
	const char *ptr = extension_list;
	size_t len = strlen(ext);

	if (ptr == NULL || *ptr == '\0')
		return false;

	while (true) {
		ptr = strstr(ptr, ext);
		if (!ptr)
			return false;

		if (ptr[len] == ' ' || ptr[len] == '\0')
			return true;

		ptr += len;
	}
}

static int match_config_to_visual(EGLDisplay egl_display,
                                  EGLint visual_id,
                                  EGLConfig *configs,
                                  int count)
{
	for (int i = 0; i < count; ++i) {
		EGLint id;

		if (!eglGetConfigAttrib(egl_display,
				configs[i], EGL_NATIVE_VISUAL_ID,
				&id))
			continue;

		if (id == visual_id) {
			return i;
		}
	}

	return -1;
}

static bool egl_choose_config(const EGLDisplay egl_display, const EGLint *attribs,
                              const EGLint visual_id, EGLConfig *config_out)
{
	EGLint count = 0;
	EGLint matched = 0;
	int config_index = -1;

	if (!eglGetConfigs(egl_display, NULL, 0, &count) || count < 1) {
		printf("No EGL configs to choose from.\n");
		return false;
	}
	EGLConfig *configs = malloc(count * sizeof *configs);
	if (!configs)
		return false;

	if (!eglChooseConfig(egl_display, attribs, configs,
					count, &matched) || !matched) {
		printf("No EGL configs with appropriate attributes.\n");
		goto out;
	}

	if (!visual_id)
		config_index = 0;

	if (config_index == -1)
		config_index = match_config_to_visual(egl_display,
							visual_id,
							configs,
							matched);

	if (config_index != -1)
		*config_out = configs[config_index];

out:
	free(configs);
	if (config_index == -1)
		return false;

	return true;
}

static bool create_framebuffer(const struct egl *egl, struct gbm_bo *bo,
                               struct framebuffer *fb)
{
	assert(egl->eglCreateImageKHR);
	assert(bo);
	assert(fb);

	// 1. Create EGLImage.
	int fd = gbm_bo_get_fd(bo);
	if (fd < 0) {
		printf("failed to get fd for bo: %d\n", fd);
		return false;
	}

	EGLint khr_image_attrs[17] = {
		EGL_WIDTH, gbm_bo_get_width(bo),
		EGL_HEIGHT, gbm_bo_get_height(bo),
		EGL_LINUX_DRM_FOURCC_EXT, (int)gbm_bo_get_format(bo),
		EGL_DMA_BUF_PLANE0_FD_EXT, fd,
		EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
		EGL_DMA_BUF_PLANE0_PITCH_EXT, gbm_bo_get_stride(bo),
		EGL_NONE, EGL_NONE,	/* modifier lo */
		EGL_NONE, EGL_NONE,	/* modifier hi */
		EGL_NONE,
	};

	if (egl->modifiers_supported) {
		const uint64_t modifier = gbm_bo_get_modifier(bo);
		if (modifier != DRM_FORMAT_MOD_LINEAR) {
			size_t attrs_index = 12;
			khr_image_attrs[attrs_index++] =
				EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT;
			khr_image_attrs[attrs_index++] = modifier & 0xfffffffful;
			khr_image_attrs[attrs_index++] =
				EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT;
			khr_image_attrs[attrs_index++] = modifier >> 32;
		}
	}

	fb->image = egl->eglCreateImageKHR(egl->display, EGL_NO_CONTEXT,
			EGL_LINUX_DMA_BUF_EXT, NULL /* no client buffer */,
			khr_image_attrs);

	if (fb->image == EGL_NO_IMAGE_KHR) {
		printf("failed to make image from buffer object\n");
		return false;
	}

	// EGLImage takes the fd ownership
	close(fd);

	// 2. Create GL texture and framebuffer
	glGenTextures(1, &fb->tex);
	glBindTexture(GL_TEXTURE_2D, fb->tex);
	egl->glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, fb->image);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenFramebuffers(1, &fb->fb);
	glBindFramebuffer(GL_FRAMEBUFFER, fb->fb);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
			fb->tex, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		printf("failed framebuffer check for created target buffer\n");
		glDeleteFramebuffers(1, &fb->fb);
		glDeleteTextures(1, &fb->tex);
		return false;
	}

	return true;
}

int init_egl_modifiers(struct egl *egl, const struct drm *drm,
                       unsigned int format)
{
	EGLBoolean *extern_only;
	EGLint i, j;
	unsigned int num_drm_mods;
	const uint64_t *drm_mods = get_drm_format_modifiers(drm, &num_drm_mods);

	if (!egl->eglQueryDmaBufModifiersEXT(egl->display,
	                                     format,
	                                     0,
	                                     NULL,
	                                     NULL,
	                                     &egl->num_modifiers)) {
		printf("Failed to query number of modifiers for format 0x%x\n",
		       format);
		return -1;
	}

	egl->modifiers = malloc(sizeof(egl->modifiers[0]) *
	                        egl->num_modifiers);
	extern_only = malloc(sizeof(extern_only[0]) * egl->num_modifiers);

	if (!egl->modifiers || !extern_only) {
		printf("Failed to allocate modifier array\n");
		free(egl->modifiers);
		free(extern_only);
		egl->modifiers = NULL;
		egl->num_modifiers = 0;
		return -1;
	}

	if (!egl->eglQueryDmaBufModifiersEXT(egl->display,
	                                     format,
	                                     egl->num_modifiers,
	                                     egl->modifiers,
	                                     extern_only,
	                                     &egl->num_modifiers)) {
		printf("Failed to query modifiers for format 0x%x\n", format);
		free(extern_only);
		free(egl->modifiers);
		egl->modifiers = NULL;
		egl->num_modifiers = 0;
		return -1;
	}

	for (i = 0; i < egl->num_modifiers; i++) {
		int remove = 0;
		/* Filter out external-only modifiers. */
		if (extern_only[i]) remove = 1;

		/* Filter out modifiers incompatible with the DRM plane */
		for (j = 0; j < (EGLint) num_drm_mods; j++) {
			if (egl->modifiers[i] == drm_mods[j]) {
				break;
			}
		}

		if (num_drm_mods > 0 && j == (EGLint) num_drm_mods) remove = 1;

		if (remove) {
			/* Shift remaining modifiers down */
			for (j = i + 1; j < egl->num_modifiers; j++) {
				egl->modifiers[j - 1] = egl->modifiers[j];
				extern_only[j - 1] = extern_only[j];
			}
			egl->num_modifiers--;
		}
	}

	free(extern_only);

	if (egl->num_modifiers <= 0) {
		printf("No usable format modifiers found for format 0x%x\n",
		       format);
		free(egl->modifiers);
		egl->modifiers = NULL;
		egl->num_modifiers = 0;
		return -1;
	}

	return 0;
}

const struct egl * init_egl(const struct gbm *gbm, uint64_t modifier, bool surfaceless)
{
	EGLint major, minor;

	const char *egl_exts_client, *egl_exts_dpy, *gl_exts;

	int res;

#define get_proc_client(ext, name) do { \
		if (has_ext(egl_exts_client, #ext)) \
			egl.name = (void *)eglGetProcAddress(#name); \
	} while (0)
#define get_proc_dpy(ext, name) do { \
		if (has_ext(egl_exts_dpy, #ext)) \
			egl.name = (void *)eglGetProcAddress(#name); \
	} while (0)

#define get_proc_gl(ext, name) do { \
		if (has_ext(gl_exts, #ext)) \
			egl.name = (void *)eglGetProcAddress(#name); \
	} while (0)

	egl_exts_client = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
	get_proc_client(EGL_EXT_platform_base, eglGetPlatformDisplayEXT);

	if (egl.eglGetPlatformDisplayEXT) {
		egl.display = egl.eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR,
				gbm->dev, NULL);
	} else {
		egl.display = eglGetDisplay((void *)gbm->dev);
	}

	if (!eglInitialize(egl.display, &major, &minor)) {
		printf("Failed to initialize EGL\n");
		return NULL;
	}

	egl_exts_dpy = eglQueryString(egl.display, EGL_EXTENSIONS);
	get_proc_dpy(EGL_KHR_image_base, eglCreateImageKHR);
	get_proc_dpy(EGL_KHR_image_base, eglDestroyImageKHR);
	get_proc_dpy(EGL_EXT_image_dma_buf_import_modifiers, eglQueryDmaBufModifiersEXT);
	get_proc_dpy(EGL_KHR_fence_sync, eglCreateSyncKHR);
	get_proc_dpy(EGL_KHR_fence_sync, eglDestroySyncKHR);
	get_proc_dpy(EGL_KHR_fence_sync, eglWaitSyncKHR);
	get_proc_dpy(EGL_KHR_fence_sync, eglClientWaitSyncKHR);
	get_proc_dpy(EGL_ANDROID_native_fence_sync, eglDupNativeFenceFDANDROID);

	egl.modifiers_supported = has_ext(egl_exts_dpy,
					"EGL_EXT_image_dma_buf_import_modifiers");

	printf("Using display %p with EGL version %d.%d\n",
			egl.display, major, minor);

	printf("===================================\n");
	printf("EGL information:\n");
	printf("  version: \"%s\"\n", eglQueryString(egl.display, EGL_VERSION));
	printf("  vendor: \"%s\"\n", eglQueryString(egl.display, EGL_VENDOR));
	printf("===================================\n");

	if (!eglBindAPI(EGL_OPENGL_API)) {
		printf("Failed to bind EGL_OPENGL_API\n");
		return NULL;
	}

	const EGLint config_attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_CONFORMANT, EGL_OPENGL_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,

		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_DEPTH_SIZE, 24,
		EGL_STENCIL_SIZE, 8,

		//EGL_RED_SIZE, 1,
		//EGL_GREEN_SIZE, 1,
		//EGL_BLUE_SIZE, 1,
		//EGL_ALPHA_SIZE, 0,

		EGL_NONE
	};

	if (!egl_choose_config(egl.display, config_attribs, gbm->format,
			&egl.config)) {
		printf("Failed to choose EGL config\n");
		return NULL;
	}

	static const EGLint context_attribs[] = {
		//EGL_CONTEXT_CLIENT_VERSION, 5,
		EGL_CONTEXT_MAJOR_VERSION, 4,
		EGL_CONTEXT_MINOR_VERSION, 5,
		//EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
		EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
		EGL_NONE
	};

	egl.context = eglCreateContext(egl.display, egl.config,
			EGL_NO_CONTEXT, context_attribs);
	if (egl.context == EGL_NO_CONTEXT) {
		printf("Failed to create EGL context\n");
		return NULL;
	}

	if (egl.modifiers_supported) {
		if (modifier == DRM_FORMAT_MOD_INVALID &&
		    init_egl_modifiers(&egl, gbm->drm, gbm->format)) {
			printf("Not using modifiers\n");
			egl.modifiers_supported = 0;
			modifier = DRM_FORMAT_MOD_LINEAR;
		}
	}

	int (*init_gbm)(const uint64_t *modifiers, const unsigned int count);
	if (surfaceless) {
		init_gbm = init_gbm_buffer_objects;
	} else {
		init_gbm = init_gbm_surface;
	}

	if (egl.num_modifiers) {
		res = init_gbm(egl.modifiers, egl.num_modifiers);
	} else {
		res = init_gbm(&modifier, 1);
	}
	if (res) {
		fprintf(stderr, "Failed to init GBM surface\n");
		return NULL;
	}

	if (!gbm->surface) {
		egl.surface = EGL_NO_SURFACE;
	} else {
		egl.surface = eglCreateWindowSurface(egl.display, egl.config,
				(EGLNativeWindowType)gbm->surface, NULL);
		if (egl.surface == EGL_NO_SURFACE) {
			printf("Failed to create EGL surface\n");
			return NULL;
		}
	}

	/* connect the context to the surface */
	eglMakeCurrent(egl.display, egl.surface, egl.surface, egl.context);

	gl_exts = (char *) glGetString(GL_EXTENSIONS);
	printf("OpenGL information:\n");
	printf("  version: \"%s\"\n", glGetString(GL_VERSION));
	printf("  shading language version: \"%s\"\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("  vendor: \"%s\"\n", glGetString(GL_VENDOR));
	printf("  renderer: \"%s\"\n", glGetString(GL_RENDERER));
	printf("===================================\n");

	get_proc_gl(GL_OES_EGL_image, glEGLImageTargetTexture2DOES);

	get_proc_gl(GL_AMD_performance_monitor, glGetPerfMonitorGroupsAMD);
	get_proc_gl(GL_AMD_performance_monitor, glGetPerfMonitorCountersAMD);
	get_proc_gl(GL_AMD_performance_monitor, glGetPerfMonitorGroupStringAMD);
	get_proc_gl(GL_AMD_performance_monitor, glGetPerfMonitorCounterStringAMD);
	get_proc_gl(GL_AMD_performance_monitor, glGetPerfMonitorCounterInfoAMD);
	get_proc_gl(GL_AMD_performance_monitor, glGenPerfMonitorsAMD);
	get_proc_gl(GL_AMD_performance_monitor, glDeletePerfMonitorsAMD);
	get_proc_gl(GL_AMD_performance_monitor, glSelectPerfMonitorCountersAMD);
	get_proc_gl(GL_AMD_performance_monitor, glBeginPerfMonitorAMD);
	get_proc_gl(GL_AMD_performance_monitor, glEndPerfMonitorAMD);
	get_proc_gl(GL_AMD_performance_monitor, glGetPerfMonitorCounterDataAMD);

	if (!gbm->surface) {
		for (unsigned i = 0; i < ARRAY_SIZE(gbm->bos); i++) {
			if (!create_framebuffer(&egl, gbm->bos[i], &egl.fbs[i])) {
				printf("Failed to create framebuffer\n");
				return NULL;
			}
		}
	}

	return &egl;
}

int create_program(const char *vs_src, const char *fs_src)
{
	GLuint vertex_shader, fragment_shader, program;
	GLint ret;

	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	if (vertex_shader == 0) {
		printf("vertex shader creation failed!\n");
		return -1;
	}

	glShaderSource(vertex_shader, 1, &vs_src, NULL);
	glCompileShader(vertex_shader);

	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &ret);
	if (!ret) {
		char *log;

		printf("vertex shader compilation failed!:\n");
		glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &ret);
		if (ret > 1) {
			log = malloc(ret);
			glGetShaderInfoLog(vertex_shader, ret, NULL, log);
			printf("%s", log);
			free(log);
		}

		return -1;
	}

	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	if (fragment_shader == 0) {
		printf("fragment shader creation failed!\n");
		return -1;
	}

	glShaderSource(fragment_shader, 1, &fs_src, NULL);
	glCompileShader(fragment_shader);

	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &ret);
	if (!ret) {
		char *log;

		printf("fragment shader compilation failed!:\n");
		glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &ret);

		if (ret > 1) {
			log = malloc(ret);
			glGetShaderInfoLog(fragment_shader, ret, NULL, log);
			printf("%s", log);
			free(log);
		}

		return -1;
	}

	program = glCreateProgram();

	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);

	return program;
}

int link_program(unsigned program)
{
	GLint ret;

	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &ret);
	if (!ret) {
		char *log;

		printf("program linking failed!:\n");
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &ret);

		if (ret > 1) {
			log = malloc(ret);
			glGetProgramInfoLog(program, ret, NULL, log);
			printf("%s", log);
			free(log);
		}

		return -1;
	}

	return 0;
}

uint64_t get_time_ns(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_MONOTONIC, &tv);
	return tv.tv_nsec + tv.tv_sec * NSEC_PER_SEC;
}

Callbacks onInitCallbacks;
Callbacks onRenderCallbacks;

void addCallback(Callbacks *callbacks, void callback()) {
	if (!callbacks->callbacks) {
		callbacks->length = 1;
		callbacks->callbacks = malloc(sizeof(callbacks->callbacks));
		callbacks->callbacks[0] = callback;
	} else {
		callbacks->length++;
		callbacks->callbacks = realloc(callbacks->callbacks, callbacks->length * sizeof(callbacks->callbacks));
		callbacks->callbacks[callbacks->length - 1] = callback;
	}
}

void onInit(const onInitCallback callback) {
	addCallback(&onInitCallbacks, (void (*)) callback);
}

void onRender(const onRenderCallback callback) {
	addCallback(&onRenderCallbacks, (void (*)) callback);
}

void callInitCallbacks(const unsigned int width, const unsigned int height) {
	for (uint i = 0; i < onInitCallbacks.length; i++) {
		((onInitCallback) onInitCallbacks.callbacks[i])(width, height);
	}
}

void callRenderCallbacks(const uint64_t frame, const float time) {
	for (uint i = 0; i < onRenderCallbacks.length; i++) {
		((onRenderCallback) onRenderCallbacks.callbacks[i])(frame, time);
	}
}
