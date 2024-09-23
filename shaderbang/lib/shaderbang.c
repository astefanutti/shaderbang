/*
 * Copyright (c) 2012 Arvin Schnell <arvin.schnell@gmail.com>
 * Copyright (c) 2012 Rob Clark <rob@ti.com>
 * Copyright (c) 2020 Antonin Stefanutti <antonin.stefanutti@gmail.com>
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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <pthread.h>

#include "shaderbang.h"
#include "drm-common.h"

#include "lease.h"

static const struct egl *egl;
static const struct gbm *gbm;
static const struct drm *drm;

static const drmModeCrtc *crtc;

static void cleanup(void) {
	if (drm && crtc) {
		// Restore previous mode
		drmModeSetCrtc(drm->fd, crtc->crtc_id, crtc->buffer_id, crtc->x, crtc->y,
		               &drm->connector_id, 1, &crtc->mode);
	}
}

int render(const u_int64_t start_time, const uint frame) {
	const float time = (float) (get_time_ns() - start_time) / NSEC_PER_SEC;

	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	const int ret = callRenderCallbacks(frame, time);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

	pthread_testcancel();

	return ret;
}

int init(const struct options *options) {
	int ret;
	int fd;

	setbuf(stdout, 0);

	atexit(cleanup);

	uint32_t format = DRM_FORMAT_XRGB8888;
	if (options->format) {
		format = options->format;
	}

	uint64_t modifier = DRM_FORMAT_MOD_INVALID;
	if (options->modifier) {
		modifier = options->modifier;
	}

	if (options->device) {
		fd = open(options->device, O_RDWR | O_NONBLOCK | O_CLOEXEC);
	} else {
#if XCB_LEASE
		xcb_connection_t *connection;
		int screen;

		connection = xcb_connect(NULL, &screen);
		int err = xcb_connection_has_error(connection);
		if (err > 0) {
			printf("Connection attempt to X server failed with error %d, falling back to DRM\n", err);
			xcb_disconnect(connection);

			fd = find_drm_device();
		} else {
			xcb_randr_query_version_cookie_t rqv_c = xcb_randr_query_version(connection,XCB_RANDR_MAJOR_VERSION,XCB_RANDR_MINOR_VERSION);
			xcb_randr_query_version_reply_t *rqv_r = xcb_randr_query_version_reply(connection, rqv_c, NULL);
			if (!rqv_r || rqv_r->minor_version < 6) {
				printf("No new-enough RandR version: %d\n", rqv_r->minor_version);
				return -1;
			}
			free(rqv_r);

			fd = xcb_lease(connection, &screen);
		}
#else
		fd = find_drm_device();
#endif
	}
	if (fd < 0) {
		printf("could not open DRM device\n");
		return -1;
	}

	if (options->atomic_drm_mode) {
		drm = init_drm_atomic(fd, options);
	} else {
		drm = init_drm_legacy(fd, options);
	}
	if (!drm) {
		printf("failed to initialize %s DRM\n", options->atomic_drm_mode ? "atomic" : "legacy");
		return -1;
	}

	crtc = drmModeGetCrtc(drm->fd, drm->crtc_id);
	if (!crtc) {
		printf("failed to get CRTC\n");
		return -1;
	}

	gbm = init_gbm_device(drm, format);
	if (!gbm) {
		printf("failed to initialize GBM\n");
		return -1;
	}

	egl = init_egl(gbm, modifier, options->surfaceless);
	if (!egl) {
		printf("failed to initialize EGL\n");
		return -1;
	}

	ret = callInitCallbacks(gbm->width, gbm->height);
	if (ret < 0) {
		return -1;
	}

	return 0;
}

void *thread_run() {
	eglMakeCurrent(egl->display, egl->surface, egl->surface, egl->context);

	return (void*) (long) drm->run(gbm, egl, &render);
}

volatile pthread_t thread;

int run() {
	eglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

	return pthread_create(&thread, NULL, thread_run, NULL);
}

int join() {
    return pthread_join(thread, NULL);
}

void stop() {
    pthread_cancel(thread);
}
