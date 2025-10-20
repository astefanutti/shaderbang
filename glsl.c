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
#include <getopt.h>
#include <fcntl.h>
#include <pthread.h>

#include "glsl.h"
#include "drm-common.h"

#include "lease.h"

static const struct egl *egl;
static const struct gbm *gbm;
static const struct drm *drm;

static const char *shortopts = "aAC:D:f:hm:n:p:v:x";

static const struct option longopts[] = {
		{"async",        no_argument,       0, 'a'},
		{"atomic",       no_argument,       0, 'A'},
		{"connector",    required_argument, 0, 'C'},
		{"device",       required_argument, 0, 'D'},
		{"format",       required_argument, 0, 'f'},
		{"help",         no_argument,       0, 'h'},
		{"modifier",     required_argument, 0, 'm'},
		{"frames",       required_argument, 0, 'n'},
		{"perfcntr",     required_argument, 0, 'p'},
		{"vmode",        required_argument, 0, 'v'},
		{"surfaceless",  no_argument,       0, 'x'},
		{0,              0,                 0, 0}
};

static void usage(const char *name) {
	printf("Usage: %s [-aACDfmnpvx] <shader_file>\n"
	       "\n"
	       "options:\n"
	       "    -a, --async              use async page flipping\n"
	       "    -A, --atomic             use atomic mode setting and fencing\n"
	       "    -C, --connector=ID       use the connector with the provided ID (see drm_info)\n"
	       "    -D, --device=DEVICE      use the given device\n"
	       "    -f, --format=FOURCC      framebuffer format\n"
	       "    -h, --help               print usage\n"
	       "    -m, --modifier=MODIFIER  hardcode the selected modifier\n"
	       "    -n, --frames=N           run for the given number of frames and exit\n"
	       "    -p, --perfcntr=LIST      sample specified performance counters using\n"
	       "                             the AMD_performance_monitor extension (comma\n"
	       "                             separated list)\n"
	       "    -v, --vmode=VMODE        specify the video mode in the format\n"
	       "                             <mode>[-<vrefresh>]\n"
	       "    -x, --surfaceless        use surfaceless mode, instead of GBM surface\n",
	       name);
}

int init(const struct options *options) {
	int ret;
	int fd;

	if (options->device) {
		fd = open(options->device, O_RDWR);
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

	uint32_t format = DRM_FORMAT_XRGB8888;
	if (options->format) {
		format = options->format;
	}
	uint64_t modifier = DRM_FORMAT_MOD_INVALID;
	if (options->modifier) {
		modifier = options->modifier;
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

	ret = init_shaderbang(gbm, egl);
	if (ret < 0) {
		return -1;
	}

	//glClearColor((GLfloat) 0.5, (GLfloat) 0.5, (GLfloat) 0.5, (GLfloat) 1.0);
	//glClear(GL_COLOR_BUFFER_BIT);

	return 0;
}

int main(int argc, char *argv[]) {
	const char *shadertoy = NULL;
	const char *perfcntr = NULL;

	struct options options = {
			.connector = -1,
			.frames = 0,
			.mode = "",
	};

	int ret;

	char *p;
	int opt;
	unsigned int len;
	while ((opt = getopt_long_only(argc, argv, shortopts, longopts, NULL)) != -1) {
		switch (opt) {
			case 'a':
				options.async_page_flip = true;
				break;
			case 'A':
				options.atomic_drm_mode = true;
				break;
			case 'C':
				options.connector = strtoul(optarg, NULL, 0);
				break;
			case 'D':
				options.device = optarg;
				break;
			case 'f': {
				char fourcc[4] = "    ";
				uint length = strlen(optarg);
				if (length > 0)
					fourcc[0] = optarg[0];
				if (length > 1)
					fourcc[1] = optarg[1];
				if (length > 2)
					fourcc[2] = optarg[2];
				if (length > 3)
					fourcc[3] = optarg[3];
				options.format = fourcc_code(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
				break;
			}
			case 'h':
				usage(argv[0]);
				return 0;
			case 'm':
				options.modifier = strtoull(optarg, NULL, 0);
				break;
			case 'n':
				options.frames = strtoul(optarg, NULL, 0);
				break;
			case 'p':
				perfcntr = optarg;
				break;
			case 'v':
				p = strchr(optarg, '-');
				if (p == NULL) {
					len = strlen(optarg);
				} else {
					options.vrefresh = strtoul(p + 1, NULL, 0);
					len = p - optarg;
				}
				if (len > sizeof(options.mode) - 1)
					len = sizeof(options.mode) - 1;
				strncpy(options.mode, optarg, len);
				options.mode[len] = '\0';
				break;
			case 'x':
				options.surfaceless = true;
				break;
			default:
				usage(argv[0]);
				return -1;
		}
	}

	if (argc - optind != 1) {
		usage(argv[0]);
		return -1;
	}
	shadertoy = argv[optind];

	ret = init(&options);
	if (ret < 0) {
		return -1;
	}

	if (perfcntr) {
		init_perfcntrs(egl, perfcntr);
	}

	return drm->run(gbm, egl);
}

void *thread_run() {
	eglMakeCurrent(egl->display, egl->surface, egl->surface, egl->context);

	return (void *) drm->run(gbm, egl);
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
