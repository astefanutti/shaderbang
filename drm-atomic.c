/*
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
 * Copyright (c) 2019 NVIDIA Corporation
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

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "drm-common.h"

static struct drm drm;

static int add_connector_property(drmModeAtomicReq *req, uint32_t obj_id,
                                  const char *name, uint64_t value)
{
	struct connector *obj = drm.connector;
	unsigned int i;
	int prop_id = 0;

	for (i = 0; i < obj->props->count_props; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			prop_id = obj->props_info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0) {
		printf("No connector property: %s\n", name);
		return -EINVAL;
	}

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
}

static int add_crtc_property(drmModeAtomicReq *req, uint32_t obj_id,
                             const char *name, uint64_t value)
{
	struct crtc *obj = drm.crtc;
	unsigned int i;
	int prop_id = -1;

	for (i = 0; i < obj->props->count_props; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			prop_id = obj->props_info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0) {
		printf("No CRTC property: %s\n", name);
		return -EINVAL;
	}

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
}

static int add_plane_property(drmModeAtomicReq *req, uint32_t obj_id,
                              const char *name, uint64_t value)
{
	struct plane *obj = drm.plane;
	unsigned int prop_idx;
	int res = find_plane_prop(&drm, name, &prop_idx);
	const drmModePropertyRes *prop_info;

	if (res) return res;

	prop_info = obj->props_info[prop_idx];
	return drmModeAtomicAddProperty(req, obj_id, prop_info->prop_id, value);
}

static int drm_atomic_commit(uint32_t fb_id, uint32_t flags)
{
	drmModeAtomicReq *req;
	uint32_t plane_id = drm.plane->plane->plane_id;
	uint32_t blob_id;
	int ret;

	req = drmModeAtomicAlloc();

	if (flags & DRM_MODE_ATOMIC_ALLOW_MODESET) {
		if (add_connector_property(req, drm.connector_id, "CRTC_ID",
		                           drm.crtc_id) < 0)
			return -1;

		if (drmModeCreatePropertyBlob(drm.fd, drm.mode, sizeof(*drm.mode),
		                              &blob_id) != 0)
			return -1;

		if (add_crtc_property(req, drm.crtc_id, "MODE_ID", blob_id) < 0)
			return -1;

		if (add_crtc_property(req, drm.crtc_id, "ACTIVE", 1) < 0)
			return -1;
	}

	add_plane_property(req, plane_id, "FB_ID", fb_id);
	add_plane_property(req, plane_id, "CRTC_ID", drm.crtc_id);
	add_plane_property(req, plane_id, "SRC_X", 0);
	add_plane_property(req, plane_id, "SRC_Y", 0);
	add_plane_property(req, plane_id, "SRC_W", drm.mode->hdisplay << 16);
	add_plane_property(req, plane_id, "SRC_H", drm.mode->vdisplay << 16);
	add_plane_property(req, plane_id, "CRTC_X", 0);
	add_plane_property(req, plane_id, "CRTC_Y", 0);
	add_plane_property(req, plane_id, "CRTC_W", drm.mode->hdisplay);
	add_plane_property(req, plane_id, "CRTC_H", drm.mode->vdisplay);

	ret = drmModeAtomicCommit(drm.fd, req, flags, NULL);

	drmModeAtomicFree(req);

	return ret;
}

static void page_flip_handler(int fd, unsigned int frame,
                              unsigned int sec, unsigned int usec, void *data)
{
	/* suppress 'unused parameter' warnings */
	(void) fd, (void) frame, (void) sec, (void) usec, (void) data;
	//	printf("page flip event occurred: %12.6f\n", sec + (usec / 1000000.0));
}

static int atomic_run(const struct gbm *gbm, const struct egl *egl)
{
	struct gbm_bo *bo = NULL;
	struct drm_fb *fb;
	uint32_t i = 0;
	uint64_t start_time, report_time, cur_time;
	int ret;

	uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;
	if (drm.async_page_flip) {
		flags |= DRM_MODE_PAGE_FLIP_ASYNC;
	} else {
		flags |= DRM_MODE_PAGE_FLIP_EVENT;
	}

	drmEventContext evctx = {
			.version = 4,
			.page_flip_handler = page_flip_handler
	};

	/* Allow a modeset change for the first commit only. */
	flags |= DRM_MODE_ATOMIC_ALLOW_MODESET;

	start_time = report_time = get_time_ns();

	while (drm.frames == 0 || i < drm.frames) {
		unsigned frame = i;
		struct gbm_bo *next_bo;

		/* Start fps measuring on second frame, to remove the time spent
		 * compiling shader, etc, from the fps:
		 */
		if (i == 1) {
			start_time = report_time = get_time_ns();
		}

		if (!gbm->surface) {
			glBindFramebuffer(GL_FRAMEBUFFER, egl->fbs[frame % NUM_BUFFERS].fb);
		}

		ret = egl->draw(start_time, i++);
		if (ret) {
			return -1;
		}

		/* Block until all the buffered GL operations are completed.
		 * This is required on NVIDIA GPUs, for which the DRM drivers
		 * do not wait for the rendering to complete, upon executing
		 * page flipping operations.
		 */
		glFinish();

		if (gbm->surface) {
		    eglSwapBuffers(egl->display, egl->surface);
			next_bo = gbm_surface_lock_front_buffer(gbm->surface);
		} else {
			next_bo = gbm->bos[frame % NUM_BUFFERS];
		}
		if (!next_bo) {
			printf("Failed to lock front buffer\n");
			return -1;
		}
		fb = drm_fb_get_from_bo(next_bo);
		if (!fb) {
			printf("Failed to get a new framebuffer BO\n");
			return -1;
		}

		cur_time = get_time_ns();
		if (cur_time > (report_time + 2 * NSEC_PER_SEC)) {
			double elapsed_time = cur_time - start_time;
			double secs = elapsed_time / (double) NSEC_PER_SEC;
			unsigned frames = i - 1;  /* first frame ignored */
			printf("Rendered %u frames in %f sec (%f fps)\n",
			       frames, secs, (double) frames / secs);
			report_time = cur_time;
		}

		/* Check for user input: */
		struct pollfd fdset[] = {
				{
						.fd = STDIN_FILENO,
						.events = POLLIN,
				}
		};
		ret = poll(fdset, ARRAY_SIZE(fdset), 0);
		if (ret > 0) {
			printf("user interrupted!\n");
			return 0;
		}

		/*
		 * Here you could also update drm plane layers if you want
		 * hw composition
		 */
		ret = drm_atomic_commit(fb->fb_id, flags);
		if (ret) {
			printf("failed to commit: %s\n", strerror(errno));
			return -1;
		}

		if (!drm.async_page_flip) {
			ret = drmHandleEvent(drm.fd, &evctx);
			if (ret) {
				printf("failed to wait for page flip completion\n");
				return -1;
			}
		}

		/* release last buffer to render on again: */
		if (bo && gbm->surface)
			gbm_surface_release_buffer(gbm->surface, bo);
		bo = next_bo;

		/* Allow a modeset change for the first commit only. */
		flags &= ~(DRM_MODE_ATOMIC_ALLOW_MODESET);
	}

	finish_perfcntrs();

	cur_time = get_time_ns();
	double elapsed_time = cur_time - start_time;
	double secs = elapsed_time / (double) NSEC_PER_SEC;
	unsigned frames = i - 1;  /* first frame ignored */
	printf("Rendered %u frames in %f sec (%f fps)\n",
	       frames, secs, (double) frames / secs);

	dump_perfcntrs(frames, elapsed_time);

	return ret;
}

const struct drm * init_drm_atomic(int fd, const struct options *options)
{
	int ret;

	ret = drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);
	if (ret) {
		printf("No atomic mode setting support: %s\n", strerror(errno));
		return NULL;
	}

	ret = init_drm(&drm, fd, options);
	if (ret)
		return NULL;

	drm.run = atomic_run;

	return &drm;
}
