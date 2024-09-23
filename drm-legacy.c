/*
 * Copyright (c) 2017 Rob Clark <rclark@redhat.com>
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
#include <stdio.h>
#include <string.h>
#include <sys/select.h>

#include "common.h"
#include "drm-common.h"

static struct drm drm;

static void page_flip_handler(int fd, unsigned int frame,
                              unsigned int sec, unsigned int usec, void *data)
{
	/* suppress 'unused parameter' warnings */
	(void) fd, (void) frame, (void) sec, (void) usec;

	int *waiting_for_flip = data;
	*waiting_for_flip = 0;
}

static int legacy_run(const struct gbm *gbm, const struct egl *egl)
{
	fd_set fds;
	drmEventContext evctx = {
			.version = 2,
			.page_flip_handler = page_flip_handler,
	};
	struct gbm_bo *bo;
	struct drm_fb *fb;
	uint32_t i = 0;
	uint64_t start_time, report_time, cur_time;
	int ret;

	if (gbm->surface) {
		eglSwapBuffers(egl->display, egl->surface);
		bo = gbm_surface_lock_front_buffer(gbm->surface);
	} else {
		bo = gbm->bos[0];
	}
	fb = drm_fb_get_from_bo(bo);
	if (!fb) {
		fprintf(stderr, "Failed to get a new framebuffer BO\n");
		return -1;
	}

	/* set mode: */
	ret = drmModeSetCrtc(drm.fd, drm.crtc_id, fb->fb_id, 0, 0,
	                     &drm.connector_id, 1, drm.mode);
	if (ret) {
		printf("Failed to set mode: %s\n", strerror(errno));
		return ret;
	}

	uint32_t flags;

	if (drm.async_page_flip) {
		flags = DRM_MODE_PAGE_FLIP_ASYNC;
	} else {
		flags = DRM_MODE_PAGE_FLIP_EVENT;
	}

	start_time = report_time = get_time_ns();

	while (drm.frames == 0 || i < drm.frames) {
		unsigned frame = i;
		struct gbm_bo *next_bo;
		int waiting_for_flip = 1;

		/* Start fps measuring on second frame, to remove the time spent
		 * compiling shader, etc, from the fps:
		 */
		if (i == 1) {
			start_time = report_time = get_time_ns();
		}

		if (!gbm->surface) {
			glBindFramebuffer(GL_FRAMEBUFFER, egl->fbs[frame % NUM_BUFFERS].fb);
		}

		egl->draw(start_time, i++);

		/* Block until all the buffered GL operations are completed.
		 * This is required on NVIDIA GPUs, for which the DRM drivers
		 * do not wait for the rendering to complete, upon executing
		 * page flipping operations, such as drmModePageFlip().
		 */
		glFinish();

		if (gbm->surface) {
			eglSwapBuffers(egl->display, egl->surface);
			next_bo = gbm_surface_lock_front_buffer(gbm->surface);
		} else {
			next_bo = gbm->bos[frame % NUM_BUFFERS];
		}
		fb = drm_fb_get_from_bo(next_bo);
		if (!fb) {
			fprintf(stderr, "Failed to get a new framebuffer BO\n");
			return -1;
		}

		/*
		 * Here you could also update drm plane layers if you want
		 * hw composition
		 */

		ret = drmModePageFlip(drm.fd, drm.crtc_id, fb->fb_id,
		                      flags, &waiting_for_flip);
		if (ret) {
			printf("failed to queue page flip: %s\n", strerror(errno));
			return -1;
		}

		if (!drm.async_page_flip) {
			while (waiting_for_flip) {
				FD_ZERO(&fds);
				FD_SET(0, &fds);
				FD_SET(drm.fd, &fds);

				ret = select(drm.fd + 1, &fds, NULL, NULL, NULL);
				if (ret < 0) {
					printf("select err: %s\n", strerror(errno));
					return ret;
				}
				if (ret == 0) {
					printf("select timeout!\n");
					return -1;
				}
				if (FD_ISSET(0, &fds)) {
					printf("user interrupted!\n");
					return 0;
				}
				drmHandleEvent(drm.fd, &evctx);
			}
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

		/* release last buffer to render on again: */
		if (gbm->surface) {
			gbm_surface_release_buffer(gbm->surface, bo);
		}
		bo = next_bo;
	}

	finish_perfcntrs();

	cur_time = get_time_ns();
	double elapsed_time = cur_time - start_time;
	double secs = elapsed_time / (double) NSEC_PER_SEC;
	unsigned frames = i - 1;  /* first frame ignored */
	printf("Rendered %u frames in %f sec (%f fps)\n",
	       frames, secs, (double) frames / secs);

	dump_perfcntrs(frames, elapsed_time);

	return 0;
}

const struct drm * init_drm_legacy(int fd, const struct options *options)
{
	int ret;

	ret = drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
	if (ret) {
		printf("No universal planes support: %s\n", strerror(errno));
		return NULL;
	}

	ret = init_drm(&drm, fd, options);
	if (ret)
		return NULL;

	drm.run = legacy_run;

	return &drm;
}
