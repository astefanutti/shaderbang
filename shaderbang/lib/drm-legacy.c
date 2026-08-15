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
#include <pthread.h>
#include <stdio.h>
#include <string.h>

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

static int legacy_run(const struct gbm *gbm, const struct egl *egl, int (*render)(uint64_t start_time, uint frame))
{
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

		ret = render(start_time, i++);
		if (ret) {
			return -1;
		}

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
			ret = drm_wait_flip(&drm, &waiting_for_flip, &evctx);
			if (ret > 0)
				return 0;
			if (ret < 0)
				return ret;
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

	cur_time = get_time_ns();
	double elapsed_time = cur_time - start_time;
	double secs = elapsed_time / (double) NSEC_PER_SEC;
	unsigned frames = i - 1;  /* first frame ignored */
	printf("Rendered %u frames in %f sec (%f fps)\n",
	       frames, secs, (double) frames / secs);

	return 0;
}

/*
 * Triple-buffered variant of the render loop: after queueing a page
 * flip, the next frame is immediately rendered into a third buffer,
 * instead of blocking until the flip completes. The pending flip is
 * only waited for right before queueing the next one, so at most one
 * flip is outstanding at any time, while one buffer is scanned out
 * and another one is being rendered into. This prevents a frame whose
 * rendering takes slightly longer than a vblank interval from
 * quantizing the frame rate down to the next vblank multiple, without
 * introducing tearing: flips remain synchronized to vblank.
 */
static int legacy_run_triple(const struct gbm *gbm, const struct egl *egl, int (*render)(uint64_t start_time, uint frame))
{
	drmEventContext evctx = {
			.version = 2,
			.page_flip_handler = page_flip_handler,
	};
	/* volatile: these live across pthread_cleanup_push, which glibc
	 * implements with setjmp, and would be clobbered otherwise: */
	struct gbm_bo *volatile bo = NULL;
	struct gbm_bo *volatile pending_bo = NULL;
	struct drm_fb *fb;
	volatile uint32_t i = 0;
	uint64_t start_time, report_time, cur_time;
	int waiting_for_flip = 0;
	struct flip_drain drain = {
			.drm = &drm,
			.waiting_for_flip = &waiting_for_flip,
			.evctx = &evctx,
	};
	volatile int err = 0;
	int ret;

	eglSwapBuffers(egl->display, egl->surface);
	bo = gbm_surface_lock_front_buffer(gbm->surface);
	if (!bo) {
		fprintf(stderr, "Failed to lock front buffer\n");
		return -1;
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

	start_time = report_time = get_time_ns();

	/* Make sure a pending flip never outlives this loop, in
	 * particular when the render thread gets cancelled: */
	pthread_cleanup_push(drm_drain_flip, &drain);

	while (drm.frames == 0 || i < drm.frames) {
		struct gbm_bo *next_bo;

		/* Start fps measuring on second frame, to remove the time spent
		 * compiling shader, etc, from the fps:
		 */
		if (i == 1) {
			start_time = report_time = get_time_ns();
		}

		/* Back-pressure: if the GBM surface has run out of free
		 * buffers to render into, wait for the pending flip to
		 * retire the current scan-out buffer first:
		 */
		if (waiting_for_flip && !gbm_surface_has_free_buffers(gbm->surface)) {
			ret = drm_wait_flip(&drm, &waiting_for_flip, &evctx);
			if (ret) {
				err = ret < 0 ? ret : 0;
				break;
			}
			/* the flip away from the previous scan-out buffer has
			 * completed, release it to render on again: */
			gbm_surface_release_buffer(gbm->surface, bo);
			bo = pending_bo;
			pending_bo = NULL;
		}

		ret = render(start_time, i++);
		if (ret) {
			err = -1;
			break;
		}

		/* Block until all the buffered GL operations are completed.
		 * This is required on NVIDIA GPUs, for which the DRM drivers
		 * do not wait for the rendering to complete, upon executing
		 * page flipping operations, such as drmModePageFlip().
		 */
		glFinish();

		eglSwapBuffers(egl->display, egl->surface);
		next_bo = gbm_surface_lock_front_buffer(gbm->surface);
		if (!next_bo) {
			fprintf(stderr, "Failed to lock front buffer\n");
			err = -1;
			break;
		}
		fb = drm_fb_get_from_bo(next_bo);
		if (!fb) {
			fprintf(stderr, "Failed to get a new framebuffer BO\n");
			err = -1;
			break;
		}

		/* Wait for the previously queued flip right before queueing
		 * the next one, keeping at most one flip outstanding:
		 */
		if (waiting_for_flip) {
			ret = drm_wait_flip(&drm, &waiting_for_flip, &evctx);
			if (ret) {
				err = ret < 0 ? ret : 0;
				break;
			}
			/* the flip away from the previous scan-out buffer has
			 * completed, release it to render on again: */
			gbm_surface_release_buffer(gbm->surface, bo);
			bo = pending_bo;
			pending_bo = NULL;
		}

		waiting_for_flip = 1;
		ret = drmModePageFlip(drm.fd, drm.crtc_id, fb->fb_id,
		                      DRM_MODE_PAGE_FLIP_EVENT, &waiting_for_flip);
		if (ret) {
			printf("failed to queue page flip: %s\n", strerror(errno));
			waiting_for_flip = 0;
			err = -1;
			break;
		}
		pending_bo = next_bo;

		cur_time = get_time_ns();
		if (cur_time > (report_time + 2 * NSEC_PER_SEC)) {
			double elapsed_time = cur_time - start_time;
			double secs = elapsed_time / (double) NSEC_PER_SEC;
			unsigned frames = i - 1;  /* first frame ignored */
			printf("Rendered %u frames in %f sec (%f fps)\n",
			       frames, secs, (double) frames / secs);
			report_time = cur_time;
		}
	}

	/* drain the pending flip, if any, before returning: */
	pthread_cleanup_pop(1);

	if (pending_bo && !waiting_for_flip) {
		/* the drained flip moved scan-out away from bo: */
		gbm_surface_release_buffer(gbm->surface, bo);
	}

	cur_time = get_time_ns();
	double elapsed_time = cur_time - start_time;
	double secs = elapsed_time / (double) NSEC_PER_SEC;
	unsigned frames = i - 1;  /* first frame ignored */
	printf("Rendered %u frames in %f sec (%f fps)\n",
	       frames, secs, (double) frames / secs);

	return err;
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

	if (options->triple_buffer) {
		if (options->async_page_flip) {
			printf("triple buffering disabled: superseded by async page flips\n");
		} else if (options->surfaceless) {
			printf("triple buffering disabled: not supported in surfaceless mode\n");
		} else {
			drm.run = legacy_run_triple;
		}
	}

	return &drm;
}
