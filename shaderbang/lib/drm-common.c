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
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <unistd.h>

#include "common.h"
#include "drm-common.h"

WEAK union gbm_bo_handle
gbm_bo_get_handle_for_plane(struct gbm_bo *bo, int plane);

WEAK uint64_t
gbm_bo_get_modifier(struct gbm_bo *bo);

WEAK int
gbm_bo_get_plane_count(struct gbm_bo *bo);

WEAK uint32_t
gbm_bo_get_stride_for_plane(struct gbm_bo *bo, int plane);

WEAK uint32_t
gbm_bo_get_offset(struct gbm_bo *bo, int plane);

static void drm_fb_destroy_callback(struct gbm_bo *bo, void *data)
{
	int drm_fd = gbm_device_get_fd(gbm_bo_get_device(bo));
	struct drm_fb *fb = data;

	if (fb->fb_id)
		drmModeRmFB(drm_fd, fb->fb_id);

	free(fb);
}

struct drm_fb * drm_fb_get_from_bo(struct gbm_bo *bo)
{
	int drm_fd = gbm_device_get_fd(gbm_bo_get_device(bo));
	struct drm_fb *fb = gbm_bo_get_user_data(bo);
	uint32_t width, height, format,
			strides[4] = {0}, handles[4] = {0},
			offsets[4] = {0}, flags = 0;
	int ret = -1;

	if (fb)
		return fb;

	fb = calloc(1, sizeof *fb);
	fb->bo = bo;

	width = gbm_bo_get_width(bo);
	height = gbm_bo_get_height(bo);
	format = gbm_bo_get_format(bo);

	if (gbm_bo_get_handle_for_plane && gbm_bo_get_modifier &&
	    gbm_bo_get_plane_count && gbm_bo_get_stride_for_plane &&
	    gbm_bo_get_offset) {

		uint64_t modifiers[4] = {0};
		modifiers[0] = gbm_bo_get_modifier(bo);
		const int num_planes = gbm_bo_get_plane_count(bo);
		for (int i = 0; i < num_planes; i++) {
			handles[i] = gbm_bo_get_handle_for_plane(bo, i).u32;
			strides[i] = gbm_bo_get_stride_for_plane(bo, i);
			offsets[i] = gbm_bo_get_offset(bo, i);
			modifiers[i] = modifiers[0];
		}

		if (modifiers[0] && modifiers[0] != DRM_FORMAT_MOD_INVALID) {
			flags = DRM_MODE_FB_MODIFIERS;
		}

		ret = drmModeAddFB2WithModifiers(drm_fd, width, height,
		                                 format, handles, strides, offsets,
		                                 modifiers, &fb->fb_id, flags);
	}

	if (ret) {
		if (flags)
			fprintf(stderr, "Modifiers failed!\n");

		memcpy(handles, (uint32_t[4]) {gbm_bo_get_handle(bo).u32, 0, 0, 0}, 16);
		memcpy(strides, (uint32_t[4]) {gbm_bo_get_stride(bo), 0, 0, 0}, 16);
		memset(offsets, 0, 16);
		ret = drmModeAddFB2(drm_fd, width, height, format,
		                    handles, strides, offsets, &fb->fb_id, 0);
	}

	if (ret) {
		printf("failed to create fb: %s\n", strerror(errno));
		free(fb);
		return NULL;
	}

	gbm_bo_set_user_data(bo, fb, drm_fb_destroy_callback);

	return fb;
}

int drm_wait_flip(const struct drm *drm, int *waiting_for_flip, drmEventContext *evctx)
{
	fd_set fds;
	int ret;

	while (*waiting_for_flip) {
		FD_ZERO(&fds);
		FD_SET(0, &fds);
		FD_SET(drm->fd, &fds);

		ret = select(drm->fd + 1, &fds, NULL, NULL, NULL);
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
			return 1;
		}
		drmHandleEvent(drm->fd, evctx);
	}

	return 0;
}

void drm_drain_flip(void *data)
{
	struct flip_drain *drain = data;
	int oldstate;

	/* This may run as a pthread cleanup handler upon cancellation;
	 * make sure the cancellation points below do not re-trigger. */
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldstate);

	while (*drain->waiting_for_flip) {
		fd_set fds;
		/* a queued page flip completes within a refresh interval,
		 * so a bounded wait is enough (and cannot hang shutdown): */
		struct timeval timeout = {.tv_sec = 1, .tv_usec = 0};

		FD_ZERO(&fds);
		FD_SET(drain->drm->fd, &fds);

		if (select(drain->drm->fd + 1, &fds, NULL, NULL, &timeout) <= 0)
			break;

		drmHandleEvent(drain->drm->fd, drain->evctx);
	}

	pthread_setcancelstate(oldstate, NULL);
}

static int32_t find_crtc_for_encoder(const drmModeRes *resources,
                                     const drmModeEncoder *encoder)
{
	int i;

	for (i = 0; i < resources->count_crtcs; i++) {
		/* possible_crtcs is a bitmask as described here:
		 * https://dvdhrm.wordpress.com/2012/09/13/linux-drm-mode-setting-api
		 */
		const uint32_t crtc_mask = 1 << i;
		const uint32_t crtc_id = resources->crtcs[i];
		if (encoder->possible_crtcs & crtc_mask) {
			return crtc_id;
		}
	}

	/* no match found */
	return -1;
}

static int32_t find_crtc_for_connector(const struct drm *drm,
                                       const drmModeRes *resources,
                                       const drmModeConnector *connector)
{
	int i;

	for (i = 0; i < connector->count_encoders; i++) {
		const uint32_t encoder_id = connector->encoders[i];
		drmModeEncoder *encoder = drmModeGetEncoder(drm->fd, encoder_id);

		if (encoder) {
			const int32_t crtc_id = find_crtc_for_encoder(resources, encoder);

			drmModeFreeEncoder(encoder);
			if (crtc_id != 0) {
				return crtc_id;
			}
		}
	}

	/* no match found */
	return -1;
}

static drmModeConnector *find_drm_connector(int fd, drmModeRes *resources,
                                            int connector_id)
{
	drmModeConnector *connector = NULL;
	int i;

	if (connector_id >= 0) {
		if (connector_id >= resources->count_connectors)
			return NULL;

		connector = drmModeGetConnector(fd, resources->connectors[connector_id]);
		if (connector && connector->connection == DRM_MODE_CONNECTED)
			return connector;

		drmModeFreeConnector(connector);
		return NULL;
	}

	for (i = 0; i < resources->count_connectors; i++) {
		connector = drmModeGetConnector(fd, resources->connectors[i]);
		if (connector && connector->connection == DRM_MODE_CONNECTED) {
			/* it's connected, let's use this! */
			break;
		}
		drmModeFreeConnector(connector);
		connector = NULL;
	}

	return connector;
}

static int get_resources(int fd, drmModeRes **resources)
{
	*resources = drmModeGetResources(fd);
	if (*resources == NULL)
		return -1;
	return 0;
}

#define MAX_DRM_DEVICES 64

int find_drm_device()
{
	drmDevicePtr devices[MAX_DRM_DEVICES] = {NULL};
	int num_devices, fd = -1;

	num_devices = drmGetDevices2(0, devices, MAX_DRM_DEVICES);
	if (num_devices < 0) {
		printf("drmGetDevices2 failed: %s\n", strerror(-num_devices));
		return -1;
	}

	for (int i = 0; i < num_devices; i++) {
		drmModeRes *resources;
		drmDevicePtr device = devices[i];
		int ret;

		if (!(device->available_nodes & (1 << DRM_NODE_PRIMARY)))
			continue;
		/* OK, it's a primary device. If we can get the
		 * drmModeResources, it means it's also a
		 * KMS-capable device.
		 */
		fd = open(device->nodes[DRM_NODE_PRIMARY], O_RDWR | O_NONBLOCK | O_CLOEXEC);
		if (fd < 0)
			continue;
		ret = get_resources(fd, &resources);
		drmModeFreeResources(resources);
		if (!ret)
			break;
		close(fd);
		fd = -1;
	}
	drmFreeDevices(devices, num_devices);

	if (fd < 0)
		printf("no DRM device found!\n");
	return fd;
}

int find_plane_prop(const struct drm *drm, const char *name, unsigned int *prop_idx)
{
	struct plane *obj = drm->plane;
	unsigned int i;

	for (i = 0; i < obj->props->count_props; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			break;
		}
	}

	if (i == obj->props->count_props) {
		printf("no plane property: %s\n", name);
		return -EINVAL;
	}

	*prop_idx = i;

	return 0;
}

static int get_plane_property_val(const struct drm *drm, const char *name, uint64_t *val)
{
	struct plane *obj = drm->plane;
	unsigned int prop_idx;
	int res = find_plane_prop(drm, name, &prop_idx);

	if (res) return res;

	*val = obj->props->prop_values[prop_idx];

	return 0;
}

static int get_plane_format_modifiers(const struct drm *drm)
{
	struct plane *plane = drm->plane;
	drmModePropertyBlobPtr blob;
	struct drm_format_modifier_blob *format_blob;
	struct drm_format_modifier *modifiers;
	uint32_t *formats;
	uint64_t blob_id;
	int res;
	uint32_t i;

	if ((res = get_plane_property_val(drm, "IN_FORMATS", &blob_id))) return res;

	blob = drmModeGetPropertyBlob(drm->fd, blob_id);

	if (!blob) {
		return -ENOMEM;
	}

	format_blob = blob->data;
	plane->formats = calloc(format_blob->count_formats,
	                        sizeof(plane->formats[0]));
	plane->modifiers = calloc(format_blob->count_modifiers,
	                          sizeof(plane->modifiers[0]));

	if (!plane->formats || !plane->modifiers) {
		free(plane->formats);
		free(plane->modifiers);
		return -ENOMEM;
	}

	formats = (uint32_t *) ((char *) format_blob +
	                        format_blob->formats_offset);
	modifiers = (struct drm_format_modifier *)
			((char *) format_blob + format_blob->modifiers_offset);

	memcpy(plane->formats, formats,
	       sizeof(plane->formats[0]) * format_blob->count_formats);

	for (i = 0; i < format_blob->count_modifiers; i++) {
		/*
		 * XXX Not quite right.  Should also stash which formats each
		 * modifier supports next to it in a per-modifier array, or
		 * vice-versa.
		 */
		plane->modifiers[i] = modifiers[i].modifier;
	}

	plane->num_formats = format_blob->count_formats;
	plane->num_modifiers = format_blob->count_modifiers;

	return 0;
}

/* Pick a plane, something that at a minimum can be connected to
 * the chosen CRTC, but prefer primary plane.
 *
 * Seems like there is some room for a drmModeObjectGetNamedProperty()
 * type helper in libdrm.
 */
static int get_plane_id(const struct drm *drm)
{
	drmModePlaneResPtr plane_resources;
	uint32_t i, j;
	int ret = -EINVAL;
	int found_primary = 0;

	plane_resources = drmModeGetPlaneResources(drm->fd);
	if (!plane_resources) {
		printf("drmModeGetPlaneResources failed: %s\n", strerror(errno));
		return -1;
	}

	for (i = 0; (i < plane_resources->count_planes) && !found_primary; i++) {
		uint32_t id = plane_resources->planes[i];
		drmModePlanePtr plane = drmModeGetPlane(drm->fd, id);
		if (!plane) {
			printf("drmModeGetPlane(%u) failed: %s\n", id, strerror(errno));
			continue;
		}

		if (plane->possible_crtcs & (1 << drm->crtc_index)) {
			drmModeObjectPropertiesPtr props =
					drmModeObjectGetProperties(drm->fd, id, DRM_MODE_OBJECT_PLANE);

			/* primary or not, this plane is good enough to use: */
			ret = id;

			for (j = 0; j < props->count_props; j++) {
				drmModePropertyPtr p =
						drmModeGetProperty(drm->fd, props->props[j]);

				if ((strcmp(p->name, "type") == 0) &&
				    (props->prop_values[j] == DRM_PLANE_TYPE_PRIMARY)) {
					/* found our primary plane, lets use that: */
					found_primary = 1;
				}

				drmModeFreeProperty(p);
			}

			drmModeFreeObjectProperties(props);
		}

		drmModeFreePlane(plane);
	}

	drmModeFreePlaneResources(plane_resources);

	return ret;
}

const uint64_t *get_drm_format_modifiers(const struct drm *drm,
                                         unsigned int *count)
{
	if (drm->plane) {
		*count = drm->plane->num_modifiers;
		return drm->plane->modifiers;
	}

	*count = 0;
	return NULL;
}

int init_drm(struct drm *drm, const int fd, const struct options *options)
{
	drmModeRes *resources;
	drmModeConnector *connector = NULL;
	drmModeEncoder *encoder = NULL;
	int i, area;

	drm->fd = fd;
	drm->async_page_flip = options->async_page_flip;
	drm->frames = options->frames;

	get_resources(drm->fd, &resources);
	if (!resources) {
		printf("drmModeGetResources failed: %s\n", strerror(errno));
		return -1;
	}

	/* find a connected connector: */
	connector = find_drm_connector(drm->fd, resources, options->connector);
	if (!connector) {
		/* we could be fancy and listen for hot-plug events and wait for
		 * a connector.
		 */
		printf("no connected connector!\n");
		return -1;
	}

	/* find requested mode: */
	if (*options->mode || options->refresh) {
		for (i = 0; i < connector->count_modes; i++) {
			drmModeModeInfo *current_mode = &connector->modes[i];

			if (*options->mode) {
				if (strcmp(current_mode->name, options->mode) == 0) {
					if (options->refresh == 0 || current_mode->vrefresh == options->refresh) {
						drm->mode = current_mode;
						break;
					}
				}
			} else if (current_mode->vrefresh == options->refresh) {
				drm->mode = current_mode;
				break;
			}
		}

		if (!drm->mode)
			printf("requested mode not found, using default mode!\n");
	}

	/* find preferred mode or the highest resolution mode: */
	if (!drm->mode) {
		for (i = 0, area = 0; i < connector->count_modes; i++) {
			drmModeModeInfo *current_mode = &connector->modes[i];

			if (current_mode->type & DRM_MODE_TYPE_PREFERRED) {
				drm->mode = current_mode;
				break;
			}

			int current_area = current_mode->hdisplay * current_mode->vdisplay;
			if (current_area > area) {
				drm->mode = current_mode;
				area = current_area;
			}
		}
	}

	if (!drm->mode) {
		printf("could not find mode!\n");
		return -1;
	}

	/* find encoder: */
	for (i = 0; i < resources->count_encoders; i++) {
		encoder = drmModeGetEncoder(drm->fd, resources->encoders[i]);
		if (encoder->encoder_id == connector->encoder_id)
			break;
		drmModeFreeEncoder(encoder);
		encoder = NULL;
	}

	if (encoder) {
		drm->crtc_id = encoder->crtc_id;
	} else {
		int32_t crtc_id = find_crtc_for_connector(drm, resources, connector);
		if (crtc_id == -1) {
			printf("No CRTC found!\n");
			return -1;
		}

		drm->crtc_id = crtc_id;
	}

	for (i = 0; i < resources->count_crtcs; i++) {
		if (resources->crtcs[i] == drm->crtc_id) {
			drm->crtc_index = i;
			break;
		}
	}

	drmModeFreeResources(resources);

	drm->connector_id = connector->connector_id;

	int plane_id = get_plane_id(drm);
	if (!plane_id) {
		printf("could not find a suitable plane\n");
		return -1;
	}

	/* We only do single plane to single CRTC to single connector, no
	 * fancy multi-monitor or multi-plane stuff. So just grab the
	 * plane/crtc/connector property info for one of each:
	 */
	drm->plane = calloc(1, sizeof(*drm->plane));
	drm->crtc = calloc(1, sizeof(*drm->crtc));
	drm->connector = calloc(1, sizeof(*drm->connector));

#define get_resource(type, Type, id) do {                    \
            drm->type->type = drmModeGet##Type(drm->fd, id); \
            if (!drm->type->type) {                          \
                printf("could not get %s %i: %s\n",          \
                        #type, id, strerror(errno));         \
                return -1;                                   \
            }                                                \
        } while (0)

	get_resource(plane, Plane, plane_id);
	get_resource(crtc, Crtc, drm->crtc_id);
	get_resource(connector, Connector, drm->connector_id);

#define get_properties(type, TYPE, id) do {                           \
        uint32_t i;                                                   \
        drm->type->props = drmModeObjectGetProperties(drm->fd,        \
                id, DRM_MODE_OBJECT_##TYPE);                          \
        if (!drm->type->props) {                                      \
            printf("could not get %s %u properties: %s\n",            \
                    #type, id, strerror(errno));                      \
            return -1;                                                \
        }                                                             \
        drm->type->props_info = calloc(drm->type->props->count_props, \
                sizeof(*drm->type->props_info));                      \
        for (i = 0; i < drm->type->props->count_props; i++) {         \
            drm->type->props_info[i] = drmModeGetProperty(drm->fd,    \
                    drm->type->props->props[i]);                      \
        }                                                             \
    } while (0)

	get_properties(plane, PLANE, plane_id);
	get_properties(crtc, CRTC, drm->crtc_id);
	get_properties(connector, CONNECTOR, drm->connector_id);

	get_plane_format_modifiers(drm);

	return 0;
}
