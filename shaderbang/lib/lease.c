/*
 * Copyright (c) 2023 Antonin Stefanutti <antonin.stefanutti@gmail.com>
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

#include "lease.h"

#if XCB_LEASE

int xcb_lease(xcb_connection_t *connection, int *screen) {
	xcb_screen_iterator_t s_i;
	int i_s = 0;
	for (s_i = xcb_setup_roots_iterator(xcb_get_setup(connection)); s_i.rem; xcb_screen_next(&s_i), i_s++) {
		if (i_s == *screen)
			break;
	}

	xcb_window_t root = s_i.data->root;

	xcb_randr_get_screen_resources_cookie_t gsr_c = xcb_randr_get_screen_resources(connection, root);
	xcb_randr_get_screen_resources_reply_t *gsr_r = xcb_randr_get_screen_resources_reply(connection, gsr_c, NULL);
	if (!gsr_r) {
		printf("xcb_randr_get_screen_resources failed\n");
		return -1;
	}

	xcb_randr_output_t *ro = xcb_randr_get_screen_resources_outputs(gsr_r);
	int o, c;

	xcb_randr_output_t output = 0;

	/* Use the first connected output */
	for (o = 0; output == 0 && o < gsr_r->num_outputs; o++) {
		xcb_randr_get_output_info_cookie_t goi_c = xcb_randr_get_output_info(connection, ro[o], gsr_r->config_timestamp);
		xcb_randr_get_output_info_reply_t *goi_r = xcb_randr_get_output_info_reply(connection, goi_c, NULL);
		if (!goi_r) {
			printf("xcb_randr_get_output_info failed\n");
			return -1;
		}

		if (goi_r->connection == XCB_RANDR_CONNECTION_CONNECTED && goi_r->crtc != 0) {
			output = ro[o];
		}

		free(goi_r);
	}

	xcb_randr_crtc_t *rc = xcb_randr_get_screen_resources_crtcs(gsr_r);
	xcb_randr_crtc_t crtc = 0;

	/* Use the first connected crtc */
	for (c = 0; crtc == 0 && c < gsr_r->num_crtcs; c++) {
		xcb_randr_get_crtc_info_cookie_t gci_c = xcb_randr_get_crtc_info(connection, rc[c], gsr_r->config_timestamp);
		xcb_randr_get_crtc_info_reply_t *gci_r = xcb_randr_get_crtc_info_reply(connection, gci_c, NULL);
		if (!gci_r) {
			printf("xcb_randr_get_crtc_info failed\n");
			return -1;
		}

//		if (gci_r->mode != 0) {
		crtc = rc[c];
//		}

		free(gci_r);
	}

	free(gsr_r);

	xcb_randr_lease_t lease = xcb_generate_id(connection);

	xcb_randr_create_lease_cookie_t rcl_c = xcb_randr_create_lease(connection,root,lease,1,1,&crtc,&output);
	xcb_randr_create_lease_reply_t *rcl_r = xcb_randr_create_lease_reply(connection, rcl_c, NULL);
	if (!rcl_r) {
		printf("xcb_randr_create_lease failed\n");
		return -1;
	}

	int *rcl_f = xcb_randr_create_lease_reply_fds(connection, rcl_r);
	int fd = rcl_f[0];
	free(rcl_r);

	return fd;
}

#endif
