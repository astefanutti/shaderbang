/*
* Copyright © 2024 Antonin Stefanutti <antonin.stefanutti@gmail.com>
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

#include <pthread.h>
#include <stdio.h>

#include "common.h"

static int render_shaderbang(const uint64_t start_time, const unsigned frame) {
	const float time = (float) (get_time_ns() - start_time) / NSEC_PER_SEC;

	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	int ret = callRenderCallbacks(frame, time);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

	return ret;
}

int init_shaderbang(const struct gbm *gbm, struct egl *egl) {
	egl->draw = render_shaderbang;

	return callInitCallbacks(gbm->width, gbm->height);
}
