/*
 * Copyright © 2020 Antonin Stefanutti <antonin.stefanutti@gmail.com>
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

#define _GNU_SOURCE

#include <err.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <regex.h>
#include <stdlib.h>

#include <GLES3/gl3.h>

#include "common.h"

GLint iTime, iFrame;

static const char *shadertoy_vs_tmpl_100 =
		"// version (default: 1.10)              \n"
		"%s                                      \n"
		"                                        \n"
		"attribute vec3 position;                \n"
		"                                        \n"
		"void main()                             \n"
		"{                                       \n"
		"    gl_Position = vec4(position, 1.0);  \n"
		"}                                       \n";

static const char *shadertoy_vs_tmpl_300 =
		"// version                              \n"
		"%s                                      \n"
		"                                        \n"
		"in vec3 position;                       \n"
		"                                        \n"
		"void main()                             \n"
		"{                                       \n"
		"    gl_Position = vec4(position, 1.0);  \n"
		"}                                       \n";

static const char *shadertoy_fs_tmpl_100 =
		"// version (default: 1.10)                                                           \n"
		"%s                                                                                   \n"
		"                                                                                     \n"
		"#ifdef GL_FRAGMENT_PRECISION_HIGH                                                    \n"
		"precision highp float;                                                               \n"
		"#else                                                                                \n"
		"precision mediump float;                                                             \n"
		"#endif                                                                               \n"
		"                                                                                     \n"
		"uniform vec3      iResolution;           // viewport resolution (in pixels)          \n"
		"uniform float     iTime;                 // shader playback time (in seconds)        \n"
		"uniform int       iFrame;                // current frame number                     \n"
		"uniform vec4      iMouse;                // mouse pixel coords                       \n"
		"uniform vec4      iDate;                 // (year, month, day, time in seconds)      \n"
		"                                                                                     \n"
		"// Shader body                                                                       \n"
		"%s                                                                                   \n"
		"                                                                                     \n"
		"void main()                                                                          \n"
		"{                                                                                    \n"
		"    mainImage(gl_FragColor, gl_FragCoord.xy);                                        \n"
		"}                                                                                    \n";

static const char *shadertoy_fs_tmpl_300 =
		"// version                                                                           \n"
		"%s                                                                                   \n"
		"                                                                                     \n"
		"#ifdef GL_FRAGMENT_PRECISION_HIGH                                                    \n"
		"precision highp float;                                                               \n"
		"#else                                                                                \n"
		"precision mediump float;                                                             \n"
		"#endif                                                                               \n"
		"                                                                                     \n"
		"out vec4 fragColor;                                                                  \n"
		"                                                                                     \n"
		"uniform vec3      iResolution;           // viewport resolution (in pixels)          \n"
		"uniform float     iTime;                 // shader playback time (in seconds)        \n"
		"uniform int       iFrame;                // current frame number                     \n"
		"uniform vec4      iMouse;                // mouse pixel coords                       \n"
		"uniform vec4      iDate;                 // (year, month, day, time in seconds)      \n"
		"                                                                                     \n"
		"// Shader body                                                                       \n"
		"%s                                                                                   \n"
		"                                                                                     \n"
		"void main()                                                                          \n"
		"{                                                                                    \n"
		"    mainImage(fragColor, gl_FragCoord.xy);                                           \n"
		"}                                                                                    \n";

static const GLfloat vertices[] = {
		// First triangle:
		1.0f, 1.0f,
		-1.0f, 1.0f,
		-1.0f, -1.0f,
		// Second triangle:
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
};

static const char *load_shader(const char *file) {
	struct stat statbuf;
	int fd, ret;

	fd = open(file, 0);
	if (fd < 0) {
		err(fd, "could not open '%s'", file);
	}

	ret = fstat(fd, &statbuf);
	if (ret < 0) {
		err(ret, "could not stat '%s'", file);
	}

	return mmap(NULL, statbuf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
}

#define GLSL_VERSION_REGEX "(GLSL[[:space:]]*(ES)?[[:space:]]*)?([[:digit:]]+)\\.([[:digit:]]+)"

static char *extract_group(const char *str, regmatch_t group) {
	char *c = calloc(group.rm_eo - group.rm_so, sizeof(char));
	memcpy(c, &str[group.rm_so], group.rm_eo - group.rm_so);
	return c;
}

static char *glsl_version() {
	int ret;
	regex_t regex;
	if ((ret = regcomp(&regex, GLSL_VERSION_REGEX, REG_EXTENDED)) != 0) {
		err(ret, "failed to compile GLSL version regex");
	}

	char *version = "";
	const char *glsl_version = (char *) glGetString(GL_SHADING_LANGUAGE_VERSION);
	if (strlen(glsl_version) == 0) {
		printf("Cannot detect GLSL version from %s\n", "GL_SHADING_LANGUAGE_VERSION");
		return version;
	}

	size_t nGroups = 5;
	regmatch_t groups[nGroups];
	ret = regexec(&regex, glsl_version, nGroups, groups, 0);
	if (ret == REG_NOMATCH) {
		printf("Cannot match GLSL version '%s'\n", glsl_version);
	} else if (ret != 0) {
		err(ret, "failed to match GLSL version '%s'", glsl_version);
	} else {
		char *es = extract_group(glsl_version, groups[2]);
		char *major = extract_group(glsl_version, groups[3]);
		char *minor = extract_group(glsl_version, groups[4]);

		if (strcmp(minor, "0") == 0) {
			free(minor);
			minor = malloc(sizeof(char) * 3);
			strcpy(minor, "00");
		}

		bool is100 = strcmp(major, "1") == 0 && strcmp(minor, "00") == 0;
		bool hasES = strcasecmp(es, "ES") == 0 && !is100;

		asprintf(&version, "%s%s%s", major, minor, hasES ? " es" : "");

		free(es);
		free(major);
		free(minor);
	}
	regfree(&regex);

	return version;
}

static int render_shadertoy(uint64_t start_time, unsigned frame) {
	const float time = (float) (get_time_ns() - start_time) / NSEC_PER_SEC;

	glUniform1f(iTime, time);
	// Replace the above to input elapsed time relative to 60 FPS
	// glUniform1f(iTime, (GLfloat) frame / 60.0f);
	glUniform1ui(iFrame, frame);

	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	int ret = callRenderCallbacks(frame, time);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	if (ret) {
		return -1;
	}

	start_perfcntrs();
	glDrawArrays(GL_TRIANGLES, 0, 6);
	end_perfcntrs();

	return 0;
}

int init_shadertoy(const struct gbm *gbm, struct egl *egl, const char *file) {
	int ret;
	char *shadertoy_vs, *shadertoy_fs;
	GLuint vbo;

	const char *shader = load_shader(file);

	const char *version = glsl_version();
	if (strlen(version) > 0) {
		char *invalid;
		long v = strtol(version, &invalid, 10);
		if (invalid == version) {
			printf("failed to parse detected GLSL version: %s\n", invalid);
			return -1;
		}
		char *version_directive;
		asprintf(&version_directive, "#version %s", version);
		printf("Using GLSL version directive: %s\n", version_directive);

		bool is_glsl_3 = v >= 300;
		asprintf(&shadertoy_vs, is_glsl_3 ? shadertoy_vs_tmpl_300 : shadertoy_vs_tmpl_100, version_directive);
		asprintf(&shadertoy_fs, is_glsl_3 ? shadertoy_fs_tmpl_300 : shadertoy_fs_tmpl_100, version_directive, shader);
	} else {
		asprintf(&shadertoy_vs, shadertoy_vs_tmpl_100, version);
		asprintf(&shadertoy_fs, shadertoy_fs_tmpl_100, version, shader);
	}

	ret = create_program(shadertoy_vs, shadertoy_fs);
	if (ret < 0) {
		printf("failed to create program\n");
		return -1;
	}

	const GLuint program = ret;

	ret = link_program(program);
	if (ret) {
		printf("failed to link program\n");
		return -1;
	}

	glViewport(0, 0, gbm->width, gbm->height);
	glUseProgram(program);

	iTime = glGetUniformLocation(program, "iTime");
	iFrame = glGetUniformLocation(program, "iFrame");
	const GLint iResolution = glGetUniformLocation(program, "iResolution");
	glUniform3f(iResolution, gbm->width, gbm->height, 0);

	ret = callInitCallbacks(gbm->width, gbm->height);
	if (ret) {
		printf("failed to init callbacks\n");
		return -1;
	}

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices[0]);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (const GLvoid *) (intptr_t) 0);
	glEnableVertexAttribArray(0);

	egl->draw = render_shadertoy;

	return 0;
}
