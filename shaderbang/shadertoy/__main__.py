# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

import argparse
import glob
import logging
import re
import os
import signal
import stat
import threading

from array import array
from contextlib import ExitStack
from ctypes import byref
from libevdev import Device, EV_ABS, EV_KEY, EV_REL, INPUT_PROP_DIRECT, INPUT_PROP_POINTER
from pathlib import Path
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread, Thread

from shaderbang import lib as sb, options
from shaderbang.inotify import INotify, IN_CREATE, IN_ATTRIB
from shaderbang.input import Input
from shaderbang.shadertoy.input import AsciiKeyboard, ButtonMouse, Touchscreen, TrackMouse
from shaderbang.shadertoy.input import CubemapTexture, ImageTexture, VolumeTexture
from shaderbang.shadertoy.metadata import Metadata

logging.getLogger('OpenGL.plugins').setLevel(logging.ERROR)

from OpenGL import setPlatform
setPlatform("egl")

from OpenGL.GL import *
from OpenGL.GLU import *


parser = argparse.ArgumentParser(description="Run Shadertoy")
parser.add_argument("shader", metavar="FILE", type=Path, nargs=1,
                    help="the shadertoy file")
parser.add_argument("-D", "--device", metavar="DEVICE", type=Path,
                    help="the DRM device")
parser.add_argument("-C", "--connector", metavar="CONNECTOR", type=int,
                    help="the DRM connector")
parser.add_argument("--mode", metavar="MODE", type=str,
                    help="the name of the video mode, e.g., 1920x1080")
parser.add_argument("--refresh", metavar="FREQ", type=int,
                    help="the vertical refresh rate in Hz")
parser.add_argument("--async-page-flip", action=argparse.BooleanOptionalAction,
                    help="use async page flipping")
parser.add_argument("--atomic-drm-mode", action=argparse.BooleanOptionalAction,
                    help="use atomic mode setting")
parser.add_argument("--triple-buffer", action=argparse.BooleanOptionalAction,
                    help="use triple buffering (vblank-synced page flips, without blocking on them)")
parser.add_argument("-n", "--frames", metavar="N", type=int,
                    help="run for N frames and exit")
parser.add_argument("-k", "--keyboard", metavar="UNIFORM", type=str,
                    help="add a keyboard")
parser.add_argument("--touchscreen", metavar="UNIFORM", type=str,
                    help="add a touchscreen")
parser.add_argument("--trackpad", metavar="UNIFORM", type=str,
                    help="add a trackpad")
parser.add_argument("-c", "--cubemap", metavar=("UNIFORM", "FILE"), type=str, nargs=2,
                    action="append", dest="cubemaps", default=[], help="add a cubemap")
parser.add_argument("-t", "--texture", metavar=("UNIFORM", "FILE"), type=str, nargs=2,
                    action="append", dest="textures", default=[], help="add a texture")
parser.add_argument("-v", "--volume", metavar=("UNIFORM", "FILE"), type=str, nargs=2,
                    action="append", dest="volumes", default=[], help="add a volume")
parser.add_argument("-m", "--metadata", metavar=("<UNIFORM>.KEY", "VALUE"), type=str, nargs=2,
                    action=Metadata, dest="metadata", default={}, help="set uniform metadata")
args = parser.parse_args()


vs_tpl_100 = """
#version {version}

attribute vec3 position;

void main()
{{
    gl_Position = vec4(position, 1.0);
}}
"""

vs_tpl_300 = """
#version {version}

in vec3 position;

void main()
{{
    gl_Position = vec4(position, 1.0);
}}
"""

fs_tpl_body = """
#version {version}

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

out vec4 fragColor;

uniform vec3      iResolution;    // viewport resolution (in pixels)
uniform float     iTime;          // shader playback time (in seconds)
uniform int       iFrame;         // current frame number
uniform vec4      iMouse;         // mouse pixel coords
uniform vec4      iDate;          // (year, month, day, time in seconds)

// Shader body
{body}
"""

fs_tpl_100_main = """
void main()
{{
    mainImage(gl_FragColor, gl_FragCoord.xy);
}}
"""

fs_tpl_300_main = """
void main()
{{
    mainImage(fragColor, gl_FragCoord.xy);
}}
"""

fs_tpl_100 = f"""
{fs_tpl_body}

{fs_tpl_100_main}
"""

fs_tpl_300 = f"""
{fs_tpl_body}

{fs_tpl_300_main}
"""


class Program(Input):

    vertices = [
        # First triangle
         1.0,  1.0,
        -1.0,  1.0,
        -1.0, -1.0,
        # Second triangle
        -1.0, -1.0,
         1.0, -1.0,
         1.0, 1.0,
    ]

    def __init__(self, file: Path):
        super().__init__("program")
        self.file = file
        self.program = None
        self.iFrame = None
        self.iTime = None
        self.iResolution = None
        self.vbo = GLuint()

    def init(self, width, height):
        self.program = glCreateProgram()

        major, minor, version = glsl_version()
        print(f"Using GLSL version directive: {version}")

        vs = glCreateShader(GL_VERTEX_SHADER)
        if major >= 3:
            glShaderSource(vs, vs_tpl_300.format(version=version))
        else:
            glShaderSource(vs, vs_tpl_100.format(version=version))
        glCompileShader(vs)
        if glGetShaderiv(vs, GL_COMPILE_STATUS) == GL_FALSE:
            print("Compile error", glGetShaderInfoLog(vs))
        glAttachShader(self.program, vs)

        fs = glCreateShader(GL_FRAGMENT_SHADER)
        with open(self.file, "r") as shader:
            body = shader.read()
            # Remove the shebang if present
            shebang = re.search(r"^#!.*\n+((.*\n?)+)$", body)
            if shebang:
                body = shebang.group(1)
            if major >= 3:
                glShaderSource(fs, fs_tpl_300.format(version=version, body=body))
            else:
                glShaderSource(fs, fs_tpl_100.format(version=version, body=body))
            glCompileShader(fs)
            if glGetShaderiv(fs, GL_COMPILE_STATUS) == GL_FALSE:
                print("Compile error", glGetShaderInfoLog(fs))
            glAttachShader(self.program, fs)

        glLinkProgram(self.program)
        if glGetProgramiv(self.program, GL_LINK_STATUS) == GL_FALSE:
            print("Link error", glGetProgramInfoLog(self.program))

        glDeleteShader(vs)
        glDeleteShader(fs)

        glViewport(0, 0, width, height)
        glUseProgram(self.program)

        self.iTime = glGetUniformLocation(self.program, "iTime")
        self.iFrame = glGetUniformLocation(self.program, "iFrame")
        self.iResolution = glGetUniformLocation(self.program, "iResolution")
        glUniform3f(self.iResolution, width, height, 0)

        glGenBuffers(1, ctypes.pointer(self.vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        vertices = array("f", self.vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.tobytes(), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self, frame, time):
        glUniform1i(self.iFrame, frame)
        glUniform1f(self.iTime, time)
        # Or elapsed time relative to 60 FPS
        # glUniform1f(self.iTime, frame / 60.0);

        glDrawArrays(GL_TRIANGLES, 0, 6)


def glsl_version() -> tuple[int, int, str]:
    version = str(glGetString(GL_SHADING_LANGUAGE_VERSION))
    if not version:
        print("Cannot detect GLSL version from GL_SHADING_LANGUAGE_VERSION")
        return 1, 0, "100"

    match = re.search(r'(GLSL\s*(ES)?\s*)?(\d+)\.(\d+)', version)
    if not match:
        print(f"Cannot match GLSL version '{version}'")
        return 1, 0, "100"

    es = bool(match.group(2))
    major = int(match.group(3))
    minor = int(match.group(4))

    version = f"{major}{minor if minor != '0' else '00'}{' es' if es else ''}"

    return int(major), int(minor), version


def input_from_device(args, dev: Device):
    if dev.has(EV_REL) and dev.has(EV_KEY.BTN_LEFT):
        # Mouse
        ButtonMouse("iMouse", dev)
    elif dev.has(EV_KEY) and dev.has(EV_KEY.KEY_A):
        # Keyboard
        AsciiKeyboard(args.keyboard if args.keyboard else "iKeyboard", dev)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_DIRECT):
        # Touchscreen
        # Only consider direct input devices, like touchscreens and drawing tablets, see:
        # https://www.kernel.org/doc/Documentation/input/event-codes.txt
        # TODO: keep the touchscreen device that's the same as the display device
        Touchscreen(args.touchscreen if args.touchscreen else "iTouchscreen", dev)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_POINTER):
        # Trackpad
        # https://www.kernel.org/doc/Documentation/input/multi-touch-protocol.txt
        TrackMouse(args.trackpad if args.trackpad else "iMouse", dev)
    else:
        dev.fd.close()


def hot_plug_devices(args, devices: ExitStack, inotify: INotify):
    with devices:
        while True:
            for ev in inotify.read():
                p = os.path.join("/dev/input", ev.name)
                if (str.startswith(ev.name, "event")
                        and os.path.exists(p)
                        and os.access(p, os.R_OK)
                        and stat.S_ISCHR(os.stat(p)[stat.ST_MODE])):
                    input_from_device(args, Device(devices.enter_context(open(p, "rb"))))


def main():
    Program(args.shader[0])

    for (uniform, path) in args.cubemaps:
        CubemapTexture(uniform, path)
    for (uniform, path) in args.textures:
        ImageTexture(uniform, path, **args.metadata[uniform] if uniform in args.metadata else {})
    for (uniform, path) in args.volumes:
        VolumeTexture(uniform, path)

    devices = ExitStack()
    with devices:
        for path in list(filter(lambda p: os.path.exists(p) and stat.S_ISCHR(os.stat(p)[stat.ST_MODE]),
                                glob.glob("{}/event*".format("/dev/input")))):
            input_from_device(args, Device(devices.enter_context(open(path, "rb"))))
        devices = devices.pop_all()

    inotify = INotify()
    inotify.add_watch("/dev/input", IN_CREATE | IN_ATTRIB)
    Thread(target=hot_plug_devices, args=[args, devices, inotify], daemon=True).start()

    ret = sb.init(byref(options(args)))
    if ret != 0:
        devices.close()
        exit(ret)

    ret = sb.run()
    if ret != 0:
        devices.close()
        exit(ret)

    stopped = threading.Event()
    pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCONT])

    def join():
        sb.join()
        stopped.set()
        pthread_kill(main_thread().ident, signal.SIGCONT)

    Thread(target=join, daemon=True).start()

    if sigwait({signal.SIGINT, signal.SIGCONT}) == signal.SIGINT:
        sb.stop()
        stopped.wait(timeout=30)

    inotify.close()
    devices.close()

if __name__ == "__main__":
    main()
