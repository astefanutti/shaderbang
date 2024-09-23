#!/usr/bin/env python

import argparse
import glob
import re
import os
import signal
import stat
import threading

from contextlib import ExitStack
from ctypes import byref
from shaderbang.inotify import INotify, IN_CREATE, IN_ATTRIB
from shaderbang.shadertoy import AsciiKeyboard, ButtonMouse, Touchscreen, TrackMouse
from shaderbang.shadertoy import CubemapTexture, ImageTexture, VolumeTexture
from lib import glsl, options
from libevdev import *
from pathlib import Path
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread, Thread


class Metadata(argparse.Action):

    def __call__(self, _, namespace, values, option=None):
        m = re.search(r"^(\w+)\.(\w+)$", values[0])
        if not m:
            raise ValueError(f"value {values[0]} for option {option} must match <UNIFORM>.KEY")
        metadata = getattr(namespace, self.dest)
        if not m.group(1) in metadata:
            metadata[m.group(1)] = {}
        metadata[m.group(1)] = {**metadata[m.group(1)], **{m.group(2): values[1]}}


def input_from_device(dev: Device):
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


def hot_plug_devices():
    with devices:
        while True:
            for ev in inotify.read():
                p = os.path.join("/dev/input", ev.name)
                if (str.startswith(ev.name, "event")
                        and os.path.exists(p)
                        and os.access(p, os.R_OK)
                        and stat.S_ISCHR(os.stat(p)[stat.ST_MODE])):
                    input_from_device(Device(devices.enter_context(open(p, "rb"))))


parser = argparse.ArgumentParser(description="Run OpenGL shaders using DRM/KMS")
parser.add_argument("shader", metavar="FILE", type=Path, nargs=1,
                    help="the shader file")
parser.add_argument("--async-page-flip", action=argparse.BooleanOptionalAction,
                    help="use async page flipping")
parser.add_argument("--atomic-drm-mode", action=argparse.BooleanOptionalAction,
                    help="use atomic mode setting and fencing")
parser.add_argument("-C", "--connector", metavar="CONNECTOR", type=int,
                    help="the DRM connector")
parser.add_argument("-D", "--device", metavar="DEVICE", type=Path,
                    help="the DRM device")
parser.add_argument("--mode", metavar="MODE", type=str,
                    help="specify the video mode in the format <resolution>[-<vrefresh>]")
parser.add_argument("-n", "--frames", metavar="N", type=int,
                    help="run for the given number of frames and exit")
parser.add_argument("-k", "--keyboard", metavar="UNIFORM", type=str,
                    help="add keyboard")
parser.add_argument("--touchscreen", metavar="UNIFORM", type=str,
                    help="add touchscreen device")
parser.add_argument("--trackpad", metavar="UNIFORM", type=str,
                    help="add trackpad device")
parser.add_argument("-c", "--cubemap", metavar=("UNIFORM", "FILE"), type=str, nargs=2,
                    action="append", dest="cubemaps", default=[], help="add cubemap")
parser.add_argument("-t", "--texture", metavar=("UNIFORM", "FILE"), type=str, nargs=2,
                    action="append", dest="textures", default=[], help="add texture")
parser.add_argument("-v", "--volume", metavar=("UNIFORM", "FILE"), type=str, nargs=2,
                    action="append", dest="volumes", default=[], help="add volume")
parser.add_argument("-m", "--metadata", metavar=("<UNIFORM>.KEY", "VALUE"), type=str, nargs=2,
                    action=Metadata, dest="metadata", default={}, help="set uniform metadata")
args = parser.parse_args()

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
        input_from_device(Device(devices.enter_context(open(path, "rb"))))
    devices = devices.pop_all()

inotify = INotify()
inotify.add_watch("/dev/input", IN_CREATE | IN_ATTRIB)
Thread(target=hot_plug_devices, daemon=True).start()

ret = glsl.init(bytes(args.shader[0].as_posix(), "utf-8"), byref(options(args)))
if ret != 0:
    devices.close()
    exit(ret)

ret = glsl.run()
if ret != 0:
    devices.close()
    exit(ret)

stopped = threading.Event()
pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCONT])


def join():
    glsl.join()
    stopped.set()
    pthread_kill(main_thread().ident, signal.SIGCONT)


Thread(target=join, daemon=True).start()

if sigwait({signal.SIGINT, signal.SIGCONT}) == signal.SIGINT:
    glsl.stop()
    stopped.wait(timeout=30)

inotify.close()
devices.close()
