from ctypes import *
from pathlib import Path
from typing import Iterator

import signal
import threading

from libevdev import EV_ABS, EV_KEY, EV_REL
from PIL import Image
from lib import glsl

from shaderbang.gl import *
from shaderbang.input import Input, InputDevice, keycodes, Mouse, FallbackInputDevice

_texture_units: Iterator[int] | None = None


def _init_slots():
    global _texture_units
    max_texture_image_units = pointer(c_uint())
    glsl.glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, max_texture_image_units)
    value = max_texture_image_units.contents.value
    _texture_units = iter([i for i in range(value if value > 0 else 16)])


class NoActiveUniformVariable(Exception):
    name = None

    def __init__(self, name):
        super().__init__(f"no active uniform variable '{name}'")
        self.name = name


class Uniform(Input):
    loc = None

    def init(self, **kwargs):
        super().init(**kwargs)

        program = c_uint()
        glsl.glGetIntegerv(GL_CURRENT_PROGRAM, pointer(program))

        self.loc = glsl.glGetUniformLocation(program, bytes(self.name, 'utf-8'))
        if self.loc < 0:
            raise NoActiveUniformVariable(self.name)


class Texture(Uniform):
    tex = None
    unit = None

    def init(self, **kwargs):
        super().init(**kwargs)

        if not _texture_units:
            _init_slots()

        self.tex = c_uint()
        self.unit = next(_texture_units)
        glsl.glUniform1i(self.loc, self.unit)
        glsl.glActiveTexture(GL_TEXTURE0 + self.unit)
        glsl.glGenTextures(1, pointer(self.tex))


class ImageTexture(Texture):
    path = ''
    transpose = None

    def __init__(self, name, path, transpose=None):
        super().__init__(name)
        self.path = path
        self.transpose = transpose

    def init(self, **kwargs):
        super().init(**kwargs)

        glsl.glBindTexture(GL_TEXTURE_2D, self.tex)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glsl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = Image.open(self.path)
        if self.transpose:
            image = image.transpose(getattr(Image, self.transpose))
        data = image.convert('RGBA').tobytes()
        glsl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glsl.glGenerateMipmap(GL_TEXTURE_2D)
        image.close()


class VolumeTexture(Texture):
    path = ''

    def __init__(self, name, path):
        super().__init__(name)
        self.path = path

    def init(self, **kwargs):
        super().init(**kwargs)

        data = Path(self.path).read_bytes()
        width = int.from_bytes(data[4:8], byteorder='little')
        height = int.from_bytes(data[8:12], byteorder='little')
        depth = int.from_bytes(data[12:16], byteorder='little')
        channels = int.from_bytes(data[16:17], byteorder='little')
        bin_format = int.from_bytes(data[18:20], byteorder='little')
        is_float = bin_format == 10

        internal_format = GL_RGBA
        source_format = GL_RGBA

        if channels == 4:
            internal_format = GL_RGBA16F if is_float else GL_RGBA8
            source_format = GL_RGBA
        elif channels == 3:
            internal_format = GL_RGB16F if is_float else GL_RGB8
            source_format = GL_RGB
        elif channels == 2:
            internal_format = GL_RG16F if is_float else GL_RG8
            source_format = GL_RG
        elif channels == 1:
            internal_format = GL_R16F if is_float else GL_R8
            source_format = GL_RED

        glsl.glBindTexture(GL_TEXTURE_3D, self.tex)
        glsl.glTexImage3D(GL_TEXTURE_3D, 0, internal_format, width, height, depth, 0, source_format,
                          GL_FLOAT if is_float else GL_UNSIGNED_BYTE, data[20:])
        glsl.glGenerateMipmap(GL_TEXTURE_3D)


class CubemapTexture(Texture):
    path = ''

    def __init__(self, name, path):
        super().__init__(name)
        self.path = path

    def init(self, **kwargs):
        super().init(**kwargs)

        glsl.glBindTexture(GL_TEXTURE_CUBE_MAP, self.tex)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glsl.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = Image.open(self.path)
        data = image.convert('RGB').tobytes()
        channels = len(image.getbands())
        for i in range(0, 6):
            glsl.glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB8, image.width, image.width, 0, GL_RGB,
                              GL_UNSIGNED_BYTE, data[i*image.width**2*channels:])
        glsl.glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        image.close()


class Keyboard(InputDevice, Texture):
    buffer = [0] * 256 * 3

    def event(self, ev, **_):
        if not ev.matches(EV_KEY):
            return
        code = -1
        if ev.code in keycodes:
            code = keycodes[ev.code]
        if code < 0:
            return

        if ev.value == 0:
            self.buffer[code] = 0
            self.buffer[256 + code] = 0
        elif ev.value == 1:
            self.buffer[code] = 255
            self.buffer[256 + code] = 255
            self.buffer[256 * 2 + code] = 255 - self.buffer[256 * 2 + code]

            # The keyboard device has been grabbed, so events are not sent to virtual
            # devices. The main thread has to be interrupted on CTRL+C explicitly.
            if ev.matches(EV_KEY.KEY_C) and self.buffer[17]:
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)

    def render(self, frame, **_):
        glsl.glActiveTexture(GL_TEXTURE0 + self.unit)
        glsl.glBindTexture(GL_TEXTURE_2D, self.tex)
        glsl.glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 256, 3, 0, GL_RED, GL_UNSIGNED_BYTE,
                          (c_ubyte * (256 * 3))(*self.buffer))
        glsl.glGenerateMipmap(GL_TEXTURE_2D)
        for i in range(256, 2 * 256):
            self.buffer[i] = 0


class ButtonMouse(Mouse, Uniform):
    click = False
    pointer_xy = drag_start = drag_xy = (1, 1)

    def event(self, ev, **_):
        if ev.matches(EV_KEY):
            if not ev.code == EV_KEY.BTN_LEFT:
                return
            if ev.value == 1:
                self.click = True
                self.drag_start = self.drag_xy = self.pointer_xy
                self.drag = True
            elif ev.value == 0:
                self.drag = False
        elif ev.matches(EV_REL):
            if ev.code == EV_REL.REL_X:
                self.pointer_xy = (max(1, min(self.pointer_xy[0] + ev.value, self.resolution[0])), self.pointer_xy[1])
            elif ev.code == EV_REL.REL_Y:
                self.pointer_xy = (self.pointer_xy[0], max(1, min(self.pointer_xy[1] - ev.value, self.resolution[1])))
            if self.drag:
                self.drag_xy = self.pointer_xy

    def render(self, **_):
        (z, w) = self.drag_start
        if self.drag:
            if not self.click:
                (z, w) = (z, -w)
        else:
            (z, w) = (-z, -w)

        glsl.glUniform4f(self.loc, c_float(self.drag_xy[0]), c_float(self.drag_xy[1]), c_float(z), c_float(w))

        if self.click:
            self.click = False


class _MTSlot:
    touch = False
    drag = (False, False)
    drag_xy = drag_start = (1, 1)


class Touchscreen(InputDevice, Uniform):
    dev_abs_max = None
    resolution: (int, int) = None
    slots = []
    u4fv = None
    dirty = False

    def init(self, width, height, **kwargs):
        try:
            super().init(width=width, height=height, **kwargs)
        except NoActiveUniformVariable:
            # Fall back to using the touchscreen as a mouse device
            raise FallbackInputDevice(lambda: TouchMouse('iMouse', self.dev))

        self.dev_abs_max = (self.dev.absinfo[EV_ABS.ABS_X].maximum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)
        self.resolution = (width, height)
        self.slots = [_MTSlot() for _ in range(self.dev.num_slots)]
        self.u4fv = [0.0] * self.dev.num_slots * 4

    def event(self, ev, **_):
        self.dirty = True
        slot = self.slots[self.dev.current_slot]
        if ev.code == EV_ABS.ABS_MT_TRACKING_ID:
            if ev.value >= 0:
                slot.touch = True
            else:
                slot.drag = (False, False)
        elif ev.code == EV_ABS.ABS_MT_POSITION_X:
            slot.drag_xy = (ev.value / self.dev_abs_max[0] * self.resolution[0], slot.drag_xy[1])
            if not slot.drag[0]:
                slot.drag_start = (slot.drag_xy[0], slot.drag_start[1])
                slot.drag = (True, slot.drag[1])
        elif ev.code == EV_ABS.ABS_MT_POSITION_Y:
            slot.drag_xy = (slot.drag_xy[0], self.resolution[1] - ev.value / self.dev_abs_max[1] * self.resolution[1])
            if not slot.drag[1]:
                slot.drag_start = (slot.drag_start[0], slot.drag_xy[1])
                slot.drag = (slot.drag[0], True)

    def render(self, **_):
        if not self.dirty:
            return

        dirty = self.dirty
        self.dirty = False

        for i, slot in enumerate(self.slots):
            self.u4fv[4 * i] = slot.drag_xy[0]
            self.u4fv[4 * i + 1] = slot.drag_xy[1]
            (z, w) = slot.drag_start
            if slot.drag[0] and slot.drag[1]:
                if not slot.touch:
                    (z, w) = (z, -w)
            else:
                (z, w) = (-z, -w)
            self.u4fv[4 * i + 2] = z
            self.u4fv[4 * i + 3] = w
            if slot.touch:
                slot.touch = False
                self.dirty = True

        if dirty:
            glsl.glUniform4fv(self.loc, len(self.u4fv), (c_float * len(self.u4fv))(*self.u4fv))


class TouchMouse(Mouse, Uniform):
    dev_abs_max = None
    _drag = (False, False)
    drag_xy = drag_start = (1, 1)
    touch = False

    def init(self, **kwargs):
        super().init(**kwargs)

        self.dev_abs_max = (self.dev.absinfo[EV_ABS.ABS_X].maximum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)

    def event(self, ev, **_):
        if ev.code == EV_KEY.BTN_TOUCH:
            if ev.value == 1:
                self.drag = True
                self.touch = True
            else:
                self.drag = False
                self._drag = (False, False)
        elif ev.code == EV_ABS.ABS_X:
            self.drag_xy = (ev.value / self.dev_abs_max[0] * self.resolution[0], self.drag_xy[1])
            if not self._drag[0]:
                self.drag_start = (self.drag_xy[0], self.drag_start[1])
                self._drag = (True, self._drag[1])
        elif ev.code == EV_ABS.ABS_Y:
            self.drag_xy = (self.drag_xy[0], self.resolution[1] - ev.value / self.dev_abs_max[1] * self.resolution[1])
            if not self._drag[1]:
                self.drag_start = (self.drag_start[0], self.drag_xy[1])
                self._drag = (self._drag[0], True)

    def render(self, **_):
        (z, w) = self.drag_start
        if self.drag:
            if not self.touch:
                (z, w) = (z, -w)
        else:
            (z, w) = (-z, -w)

        glsl.glUniform4f(self.loc, c_float(self.drag_xy[0]), c_float(self.drag_xy[1]), c_float(z), c_float(w))

        if self.touch:
            self.touch = False


class Trackpad(InputDevice, Uniform):
    dev_abs_x: (int, int) = None
    dev_abs_y: (int, int) = None
    resolution: (int, int) = None
    slots = []
    u4fv = None
    dirty = False

    def init(self, width, height, **kwargs):
        try:
            super().init(width=width, height=height, **kwargs)
        except NoActiveUniformVariable:
            # Fall back to using the trackpad as a mouse device
            raise FallbackInputDevice(lambda: TrackMouse('iMouse', self.dev))

        self.dev_abs_x = (self.dev.absinfo[EV_ABS.ABS_X].minimum, self.dev.absinfo[EV_ABS.ABS_X].maximum)
        self.dev_abs_y = (self.dev.absinfo[EV_ABS.ABS_Y].minimum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)
        self.resolution = (width, height)
        self.slots = [_MTSlot() for _ in range(self.dev.num_slots)]
        self.u4fv = [0.0] * self.dev.num_slots * 4

    def event(self, ev, **_):
        self.dirty = True
        slot = self.slots[self.dev.current_slot]
        if ev.code == EV_ABS.ABS_MT_TRACKING_ID:
            if ev.value >= 0:
                slot.touch = True
            else:
                slot.drag = (False, False)
        elif ev.code == EV_ABS.ABS_MT_POSITION_X:
            slot.drag_xy = ((ev.value - self.dev_abs_x[0]) /
                            (self.dev_abs_x[1] - self.dev_abs_x[0]) * self.resolution[0], slot.drag_xy[1])
            if not slot.drag[0]:
                slot.drag_start = (slot.drag_xy[0], slot.drag_start[1])
                slot.drag = (True, slot.drag[1])
        elif ev.code == EV_ABS.ABS_MT_POSITION_Y:
            slot.drag_xy = (slot.drag_xy[0], self.resolution[1] - (ev.value - self.dev_abs_y[0]) /
                            (self.dev_abs_y[1] - self.dev_abs_y[0]) * self.resolution[1])
            if not slot.drag[1]:
                slot.drag_start = (slot.drag_start[0], slot.drag_xy[1])
                slot.drag = (slot.drag[0], True)

    def render(self, **_):
        if not self.dirty:
            return

        dirty = self.dirty
        self.dirty = False

        for i, slot in enumerate(self.slots):
            self.u4fv[4 * i] = slot.drag_xy[0]
            self.u4fv[4 * i + 1] = slot.drag_xy[1]
            (z, w) = slot.drag_start
            if slot.drag[0] and slot.drag[1]:
                if not slot.touch:
                    (z, w) = (z, -w)
            else:
                (z, w) = (-z, -w)
            self.u4fv[4 * i + 2] = z
            self.u4fv[4 * i + 3] = w
            if slot.touch:
                slot.touch = False
                self.dirty = True

        if dirty:
            glsl.glUniform4fv(self.loc, len(self.u4fv), (c_float * len(self.u4fv))(*self.u4fv))


class TrackMouse(Mouse, Uniform):
    dev_abs_x: (int, int) = None
    dev_abs_y: (int, int) = None
    resolution: (int, int) = None
    _drag = (False, False)
    drag_xy = drag_start = (1, 1)
    touch = False

    def init(self, width, height, **kwargs):
        super().init(width=width, height=height, **kwargs)

        self.dev_abs_x = (self.dev.absinfo[EV_ABS.ABS_X].minimum, self.dev.absinfo[EV_ABS.ABS_X].maximum)
        self.dev_abs_y = (self.dev.absinfo[EV_ABS.ABS_Y].minimum, self.dev.absinfo[EV_ABS.ABS_Y].maximum)
        self.resolution = (width, height)

    def event(self, ev, **_):
        if ev.code == EV_KEY.BTN_TOUCH:
            if ev.value == 1:
                self.drag = True
                self.touch = True
            else:
                self.drag = False
                self._drag = (False, False)
        elif ev.code == EV_ABS.ABS_X:
            self.drag_xy = ((ev.value - self.dev_abs_x[0]) /
                            (self.dev_abs_x[1] - self.dev_abs_x[0]) * self.resolution[0], self.drag_xy[1])
            if not self._drag[0]:
                self.drag_start = (self.drag_xy[0], self.drag_start[1])
                self._drag = (True, self._drag[1])
        elif ev.code == EV_ABS.ABS_Y:
            self.drag_xy = (self.drag_xy[0], self.resolution[1] - (ev.value - self.dev_abs_y[0]) /
                            (self.dev_abs_y[1] - self.dev_abs_y[0]) * self.resolution[1])
            if not self._drag[1]:
                self.drag_start = (self.drag_start[0], self.drag_xy[1])
                self._drag = (self._drag[0], True)

    def render(self, **_):
        (z, w) = self.drag_start
        if self.drag:
            if not self.touch:
                (z, w) = (z, -w)
        else:
            (z, w) = (-z, -w)

        glsl.glUniform4f(self.loc, c_float(self.drag_xy[0]), c_float(self.drag_xy[1]), c_float(z), c_float(w))

        if self.touch:
            self.touch = False
