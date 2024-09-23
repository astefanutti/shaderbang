# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

from ctypes import *
from pathlib import Path
from typing import Iterator

from libevdev import Device
from PIL import Image

import shaderbang
from shaderbang import lib as sb
from shaderbang.gl import *
from shaderbang.input import Input, FallbackInput, MultiInput, MultiTouch, SkipInput, TouchSlot
from shaderbang.input import _list_inputs


_default_inputs = ["iMouse", "iKeyboard"]

_texture_units: Iterator[int] | None = None


def _init_slots():
    global _texture_units
    max_texture_image_units = pointer(c_uint())
    sb.glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, max_texture_image_units)
    value = max_texture_image_units.contents.value
    _texture_units = iter([i for i in range(value if value > 0 else 16)])


class NoActiveUniformVariable(Exception):

    def __init__(self, name):
        super().__init__(f"no active uniform variable '{name}'")
        self.name = name


class Uniform(Input):

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.loc = None

    def init(self, **kwargs):
        super().init(**kwargs)

        program = c_uint()
        sb.glGetIntegerv(GL_CURRENT_PROGRAM, pointer(program))

        self.loc = sb.glGetUniformLocation(program, bytes(self.name, "utf-8"))
        if self.loc < 0:
            if self.name in _default_inputs:
                # Avoid printing warnings for defaulted input names that are not provided by users
                raise SkipInput(self.name)
            raise NoActiveUniformVariable(self.name)


class Texture(Uniform):

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.tex = None
        self.unit = None

    def init(self, **kwargs):
        super().init(**kwargs)

        if not _texture_units:
            _init_slots()

        # Re-use existing texture if any, for example in the case of multiple keyboards with the same name
        texture = next(filter(lambda i: i.name == self.name, _list_inputs(Texture)), None)
        if texture is not None:
            self.tex = texture.tex
            self.unit = texture.unit
            return

        self.tex = c_uint()
        self.unit = next(_texture_units)
        sb.glUniform1i(self.loc, self.unit)
        sb.glActiveTexture(GL_TEXTURE0 + self.unit)
        sb.glGenTextures(1, pointer(self.tex))


class ImageTexture(Texture):

    def __init__(self, name, path, transpose=None):
        super().__init__(name)
        self.path = path
        self.transpose = transpose

    def init(self, **kwargs):
        super().init(**kwargs)

        sb.glBindTexture(GL_TEXTURE_2D, self.tex)
        sb.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        sb.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        sb.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        sb.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = Image.open(self.path)
        if self.transpose:
            image = image.transpose(getattr(Image, self.transpose))
        data = image.convert("RGBA").tobytes()
        sb.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        sb.glGenerateMipmap(GL_TEXTURE_2D)
        image.close()


class VolumeTexture(Texture):

    def __init__(self, name, path):
        super().__init__(name)
        self.path = path

    def init(self, **kwargs):
        super().init(**kwargs)

        data = Path(self.path).read_bytes()
        width = int.from_bytes(data[4:8], byteorder="little")
        height = int.from_bytes(data[8:12], byteorder="little")
        depth = int.from_bytes(data[12:16], byteorder="little")
        channels = int.from_bytes(data[16:17], byteorder="little")
        bin_format = int.from_bytes(data[18:20], byteorder="little")
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

        sb.glBindTexture(GL_TEXTURE_3D, self.tex)
        sb.glTexImage3D(GL_TEXTURE_3D, 0, internal_format, width, height, depth, 0, source_format,
                        GL_FLOAT if is_float else GL_UNSIGNED_BYTE, data[20:])
        sb.glGenerateMipmap(GL_TEXTURE_3D)


class CubemapTexture(Texture):

    def __init__(self, name, path):
        super().__init__(name)
        self.path = path

    def init(self, **kwargs):
        super().init(**kwargs)

        sb.glBindTexture(GL_TEXTURE_CUBE_MAP, self.tex)
        sb.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        sb.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        sb.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        sb.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        sb.glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = Image.open(self.path)
        data = image.convert("RGB").tobytes()
        channels = len(image.getbands())
        for i in range(0, 6):
            sb.glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB8, image.width, image.width, 0, GL_RGB,
                            GL_UNSIGNED_BYTE, data[i*image.width**2*channels:])
        sb.glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        image.close()


class Keyboard(shaderbang.input.Keyboard, Texture):

    def render(self, frame, **kwargs):
        sb.glActiveTexture(GL_TEXTURE0 + self.unit)
        sb.glBindTexture(GL_TEXTURE_2D, self.tex)
        sb.glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 256, 3, 0, GL_RED, GL_UNSIGNED_BYTE, self.buffer)
        sb.glGenerateMipmap(GL_TEXTURE_2D)


class AsciiKeyboard(Keyboard, shaderbang.input.AsciiKeyboard):

    def __init__(self, name: str, dev: Device):
        super().__init__(name, dev=dev, target=self)


class MultiKeyboard(shaderbang.input.MultiKeyboard, Keyboard):

    def __init__(self, name: str, target, **kwargs):
        super().__init__(name, target=self, **kwargs)


MultiInput.Groups.append((AsciiKeyboard, MultiKeyboard))


class Mouse(shaderbang.input.Mouse, Uniform):

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        # self.mouseX = self.mouseY = 0
        self.beginX = self.beginY = 0
        self.endX = self.endY = 0
        self.height = None

    def init(self, height, **kwargs):
        super().init(height=height, **kwargs)
        self.height = height
        self.mouseY = height

    def render(self, **kwargs):
        if self.click:
            self.beginX = self.startX
            self.beginY = self.startY
        z = self.beginX
        w = self.height - self.beginY
        if self.drag:
            self.endX = self.mouseX
            self.endY = self.mouseY
            if not self.click:
                (z, w) = (z, -w)
        else:
            (z, w) = (-z, -w)
        x = self.endX
        y = self.height - self.endY

        sb.glUniform4f(self.loc, c_float(x), c_float(y), c_float(z), c_float(w))


class MultiMouse(shaderbang.input.MultiMouse, Mouse):

    def __init__(self, name: str, target, **kwargs):
        super().__init__(name, target=self, **kwargs)


MultiInput.Groups.append((shaderbang.input.MouseDevice, MultiMouse))


class ButtonMouse(Mouse, shaderbang.input.ButtonMouse):

    def __init__(self, name: str, dev: Device):
        super().__init__(name, dev=dev, target=self)


class TouchMouse(Mouse, shaderbang.input.TouchMouse):

    def __init__(self, name: str, dev: Device):
        super().__init__(name, dev=dev, target=self)


class TrackMouse(Mouse, shaderbang.input.TrackMouse):

    def __init__(self, name: str, dev: Device):
        super().__init__(name, dev=dev, target=self)


class Touchscreen(MultiTouch[TouchSlot], shaderbang.input.Touchscreen[TouchSlot], Uniform):

    def __init__(self, name: str, dev: Device):
        super().__init__(name, dev=dev, target=self)
        self.u4fv: Array[c_float] = (c_float * 0)()

    def init(self, **kwargs):
        try:
            super().init(**kwargs)
        except NoActiveUniformVariable:
            # Fall back to using the touchscreen as a mouse device
            raise FallbackInput(lambda: TouchMouse("iMouse", self.dev))

        self.u4fv = (c_float * (len(self.slots) * 4))()

    def render(self, **kwargs):
        for i, slot in enumerate(self.slots):
            z = w = 0.0
            if slot.drag:
                self.u4fv[4 * i] = slot.touchX
                self.u4fv[4 * i + 1] = self.resolution[1] - slot.touchY
                z = slot.startX
                w = self.resolution[1] - slot.startY
                if not slot.touch:
                    (z, w) = (z, -w)
            else:
                (z, w) = (-z, -w)
            self.u4fv[4 * i + 2] = z
            self.u4fv[4 * i + 3] = w

        sb.glUniform4fv(self.loc, len(self.u4fv), self.u4fv)
