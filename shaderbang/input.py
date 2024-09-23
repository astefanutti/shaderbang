from ctypes import *
from typing import Generic, Optional, Protocol, Self, TypeVar, runtime_checkable

import collections
import signal
import threading
import traceback

from copy import copy
from enum import IntFlag
from errno import ENODEV
from libevdev import Device, EV_ABS, EV_KEY, EV_REL, EV_SYN, EventsDroppedException
from lib import glsl
from pprint import pprint, pformat
from shaderbang.gl import *
from shaderbang.keycodes import keycodes
from threading import Lock, Thread


class Input:

    def __init__(self, name: str, **kwargs):
        self.name = name
        _pending_inputs.appendleft(self)

    def init(self, width, height):
        pass

    def pre_render(self, frame, time):
        pass

    def render(self, frame, time):
        pass

    def post_render(self, frame, time):
        pass


_pending_inputs = collections.deque()
_active_inputs: list[Input] = []


def _append_input(input: Input):
    # TODO: implement fine-grained input priority
    # Simple heuristic that prioritizes inputs:
    # 1. input devices > lockable inputs > inputs
    # 2. creation order
    i = 0
    if isinstance(input, InputDevice) or isinstance(input, MultiInput):
        for i, e in enumerate(_active_inputs):
            if not isinstance(e, InputDevice):
                _active_inputs.insert(i, input)
                return
    if isinstance(input, LockableInput):
        for j, e in enumerate(_active_inputs[i:]):
            if not isinstance(e, LockableInput):
                _active_inputs.insert(i+j, input)
                return
    _active_inputs.append(input)


def _list_inputs(t: type[Input]):
    for input in _active_inputs:
        if isinstance(input, MultiInput):
            for i in input.inputs:
                if isinstance(i, t):
                    yield i
        elif isinstance(input, t):
            yield input


def _print_inputs():
    pprint(_active_inputs, indent=2)


class FallbackInput(Exception):

    def __init__(self, fallback: lambda: Input):
        self.fallback = fallback


class SkipInput(Exception):

    def __init__(self, name):
        super().__init__(f"skipping input '{name}'")
        self.name = name


def _init_input(input: Input, width: int, height: int):
    # Remove the input if its device has closed
    if isinstance(input, ClosingDevice):
        Base, Group = MultiInput.support(input.target)
        if (Base is not None and Group is not None
                and (multi := next(filter(lambda i: isinstance(i, Group), _active_inputs), None))):
            multi.remove(input.target)
            if len(multi.inputs) == 1:
                single = multi.inputs[0]
                multi.remove(single)
                _append_input(single)
                _active_inputs.remove(multi)
        else:
            _active_inputs.remove(input.target)
        return

    # Check if it's an input device that's already open
    if (isinstance(input, InputDevice)
            # Skip virtual input device
            and not input.dev is None
            # Match by FD name and type
            and next(filter(lambda i: i.dev and i.dev.fd.name == input.dev.fd.name, _list_inputs(type(input))), None)):
        return

    # Hacky way to append the synthetic input at the end of the queue, so it's popped right next
    def _push_right(f: lambda: Input): i = f(); _pending_inputs.rotate(-1); return i

    # Initialize the input and handle fallback if any
    try:
        input.init(width=width, height=height)
    except FallbackInput as e:
        i = _push_right(e.fallback)
        print(f"fallback {type(input).__name__} input '{input.name}' to {type(i).__name__} input '{i.name}'")
        return
    except SkipInput:
        return
    except Exception as e:
        print(f"invalid {type(input).__name__} input '{input.name}': {str(e)}")
        return

    # Handle the input multiplexing
    Base, Group = MultiInput.support(input)
    if Base is not None and Group is not None:
        if multi := next(filter(lambda i: isinstance(i, Group)
                                and i.target.name == input.target.name,
                                _active_inputs), None):
            multi.add(input)
        elif single := next(filter(lambda i: isinstance(i, Base)
                                   and i.target.name == input.target.name,
                                   _active_inputs), None):
            _active_inputs.remove(single)
            multi = _push_right(lambda: Group(name=input.name, target=input.target))
            multi.add(single, input)
        else:
            _append_input(input)
    else:
        _append_input(input)

    # Start processing events from the input device
    if isinstance(input, InputDevice) and not input.dev is None:
        input.dev.grab()
        input.start()


def _drain(q: collections.deque):
    while True:
        try:
            yield q.pop()
        except IndexError:
            break


@CFUNCTYPE(c_int, c_uint, c_uint)
def _setup(width, height):
    try:
        # Drain all the inputs defined during initialisation
        for input in _drain(_pending_inputs):
            _init_input(input, width, height)
    except Exception as e:
        traceback.print_exception(e)
        return -1
    return 0


@CFUNCTYPE(c_int, c_uint64, c_float)
def _update(frame, time):
    try:
        # Drain pending inputs (added at runtime)
        if len(_pending_inputs):
            viewport = (c_uint*4)()
            glsl.glGetIntegerv(GL_VIEWPORT, viewport)
            (width, height) = viewport[2:4]

            for input in _drain(_pending_inputs):
                _init_input(input, width, height)

        # Loop over active inputs
        for input in _active_inputs:
            input.pre_render(frame=frame, time=time)
        for input in _active_inputs:
            input.render(frame=frame, time=time)
        for input in reversed(_active_inputs):
            input.post_render(frame=frame, time=time)
    except Exception as e:
        traceback.print_exception(e)
        return -1
    return 0

glsl.onInit(_setup)
glsl.onRender(_update)


@runtime_checkable
class EventHandler(Protocol):

    def event(self, ev, **kwargs) -> Optional[bool]: ...


class LockableInput(Input):

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.lock = Lock()
        self._pre_render = self.pre_render
        self._post_render = self.post_render
        self.pre_render = self._pre_render_wrapper
        self.post_render = self._post_render_wrapper

    def _pre_render_wrapper(self, *args, **kwargs):
        self.lock.acquire()
        parent = super(type(self), self)
        if parent.pre_render != self._pre_render:
            parent.pre_render(*args, **kwargs)
        self._pre_render(*args, **kwargs)

    def _post_render_wrapper(self, *args, **kwargs):
        self._post_render(*args, **kwargs)
        parent = super(type(self), self)
        if parent.post_render != self._post_render:
            parent.post_render(*args, **kwargs)
        self.lock.release()


# TODO: rely on implicit Generic class once Python 3.12+ becomes a requirement
I = TypeVar('I', bound=Input)


class InputDevice(Generic[I], Input):

    def __init__(self, name: str, dev: Optional[Device], target: I):
        super().__init__(name)
        self.dev = dev
        self.handler: EventHandler = self
        self.target = target
        self.thread: Optional[threading.Thread] = None

    def __repr__(self):
        return f"{super().__repr__()} ({self.dev.name if self.dev is not None else "virtual"})"

    @property
    def device(self):
        return self.dev

    def event(self, ev, target, **kwargs):
        raise NotImplementedError

    def start(self):
        self.thread = Thread(target=self._evdev_event, daemon=True)
        self.thread.start()

    @property
    def started(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _evdev_event(self):
        try:
            while True:
                try:
                    for ev in self.dev.events():
                        self.handler.event(ev, target=self)
                except EventsDroppedException:
                    for ev in self.dev.sync():
                        self.handler.event(ev, target=self)
        except IOError as e:
            if e.errno == ENODEV:
                print(f"input device {self.dev.name} unplugged")
            else:
                print(f"error reading events from input device {self.dev.name}", traceback.format_exc())
        except:
            print(f"error reading events from input device {self.dev.name}", traceback.format_exc())
        finally:
            ClosingDevice(self)


class ClosingDevice(Input):

    def __init__(self, target: InputDevice):
        super().__init__(target.name)
        self.target = target

    @property
    def device(self):
        return self.target.dev

    def init(self, width, height):
        raise NotImplementedError

    def pre_render(self, frame, time):
        raise NotImplementedError

    def render(self, frame, time):
        raise NotImplementedError

    def post_render(self, frame, time):
        raise NotImplementedError


class KeyState(IntFlag):
    DOWN = 0
    PRESSED = 1
    TOGGLED = 2


class Keyboard(LockableInput):

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.buffer = (c_ubyte * (256 * 3))()
        for i in range(0, 256 * 3):
            self.buffer[i] = 0

    def __copy__(self):
        keyboard = Keyboard.__new__(Keyboard)
        keyboard.name = self.name
        keyboard.lock = self.lock
        keyboard.buffer = copy(self.buffer)
        return keyboard

    def down(self, key) -> bool:
        return self.state(key, KeyState.DOWN)

    def pressed(self, key) -> bool:
        return self.state(key, KeyState.PRESSED)

    def toggled(self, key) -> bool:
        return self.state(key, KeyState.TOGGLED)

    def state(self, key, state: KeyState) -> bool:
        code = keycodes.get(key, -1)
        if code < 0:
            return False
        return self.buffer[256 * state + code] == 255

    def post_render(self, **kwargs):
        for i in range(256 * KeyState.PRESSED, 256 * (KeyState.PRESSED + 1)):
            self.buffer[i] = 0


class AsciiKeyboard(InputDevice[Keyboard]):

    def event(self, ev, **kwargs):
        if not ev.matches(EV_KEY):
            return False

        code = keycodes.get(ev.code, -1)
        if code < 0:
            return False

        target = self.target
        if ev.value == 0:
            with target.lock:
                target.buffer[256 * KeyState.DOWN + code] = 0
        elif ev.value == 1:
            with target.lock:
                target.buffer[256 * KeyState.DOWN + code] = 255
                target.buffer[256 * KeyState.PRESSED + code] = 255
                target.buffer[256 * KeyState.TOGGLED + code] = 255 - target.buffer[256 * KeyState.TOGGLED + code]

            # The keyboard device has been grabbed, so events are not sent to virtual
            # devices. The main thread has to be interrupted on CTRL+C explicitly.
            if ev.matches(EV_KEY.KEY_C) and target.buffer[17]:
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        else:
            return False

        return True


class Mouse(LockableInput):

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.mouseX = self.mouseY = None
        self.startX = self.startY = None
        self.deltaX = self.deltaY = self.deltaW = 0
        self.button = None
        self._click = False
        self._drag = False
        self.resolution: (int, int) = None

    def __copy__(self):
        mouse = Mouse.__new__(Mouse)
        mouse.name = self.name
        mouse.lock = self.lock
        mouse.mouseX = self.mouseX
        mouse.mouseY = self.mouseY
        mouse.startX = self.startX
        mouse.startY = self.startY
        mouse.deltaX = self.deltaX
        mouse.deltaY = self.deltaY
        mouse.deltaW = self.deltaW
        mouse.button = self.button
        mouse._click = self._click
        mouse._drag = self._drag
        mouse.resolution = self.resolution
        return mouse

    def init(self, width, height):
        super().init(width=width, height=height)
        self.resolution = (width, height)

    @property
    def click(self):
        return self._click and self.mouseX is not None and self.mouseY is not None

    @click.setter
    def click(self, value):
        self._click = value

    @property
    def drag(self):
        return self._drag and self.mouseX is not None and self.mouseY is not None

    @drag.setter
    def drag(self, value):
        self._drag = value

    def post_render(self, **kwargs):
        if self.click:
            self.click = False
        self.deltaX = self.deltaY = self.deltaW = 0


class MouseDevice(InputDevice[Mouse]):

    def __init__(self, name: str, dev: Device, target: Mouse):
        super().__init__(name, dev=dev, target=target)
        self.resolution: (int, int) = None

    def init(self, width, height):
        super().init(width=width, height=height)
        self.resolution = (width, height)

    def proj_x(self, x:float) -> float:
        return max(1, min(self.target.mouseX + x, self.resolution[0]))

    def proj_y(self, y:float) -> float:
        return max(1, min(self.target.mouseY + y, self.resolution[1]))


class ButtonMouse(MouseDevice):

    def init(self, width, height):
        super().init(width=width, height=height)
        self.target.mouseX = width // 2
        self.target.mouseY = height // 2

    def event(self, ev, **kwargs):
        target = self.target
        if ev.matches(EV_KEY):
            if ev.code != EV_KEY.BTN_LEFT and ev.code != EV_KEY.BTN_RIGHT:
                return False
            with target.lock:
                if ev.value == 1:
                    if not target.button:
                        target.button = ev.code
                        target.click = True
                        target.drag = True
                        target.startX = target.mouseX
                        target.startY = target.mouseY
                elif ev.value == 0 and ev.code == target.button:
                    target.button = None
                    target.drag = False
                    target.startX = target.startY = None
        elif ev.matches(EV_REL):
            if ev.code == EV_REL.REL_X:
                with target.lock:
                    x = self.proj_x(ev.value)
                    target.deltaX += x - target.mouseX
                    target.mouseX = x
            elif ev.code == EV_REL.REL_Y:
                with target.lock:
                    y = self.proj_y(ev.value)
                    target.deltaY += y - target.mouseY
                    target.mouseY = y
            elif ev.matches(EV_REL.REL_WHEEL):
                with target.lock:
                    target.deltaW += ev.value
        else:
            return False
        return True


class TouchMouse(MouseDevice):

    def proj_x(self, x: float) -> float:
        return ((x - self.dev.absinfo[EV_ABS.ABS_X].minimum)
                / (self.dev.absinfo[EV_ABS.ABS_X].maximum - self.dev.absinfo[EV_ABS.ABS_X].minimum)
                * self.resolution[0])

    def proj_y(self, y: float) -> float:
        return ((y - self.dev.absinfo[EV_ABS.ABS_Y].minimum)
                / (self.dev.absinfo[EV_ABS.ABS_Y].maximum - self.dev.absinfo[EV_ABS.ABS_Y].minimum)
                * self.resolution[1])

    def event(self, ev, **kwargs):
        target = self.target
        if ev.code == EV_KEY.BTN_TOUCH:
            if ev.value == 1:
                with target.lock:
                    target.drag = True
                    target.click = True
            else:
                with target.lock:
                    target.drag = False
                    target.startX = target.startY = None
                    target.mouseX = target.mouseY = None
        elif ev.code == EV_ABS.ABS_X:
            with target.lock:
                target.mouseX = self.proj_x(ev.value)
                if not target.startX:
                    target.startX = target.mouseX
        elif ev.code == EV_ABS.ABS_Y:
            with target.lock:
                target.mouseY = self.proj_y(ev.value)
                if not target.startY:
                    target.startY = target.mouseY
        else:
            return False
        return True


class TrackMouse(MouseDevice):

    def __init__(self, name: str, dev: Optional[Device], target: Mouse):
        super().__init__(name, dev=dev, target=target)
        self._absX = self._absY = None
        self.finger = False

    def init(self, width, height):
        super().init(width=width, height=height)
        self.target.mouseX = width // 2
        self.target.mouseY = height // 2

    def proj_x(self, x: float) -> float:
        return x

    def proj_y(self, y: float) -> float:
        return y

    def event(self, ev, **kwargs):
        target = self.target
        if ev.code == EV_KEY.BTN_TOUCH:
            if ev.value == 0:
                with target.lock:
                    self._absX = self._absY = None
        elif ev.code == EV_KEY.BTN_TOOL_FINGER:
            if ev.value == 1:
                self.finger = True
                self._absX = self._absY = None
            else:
                self.finger = False
        elif ev.code == EV_KEY.BTN_LEFT:
            if ev.value == 1:
                with target.lock:
                    target.button = ev.code
                    target.click = True
                    target.drag = True
                    target.startX = target.mouseX
                    target.startY = target.mouseY
            else:
                with target.lock:
                    target.button = None
                    target.drag = False
                    target.startX = target.startY = None
        elif ev.code == EV_ABS.ABS_X:
            with target.lock:
                abs_x = self.proj_x(ev.value)
                if self._absX is None:
                    self._absX = abs_x
                else:
                    x = super().proj_x(abs_x - self._absX)
                    if self.finger:
                        target.deltaX += x - target.mouseX
                        target.mouseX = x
                    self._absX = abs_x
        elif ev.code == EV_ABS.ABS_Y:
            with target.lock:
                abs_y = self.proj_y(ev.value)
                if self._absY is None:
                    self._absY = abs_y
                else:
                    y = super().proj_y(abs_y - self._absY)
                    if self.finger:
                        target.deltaY += y - target.mouseY
                        target.mouseY = y
                    self._absY = abs_y
        else:
            return False
        return True


class TouchSlot:

    def __init__(self):
        self._touch = False
        self._drag = False
        self.startX = self.startY = None
        self.touchX = self.touchY = None
        self.deltaX = self.deltaY = 0
        self._prevX = self._prevY = None
        self.prevX = self.prevY = None

    @property
    def touch(self):
        return self._touch and self.touchX is not None and self.touchY is not None

    @touch.setter
    def touch(self, value):
        self._touch = value

    @property
    def drag(self):
        return self._drag and self.touchX is not None and self.touchY is not None

    @drag.setter
    def drag(self, value):
        self._drag = value


# TODO: rely on implicit Generic class once Python 3.12+ becomes a requirement
S = TypeVar('S', bound=TouchSlot)


class MultiTouch(Generic[S], LockableInput):

    def __init__(self, name: str, slot_type: type[S] = TouchSlot, **kwargs):
        super().__init__(name, **kwargs)
        self.slots: list[S] = []
        self._slot_type: type[S] = slot_type
        self.resolution: (int, int) = None

    def init(self, width, height):
        super().init(width=width, height=height)
        self.resolution = (width, height)

    def pre_render(self, **kwargs):
        for slot in self.slots:
            if slot.drag:
                slot.prevX = slot._prevX if slot._prevX else slot.touchX
                slot.deltaX = slot.touchX - slot.prevX
                slot._prevX = slot.touchX
                slot.prevY = slot._prevY if slot._prevY else slot.touchY
                slot.deltaY = slot.touchY - slot.prevY
                slot._prevY = slot.touchY

    def post_render(self, **kwargs):
        for slot in self.slots:
            if slot.touch:
                slot.touch = False


class MultiTouchDevice(Generic[S], InputDevice[MultiTouch[S]]):

    EVENTS = [
        EV_ABS.ABS_MT_TRACKING_ID,
        EV_ABS.ABS_MT_POSITION_X,
        EV_ABS.ABS_MT_POSITION_Y,
    ]

    def __init__(self, name: str, dev: Device, target: MultiTouch[S]):
        super().__init__(name, dev, target)
        self.resolution: (int, int) = None

    def init(self, width, height):
        super().init(width=width, height=height)
        self.resolution = (width, height)
        self.target.slots = [self.target._slot_type() for _ in range(self.dev.num_slots)]

    def proj_x(self, x: float) -> float:
        raise NotImplementedError()

    def proj_y(self, y: float) -> float:
        raise NotImplementedError()

    def event(self, ev, **kwargs):
        target = self.target
        slot = target.slots[self.dev.current_slot]
        if ev.code == EV_ABS.ABS_MT_TRACKING_ID:
            with target.lock:
                if ev.value >= 0:
                    slot.drag = True
                    slot.touch = True
                else:
                    slot.drag = False
                    slot.startX = slot.startY = None
                    slot.touchX = slot.touchY = None
                    slot.prevX = slot.prevY = None
                    slot._prevX = slot._prevY = None
                    slot.deltaX = slot.deltaY = 0
        elif ev.code == EV_ABS.ABS_MT_POSITION_X:
            with target.lock:
                slot.touchX = self.proj_x(ev.value)
                if not slot.startX:
                    slot.startX = slot.touchX
                if not slot._prevX:
                    slot._prevX = slot.touchX
        elif ev.code == EV_ABS.ABS_MT_POSITION_Y:
            with target.lock:
                slot.touchY = self.proj_y(ev.value)
                if not slot.startY:
                    slot.startY = slot.touchY
                if not slot._prevY:
                    slot._prevY = slot.touchY
        else:
            return False
        return True


class Touchscreen(Generic[S], MultiTouchDevice[S]):

    def proj_x(self, x: float) -> float:
        return ((x - self.dev.absinfo[EV_ABS.ABS_X].minimum)
                / (self.dev.absinfo[EV_ABS.ABS_X].maximum - self.dev.absinfo[EV_ABS.ABS_X].minimum)
                * self.resolution[0])

    def proj_y(self, y: float) -> float:
        return ((y - self.dev.absinfo[EV_ABS.ABS_Y].minimum)
                / (self.dev.absinfo[EV_ABS.ABS_Y].maximum - self.dev.absinfo[EV_ABS.ABS_Y].minimum)
                * self.resolution[1])


class Trackpad(Generic[S], MultiTouchDevice[S]):

    def __init__(self, name: str, dev: Device, target: MultiTouch[S], mouse: Optional[Mouse]):
        super().__init__(name, dev, target)
        # Mouse virtual device
        if mouse:
            self.mouse = TrackMouse(name, None, mouse)

    def proj_x(self, x: float) -> float:
        return x

    def proj_y(self, y: float) -> float:
        return y

    def event(self, ev, **kwargs):
        if ev.code in MultiTouchDevice.EVENTS:
            super().event(ev, **kwargs)
        elif self.mouse:
            kwargs["target"] = self.mouse
            self.mouse.handler.event(ev, **kwargs)


# TODO: rely on implicit Generic class once Python 3.12+ becomes a requirement
T = TypeVar('T', bound=Keyboard|Mouse)


class MultiInput(Generic[T], Input):

    Groups: list[tuple[type[InputDevice[T]], type[Self]]] = []

    def __init__(self, name: str, target: T, **kwargs):
        super().__init__(name, target=target, **kwargs)
        self.target = target
        self.inputs: list[InputDevice[T]] = []
        self.target = target

    def __repr__(self):
        return f"{super().__repr__()}\n{pformat(self.inputs, compact=False, indent=2)}"

    @classmethod
    def support(cls, input: Input) -> tuple[Optional[type[InputDevice[T]]], Optional[type[Self]]]:
        for Base, Group in reversed(MultiInput.Groups):
            if isinstance(input, Base):
                return Base, Group
        return None, None

    def add(self, *inputs: InputDevice[T]):
        for input in inputs:
            input.handler = self
            input.target = copy(input.target)
            self.inputs.append(input)

    def remove(self, *inputs: InputDevice[T]):
        for input in inputs:
            self.inputs.remove(input)
            input.handler = input
            input.target = self.target

    def event(self, ev, target, **kwargs):
        return target.event(ev, **kwargs)

    def pre_render(self, **kwargs):
        super().pre_render(**kwargs)
        for input in self.inputs:
            input.target.pre_render(**kwargs)

    def post_render(self, **kwargs):
        for input in self.inputs:
            input.target.post_render(**kwargs)
        super().post_render(**kwargs)


class MultiKeyboard(MultiInput[Keyboard]):

    def event(self, ev, target, **kwargs):
        if not super().event(ev, target, **kwargs):
            return

        if not ev.matches(EV_KEY):
            return

        code = keycodes.get(ev.code, -1)
        if code < 0:
            return

        target = self.target
        with target.lock:
            target.buffer[256 * KeyState.DOWN + code] = 0
            target.buffer[256 * KeyState.PRESSED + code] = 0
            for input in self.inputs:
                target.buffer[256 * KeyState.DOWN + code] |= input.target.buffer[256 * KeyState.DOWN + code]
                target.buffer[256 * KeyState.PRESSED + code] |= input.target.buffer[256 * KeyState.PRESSED + code]
            if ev.value == 1:
                target.buffer[256 * KeyState.TOGGLED + code] = 255 - target.buffer[256 * KeyState.TOGGLED + code]


MultiInput.Groups.append((AsciiKeyboard, MultiKeyboard))


class MultiMouse(MultiInput[Mouse]):

    def __init__(self, name: str, target: Mouse, **kwargs):
        super().__init__(name, target=target, **kwargs)
        self.button = None

    def init(self, width, height):
        super().init(width=width, height=height)
        target = self.target
        target.mouseX = width // 2
        target.mouseY = height // 2

    def event(self, ev, target: MouseDevice, **kwargs):
        # Forward the event to the target
        if not super().event(ev, target, **kwargs):
            return

        device = target
        source = device.target
        target = self.target

        # Two-phase sync target -> multi-mouse -> other mice
        with target.lock:
            # 1) Update the multi-mouse from the target
            if source.mouseX:
                x = target.mouseX
                target.mouseX = source.mouseX
                target.deltaX += target.mouseX - x
            if source.mouseY:
                y = target.mouseY
                target.mouseY = source.mouseY
                target.deltaY += target.mouseY - y
            if source.click:
                target.startX = source.startX
                target.startY = source.startY
            target.click = False
            target.drag = False
            target.button = None
            target.deltaW = 0
            for input in self.inputs:
                if button := getattr(input.target, 'button', None):
                    target.button = button
                target.click |= input.target.click
                target.drag |= input.target.drag
                target.deltaW += input.target.deltaW

            # 2) Sync other mice with the multi-mouse
            for input in filter(lambda i: i != device, self.inputs):
                input.target.mouseX = target.mouseX
                input.target.mouseY = target.mouseY
                if input.target.startX:
                    input.target.startX = target.startX
                if input.target.startY:
                    input.target.startY = target.startY


MultiInput.Groups.append((MouseDevice, MultiMouse))
