from collections import namedtuple
from ctypes import CDLL, c_int
from ctypes.util import find_library
from fcntl import ioctl
from io import FileIO
from os import fsencode, fsdecode, read, O_CLOEXEC
from select import poll
from struct import unpack_from, calcsize
from termios import FIONREAD

try:
    libc_so = find_library('c')
except RuntimeError:
    libc_so = None
_libc = CDLL(libc_so or 'libc.so.6', use_errno=True)

Event = namedtuple('Event', ['wd', 'mask', 'cookie', 'name'])

IN_ATTRIB = 0x00000004  # Metadata changed
IN_CREATE = 0x00000100  # Subfile was created

_EVENT_FMT = 'iIII'
_EVENT_SIZE = calcsize(_EVENT_FMT)


class INotify(FileIO):

    fd = property(FileIO.fileno)

    def __init__(self, flags=O_CLOEXEC):
        FileIO.__init__(self, _libc.inotify_init1(flags), mode='rb')

        self._poller = poll()
        self._poller.register(self.fileno())

    def add_watch(self, path, mask):
        return _libc.inotify_add_watch(self.fileno(), fsencode(path), mask)

    def rm_watch(self, wd):
        _libc.inotify_rm_watch(self.fileno(), wd)

    def read(self, timeout=None) -> [Event]:
        data = self._readall()
        if not data and timeout != 0 and self._poller.poll(timeout):
            data = self._readall()
        return parse_events(data)

    def _readall(self):
        bytes_avail = c_int()
        ioctl(self, FIONREAD, bytes_avail)
        if not bytes_avail.value:
            return b''
        return read(self.fileno(), bytes_avail.value)


def parse_events(data):
    pos = 0
    events = []
    while pos < len(data):
        wd, mask, cookie, name_size = unpack_from(_EVENT_FMT, data, pos)
        pos += _EVENT_SIZE + name_size
        name = data[pos - name_size:pos].split(b'\x00', 1)[0]
        events.append(Event(wd, mask, cookie, fsdecode(name)))
    return events
