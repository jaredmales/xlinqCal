"""Read MagAO-X streamWriter `.xrif` archives directly, without decoding to FITS.

Uses libxrif through ctypes (`LIBXRIF` overrides the library path). A streamWriter archive is an xrif header and compressed image block, followed by an xrif header and
compressed timing block (5 uint64 per frame: cnt0, atime sec, atime nsec, write time sec, write time nsec),
the layout `xrif2fits` reads.

    frames, timing = read_xrif(path)   # frames (N, height, width) in the stored type; timing (N, 5) uint64

Frames with a zero atime are empty placeholders of a partly filled archive and are dropped.
"""

import ctypes
import ctypes.util
import os

import numpy as np

XRIF_HEADER_SIZE = 48
_TYPES = {1: np.uint8, 2: np.int8, 3: np.uint16, 4: np.int16, 5: np.uint32, 6: np.int32, 7: np.uint64,
          8: np.int64, 9: np.float32, 10: np.float64, 13: np.float16}


class _Handle(ctypes.Structure):
    """The leading fields of `xrif_handle` (xrif.h), up to the raw buffer."""
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("depth", ctypes.c_uint32),
                ("frames", ctypes.c_uint32), ("type_code", ctypes.c_uint8), ("data_size", ctypes.c_size_t),
                ("raw_size", ctypes.c_size_t), ("compressed_size", ctypes.c_size_t),
                ("difference_method", ctypes.c_int), ("reorder_method", ctypes.c_int),
                ("compress_method", ctypes.c_int), ("lz4_acceleration", ctypes.c_int),
                ("omp_parallel", ctypes.c_int), ("omp_numthreads", ctypes.c_int),
                ("compress_on_raw", ctypes.c_ubyte), ("own_raw", ctypes.c_ubyte),
                ("raw_buffer", ctypes.c_void_p), ("raw_buffer_size", ctypes.c_size_t)]


_lib = None


def _load():
    global _lib
    if _lib is None:
        path = os.environ.get("LIBXRIF") or ctypes.util.find_library("xrif") or "/usr/local/lib/libxrif.so"
        lib = ctypes.CDLL(path)
        H = ctypes.POINTER(_Handle)
        lib.xrif_new.argtypes = [ctypes.POINTER(H)]
        lib.xrif_read_header.argtypes = [H, ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p]
        for f in ("xrif_allocate_raw", "xrif_allocate_reordered", "xrif_decode", "xrif_delete"):
            getattr(lib, f).argtypes = [H]
        for f in ("xrif_new", "xrif_read_header", "xrif_allocate_raw", "xrif_allocate_reordered", "xrif_decode",
                  "xrif_delete"):
            getattr(lib, f).restype = ctypes.c_int
        _lib = lib
    return _lib


def _block(lib, fp, path):
    """Decode the next header + compressed block of an open archive into a (frames, height, width) array."""
    hdr = fp.read(XRIF_HEADER_SIZE)
    if len(hdr) != XRIF_HEADER_SIZE:
        raise IOError(f"{path}: short xrif header")
    h = ctypes.POINTER(_Handle)()
    if lib.xrif_new(ctypes.byref(h)) != 0:
        raise IOError("xrif_new failed")
    try:
        hs = ctypes.c_uint32()
        if lib.xrif_read_header(h, ctypes.byref(hs), hdr) != 0:
            raise IOError(f"{path}: bad xrif header")
        if hs.value > XRIF_HEADER_SIZE:
            fp.read(hs.value - XRIF_HEADER_SIZE)
        hd = h.contents
        if hd.type_code not in _TYPES:
            raise IOError(f"{path}: unsupported xrif type code {hd.type_code}")
        if lib.xrif_allocate_raw(h) != 0 or lib.xrif_allocate_reordered(h) != 0:
            raise MemoryError(f"{path}: xrif allocation failed")
        data = fp.read(hd.compressed_size)
        if len(data) != hd.compressed_size:
            raise IOError(f"{path}: short xrif data block")
        ctypes.memmove(hd.raw_buffer, data, len(data))
        if lib.xrif_decode(h) != 0:
            raise IOError(f"{path}: xrif_decode failed")
        dt = _TYPES[hd.type_code]
        n = hd.width * hd.height * hd.depth * hd.frames
        out = np.frombuffer(ctypes.string_at(hd.raw_buffer, n * np.dtype(dt).itemsize), dtype=dt).copy()
        return out.reshape(hd.frames, hd.height * hd.depth, hd.width)
    finally:
        lib.xrif_delete(h)


def read_xrif(path):
    """Return (frames (N, height, width), timing (N, 5) uint64) from a streamWriter archive."""
    lib = _load()
    with open(path, "rb") as fp:
        frames = _block(lib, fp, path)
        timing = _block(lib, fp, path).reshape(-1, 5).astype(np.uint64)
    if len(timing) != len(frames):
        raise IOError(f"{path}: {len(frames)} frames but {len(timing)} timing records")
    ok = timing[:, 1] != 0
    return frames[ok], timing[ok]


def file_time(path):
    """UTC start time [Unix s] from a streamWriter file name `<stream>_YYYYMMDDHHMMSSnnnnnnnnn.<ext>`, else None."""
    import calendar
    import re
    import time
    m = re.search(r"_(\d{14})(\d{9})\.\w+$", os.path.basename(path))
    if not m:
        return None
    return calendar.timegm(time.strptime(m.group(1), "%Y%m%d%H%M%S")) + int(m.group(2)) * 1e-9
