"""Shared helpers for christiangeorgelucas/hdf5-tools nodes.

Bounds and rationale
---------------------
The Axiom node gRPC transport caps a message at ~4 MiB, but the *deployed*
invocation's HTTP ingress caps the request/response body far tighter
(observed ~1 MiB) — that is the real binding limit for a payload-bearing
node, and it is invisible to local `axiom test`/`axiom dev`. MAX_INPUT_BYTES
is set comfortably under that, leaving margin for base64/JSON framing
overhead at the ingress.

Decompression-bomb guard: HDF5 supports per-dataset compression (gzip/lzf/
szip). A dataset's on-disk (compressed) footprint can be tiny while its
logical (uncompressed) size is enormous — the file-size cap alone does not
bound how large a *slice read* can be. h5py exposes a Dataset's `.shape` and
`.dtype` without decompressing any data, so ReadSlice computes the
requested slice's element count from that cheap metadata and rejects the
call with TOO_LARGE *before* touching a single element if it would exceed
MAX_SLICE_ELEMENTS.

Every node operates on a private temp file: h5py/libhdf5 need a real file
path for the full feature surface used here (there is no supported
in-memory-bytes File API that covers chunked/compressed datasets reliably
across h5py's driver options), so `open_h5_temp` writes the caller's bytes
to a NamedTemporaryFile and callers MUST use it as a context manager so the
file is removed even on error — a node is otherwise stateless.
"""

import json as _json
import os
import tempfile
from contextlib import contextmanager

import h5py
import numpy as np

from gen.messages_pb2 import Error

# --- Bounds ------------------------------------------------------------

MAX_INPUT_BYTES = 640 * 1024

DEFAULT_HIERARCHY_ENTRIES = 500
MAX_HIERARCHY_ENTRIES = 2000

MAX_ATTRIBUTES = 200
MAX_ATTR_VALUE_JSON_BYTES = 4000

# Element cap for ReadSlice, computed from the *requested* slice shape
# before any data is read. At 8 bytes/element (float64/int64, the widest
# common primitive) this is <= ~1.6 MB raw; after JSON/CSV text encoding
# (worst case several bytes per number, plus separators) it stays
# comfortably under the 640 KiB response cap for realistic dtypes.
MAX_SLICE_ELEMENTS = 200_000


class NotFoundError(ValueError):
    """Raised when a group/dataset/attribute path does not exist. Callers
    map this to the NOT_FOUND error code."""


class InvalidArgumentError(ValueError):
    """Raised for a bad operation parameter. Callers map this to the
    INVALID_ARGUMENT error code."""


# --- Error helpers -------------------------------------------------------


def make_error(code: str, message: str) -> Error:
    return Error(code=code, message=message)


def too_large(message: str) -> Error:
    return make_error("TOO_LARGE", message)


def invalid_input(message: str) -> Error:
    return make_error("INVALID_INPUT", message)


def invalid_argument(message: str) -> Error:
    return make_error("INVALID_ARGUMENT", message)


def not_found(message: str) -> Error:
    return make_error("NOT_FOUND", message)


def check_input_size(data: bytes):
    """Returns an Error if `data` is empty or exceeds MAX_INPUT_BYTES, else
    None."""
    if not data:
        return invalid_input("data is empty")
    if len(data) > MAX_INPUT_BYTES:
        return too_large(
            f"input is {len(data)} bytes, over the {MAX_INPUT_BYTES}-byte cap"
        )
    return None


# --- Temp-file handling ---------------------------------------------------

# h5py/libhdf5's broadest-compatible way to open arbitrary bytes: write to
# a real temp file (cleaned up unconditionally), open read-only.
# HDF5's own file-signature check (the first 8 bytes) is what actually
# rejects non-HDF5 input; h5py.File raises OSError for anything else.


@contextmanager
def open_h5_temp(data: bytes):
    """Write `data` to a private temp file, open it read-only with h5py,
    and guarantee both the h5py.File and the temp file are closed/removed
    on the way out — including on exception. Raises OSError (uncaught,
    let the node catch it) if `data` is not a valid HDF5 file."""
    fd, path = tempfile.mkstemp(suffix=".h5")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        with h5py.File(path, "r") as h5f:
            yield h5f
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# --- dtype / value stringification ---------------------------------------


def _json_default(o):
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8")
        except UnicodeDecodeError:
            return o.hex()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def to_json_text(value) -> str:
    return _json.dumps(value, default=_json_default)


def attr_value_to_json(value):
    """Convert an h5py attribute value (numpy scalar/array, bytes, str, or
    plain Python value) to a JSON-serializable Python object."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return value


# --- HDF5 tree helpers -----------------------------------------------------


def normalize_path(path: str) -> str:
    if path in ("", "/"):
        return "/"
    return path if path.startswith("/") else "/" + path


def resolve_object(h5f: "h5py.File", path: str):
    """Resolve a path to a Group or Dataset. Raises NotFoundError if the
    path does not exist."""
    norm = normalize_path(path)
    if norm == "/":
        return h5f["/"]
    key = norm[1:]
    if key not in h5f:
        raise NotFoundError(f"no such path: {path!r}")
    return h5f[key]


def compression_opts_int(obj) -> int:
    opts = obj.compression_opts
    if opts is None:
        return -1
    if isinstance(opts, tuple):
        return int(opts[0]) if opts else -1
    try:
        return int(opts)
    except (TypeError, ValueError):
        return -1


def maxshape_field(dataset) -> list:
    """Returns [] if maxshape == shape (not resizable); otherwise one entry
    per dimension, -1 for h5py.UNLIMITED (None)."""
    if dataset.maxshape == dataset.shape:
        return []
    return [-1 if d is None else int(d) for d in dataset.maxshape]


def fillvalue_str(dataset) -> str:
    fv = dataset.fillvalue
    if fv is None:
        return ""
    if isinstance(fv, bytes):
        try:
            return fv.decode("utf-8")
        except UnicodeDecodeError:
            return fv.hex()
    if isinstance(fv, np.generic):
        val = fv.item()
    else:
        val = fv
    # A default-initialized fill value (0 / 0.0 / empty) is still reported
    # explicitly here — callers can compare against dtype defaults
    # themselves; we never hide a declared value.
    return str(val)


def resolve_slice_dims(slice_dims, shape):
    """Given the request's repeated SliceDim and the dataset's real shape,
    return (python_slices, result_shape). Raises InvalidArgumentError if
    slice_dims has more entries than the dataset has dimensions, or a step
    <= 0."""
    if len(slice_dims) > len(shape):
        raise InvalidArgumentError(
            f"slice has {len(slice_dims)} dimensions but dataset has {len(shape)}"
        )
    py_slices = []
    result_shape = []
    for i, extent in enumerate(shape):
        if i < len(slice_dims):
            d = slice_dims[i]
            start = d.start
            stop = extent if d.stop <= 0 else d.stop
            step = d.step if d.step and d.step > 0 else 1
        else:
            start, stop, step = 0, extent, 1
        start = max(0, min(start, extent))
        stop = max(0, min(stop, extent))
        py_slices.append(slice(start, stop, step))
        result_shape.append(len(range(start, stop, step)))
    return tuple(py_slices), result_shape


def slice_element_count(result_shape) -> int:
    count = 1
    for d in result_shape:
        count *= d
    return count
