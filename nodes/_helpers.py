"""Shared helpers for christiangeorgelucas/hdf5-tools nodes.

Every node is a pure input->output transform: it does not enforce any
payload-size, element-count, or other resource/cost bound on itself. A
node is a function; the Axiom platform (ingress, gRPC transport, pod
resource limits, sandboxing) owns all size/memory/DoS containment,
including decompression-bomb-style concerns. The only things a node
rejects are genuine domain errors: input that is not a well-formed HDF5
file, or an operation parameter (path, dimension count, ...) that does
not make sense against the file's actual structure.

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


class NotFoundError(ValueError):
    """Raised when a group/dataset/attribute path does not exist. Callers
    map this to the NOT_FOUND error code."""


class InvalidArgumentError(ValueError):
    """Raised for a bad operation parameter. Callers map this to the
    INVALID_ARGUMENT error code."""


# --- Error helpers -------------------------------------------------------


def make_error(code: str, message: str) -> Error:
    return Error(code=code, message=message)


def invalid_input(message: str) -> Error:
    return make_error("INVALID_INPUT", message)


def invalid_argument(message: str) -> Error:
    return make_error("INVALID_ARGUMENT", message)


def not_found(message: str) -> Error:
    return make_error("NOT_FOUND", message)


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


def _resolve_bound(value: int, extent: int) -> int:
    """Apply real Python negative-index semantics: a negative value is an
    offset from the end (-1 == extent - 1), then clamp into [0, extent]."""
    if value < 0:
        value = extent + value
    return max(0, min(value, extent))


def resolve_slice_dims(slice_dims, shape):
    """Given the request's repeated SliceDim and the dataset's real shape,
    return (python_slices, result_shape). Raises InvalidArgumentError if
    slice_dims has more entries than the dataset has dimensions, or a step
    <= 0.

    start and stop both follow real Python negative-index semantics
    (negative == offset from the end, e.g. -5 on a length-100 dimension is
    index 95), with one deliberate deviation: proto3 scalar fields have no
    unset-vs-zero distinction, so stop == 0 (the un-set default) is the
    sentinel for "through the end of this dimension" rather than literally
    index 0 — a caller who genuinely wants an empty result should use a
    stop equal to start instead.
    """
    if len(slice_dims) > len(shape):
        raise InvalidArgumentError(
            f"slice has {len(slice_dims)} dimensions but dataset has {len(shape)}"
        )
    py_slices = []
    result_shape = []
    for i, extent in enumerate(shape):
        if i < len(slice_dims):
            d = slice_dims[i]
            start = _resolve_bound(d.start, extent)
            stop = extent if d.stop == 0 else _resolve_bound(d.stop, extent)
            step = d.step if d.step and d.step > 0 else 1
        else:
            start, stop, step = 0, extent, 1
        py_slices.append(slice(start, stop, step))
        result_shape.append(len(range(start, stop, step)))
    return tuple(py_slices), result_shape


def slice_element_count(result_shape) -> int:
    count = 1
    for d in result_shape:
        count *= d
    return count
