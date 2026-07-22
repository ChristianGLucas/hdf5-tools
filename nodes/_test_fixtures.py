"""Shared fixture builders for hdf5-tools node tests."""
import io
import os
import tempfile

import h5py
import numpy as np

from gen.axiom_context import SecretStatus


class FakeAxiomContext:
    """Minimal AxiomContext implementation for unit tests."""

    class _Logger:
        def debug(self, msg: str, **attrs) -> None: pass
        def info(self, msg: str, **attrs) -> None: pass
        def warn(self, msg: str, **attrs) -> None: pass
        def error(self, msg: str, **attrs) -> None: pass

    class _Secrets:
        def __init__(self, m: dict, revoked: set) -> None:
            self._m = m or {}
            self._revoked = revoked or set()
        def get(self, name: str):
            v = self._m.get(name)
            return (v, True) if v is not None else ("", False)
        def status(self, name: str) -> SecretStatus:
            if name in self._m:
                return SecretStatus.AVAILABLE
            if name in self._revoked:
                return SecretStatus.REVOKED
            return SecretStatus.UNSET

    def __init__(self, secrets_map=None, revoked_names=None) -> None:
        self.log = self._Logger()
        self.secrets = self._Secrets(secrets_map or {}, revoked_names)
        self.execution_id = "test-execution-id"
        self.flow_id = "test-flow-id"
        self.tenant_id = "test-tenant-id"


# A small, hand-known dataset layout used across tests. Every expected value
# derived from these Python literals in the tests is an independent,
# hand-computed oracle (shape/dtype/compression settings chosen up front and
# checked against literals) — not something re-derived by reading the file
# back through the same h5py machinery under test.

TEMPERATURES = [
    [20.5, 21.0, 21.5],
    [22.0, 22.5, 23.0],
]  # shape (2, 3), float64
COUNTS = list(range(100))  # shape (100,), int32, gzip-compressed, chunked (10,)
NAMES = ["alice", "bob", "carol"]  # shape (3,), vlen UTF-8 strings

ROOT_ATTRS = {"institution": "Axiom Labs", "version": 3}
GROUP_ATTRS = {"unit": "celsius"}


def sample_h5_bytes() -> bytes:
    """Build a small HDF5 file with a known, hand-designed layout:

    /                       (root group; attrs: institution, version)
    /measurements           (group; attrs: unit)
    /measurements/temp      (dataset, float64, shape (2,3), contiguous)
    /counts                 (dataset, int32, shape (100,), gzip level 4,
                             chunked (10,), shuffle enabled)
    /names                  (dataset, vlen UTF-8 string, shape (3,))
    """
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(path, "w") as f:
            for k, v in ROOT_ATTRS.items():
                f.attrs[k] = v

            grp = f.create_group("measurements")
            for k, v in GROUP_ATTRS.items():
                grp.attrs[k] = v
            grp.create_dataset(
                "temp", data=np.array(TEMPERATURES, dtype="float64")
            )

            f.create_dataset(
                "counts",
                data=np.array(COUNTS, dtype="int32"),
                chunks=(10,),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            str_dtype = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("names", data=NAMES, dtype=str_dtype)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def empty_h5_bytes() -> bytes:
    """A minimal valid HDF5 file with nothing but the root group."""
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(path, "w"):
            pass
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def corrupted_mid_file_h5_bytes() -> bytes:
    """A file that OPENS successfully (superblock/root group intact) but
    raises deep inside h5py/libhdf5 while the tree is walked or an
    attribute is read — a byte range past the superblock is scrambled
    without changing the file's length, corrupting internal object-header
    metadata. Regression fixture for the "late exception, not at open
    time" bug class: h5py/libhdf5 can raise RuntimeError (not just OSError)
    lazily during traversal, so nodes must catch broadly, not just at
    open().
    """
    data = bytearray(sample_h5_bytes())
    mid = len(data) // 2
    for i in range(mid, mid + 200):
        data[i] = (data[i] + 137) % 256
    return bytes(data)


def h5_bytes_with_oversized_attribute(value_len: int) -> bytes:
    """A file with one dataset carrying a single very long string attribute
    — used to prove ReadAttributes truncates an oversized value rather than
    blowing the response size cap."""
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(path, "w") as f:
            ds = f.create_dataset("x", data=np.array([1, 2, 3], dtype="int32"))
            ds.attrs["huge_note"] = "z" * value_len
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def h5_bytes_with_many_attributes(count: int) -> bytes:
    """A file whose root group carries `count` distinct attributes — used
    to prove ReadAttributes' MAX_ATTRIBUTES cap and truncated flag."""
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(path, "w") as f:
            for i in range(count):
                f.attrs[f"attr_{i:04d}"] = i
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def highly_compressible_h5_bytes(shape, chunk_shape) -> bytes:
    """An HDF5 file with one all-zero, max-gzip-compressed dataset of the
    requested shape — used to prove the slice-size guard fires from cheap
    shape/dtype metadata rather than the (tiny) on-disk compressed size."""
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "zeros",
                shape=shape,
                dtype="float64",
                chunks=chunk_shape,
                compression="gzip",
                compression_opts=9,
                fillvalue=0.0,
            )
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)
