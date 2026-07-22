import h5py
import numpy as np

from gen.messages_pb2 import ReadDatasetInfoRequest, ReadDatasetInfoResult
from gen.axiom_context import AxiomContext
from nodes._helpers import (
    NotFoundError,
    compression_opts_int,
    fillvalue_str,
    invalid_argument,
    invalid_input,
    maxshape_field,
    not_found,
    open_h5_temp,
    resolve_object,
)


def read_dataset_info(ax: AxiomContext, input: ReadDatasetInfoRequest) -> ReadDatasetInfoResult:
    """Read one dataset's full storage-layout metadata without reading any
    of its actual data: dtype, shape, maxshape (resizable-dimension
    bounds), chunk shape, compression filter and its opts, shuffle/
    fletcher32 filter flags, declared fill value, logical (uncompressed)
    byte size, and attribute count. A dataset_path that does not exist, or
    that names a group instead of a dataset, returns a structured error
    (NOT_FOUND / INVALID_ARGUMENT respectively) rather than crashing.
    """
    if not input.dataset_path:
        return ReadDatasetInfoResult(
            error=invalid_argument("dataset_path is required")
        )

    try:
        with open_h5_temp(input.data) as h5f:
            try:
                obj = resolve_object(h5f, input.dataset_path)
            except NotFoundError as e:
                return ReadDatasetInfoResult(error=not_found(str(e)))

            if not isinstance(obj, h5py.Dataset):
                return ReadDatasetInfoResult(
                    error=invalid_argument(
                        f"{input.dataset_path!r} is a group, not a dataset"
                    )
                )

            nbytes = int(np.prod(obj.shape, dtype=np.int64)) * obj.dtype.itemsize if obj.shape else obj.dtype.itemsize

            return ReadDatasetInfoResult(
                dtype=str(obj.dtype),
                shape=list(obj.shape),
                maxshape=maxshape_field(obj),
                chunks=list(obj.chunks) if obj.chunks else [],
                compression=obj.compression or "",
                compression_opts=compression_opts_int(obj),
                shuffle=bool(obj.shuffle),
                fletcher32=bool(obj.fletcher32),
                fillvalue=fillvalue_str(obj),
                nbytes=int(nbytes),
                num_attrs=len(obj.attrs),
            )
    except OSError as e:
        return ReadDatasetInfoResult(error=invalid_input(f"not a valid HDF5 file: {e}"))
    except Exception as e:
        return ReadDatasetInfoResult(
            error=invalid_input(f"could not read dataset metadata: {e}")
        )
