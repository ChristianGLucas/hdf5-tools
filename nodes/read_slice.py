import csv
import io

import h5py
import numpy as np

from gen.messages_pb2 import ReadSliceRequest, ReadSliceResult, SliceOutputFormat
from gen.axiom_context import AxiomContext
from nodes._helpers import (
    MAX_SLICE_ELEMENTS,
    InvalidArgumentError,
    NotFoundError,
    check_input_size,
    invalid_argument,
    invalid_input,
    not_found,
    open_h5_temp,
    resolve_object,
    resolve_slice_dims,
    slice_element_count,
    to_json_text,
    too_large,
)


def _stringify_cell(value):
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return str(value)


def _to_csv(arr: np.ndarray) -> str:
    squeezed = arr.squeeze()
    if squeezed.ndim > 2:
        raise InvalidArgumentError(
            f"slice's effective shape has {squeezed.ndim} non-trivial "
            "dimensions; CSV can only represent 1 or 2"
        )
    buf = io.StringIO()
    writer = csv.writer(buf)
    if squeezed.ndim == 0:
        writer.writerow(["value"])
        writer.writerow([_stringify_cell(squeezed.item())])
    elif squeezed.ndim == 1:
        writer.writerow(["index", "value"])
        for i, v in enumerate(squeezed.tolist()):
            writer.writerow([i, _stringify_cell(v)])
    else:
        ncols = squeezed.shape[1]
        writer.writerow(["row"] + [f"col_{j}" for j in range(ncols)])
        for i, row in enumerate(squeezed.tolist()):
            writer.writerow([i] + [_stringify_cell(v) for v in row])
    return buf.getvalue()


def read_slice(ax: AxiomContext, input: ReadSliceRequest) -> ReadSliceResult:
    """Extract a bounded rectangular slice of one dataset — an independent
    Python-slice-style (start, stop, step) selection per dimension — and
    return it as a JSON value (nested array matching the slice's shape,
    any rank) or as RFC 4180 CSV (1-D or 2-D slices only; higher rank with
    CSV requested is a structured INVALID_ARGUMENT error). The requested
    slice's element count is computed from shape/dtype metadata and
    checked against a documented cap *before* any data is read, so a small
    but highly compressed dataset cannot be used to force an oversized
    read; a slice over the cap is rejected with a structured TOO_LARGE
    error naming the limit, not silently truncated. A dataset_path that
    does not exist, that names a group, or a slice with more dimensions
    than the dataset has returns a structured error rather than crashing.
    """
    size_err = check_input_size(input.data)
    if size_err is not None:
        return ReadSliceResult(error=size_err)

    if not input.dataset_path:
        return ReadSliceResult(error=invalid_argument("dataset_path is required"))

    if input.output_format == SliceOutputFormat.SLICE_OUTPUT_FORMAT_UNSPECIFIED:
        return ReadSliceResult(
            error=invalid_argument(
                "output_format must be SLICE_OUTPUT_FORMAT_JSON or "
                "SLICE_OUTPUT_FORMAT_CSV"
            )
        )

    try:
        with open_h5_temp(input.data) as h5f:
            try:
                obj = resolve_object(h5f, input.dataset_path)
            except NotFoundError as e:
                return ReadSliceResult(error=not_found(str(e)))

            if not isinstance(obj, h5py.Dataset):
                return ReadSliceResult(
                    error=invalid_argument(
                        f"{input.dataset_path!r} is a group, not a dataset"
                    )
                )

            try:
                py_slices, result_shape = resolve_slice_dims(input.slice, obj.shape)
            except InvalidArgumentError as e:
                return ReadSliceResult(error=invalid_argument(str(e)))

            count = slice_element_count(result_shape)
            if count > MAX_SLICE_ELEMENTS:
                return ReadSliceResult(
                    error=too_large(
                        f"requested slice has {count} elements, over the "
                        f"{MAX_SLICE_ELEMENTS}-element cap; narrow the slice"
                    )
                )

            arr = obj[py_slices] if py_slices else obj[()]
            arr = np.asarray(arr)

            result = ReadSliceResult(
                shape=list(result_shape),
                total_elements=count,
                dtype=str(obj.dtype),
            )

            if input.output_format == SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON:
                result.json = to_json_text(arr.tolist())
            else:
                try:
                    result.csv = _to_csv(arr)
                except InvalidArgumentError as e:
                    return ReadSliceResult(error=invalid_argument(str(e)))

            return result
    except OSError as e:
        return ReadSliceResult(error=invalid_input(f"not a valid HDF5 file: {e}"))
    except Exception as e:
        return ReadSliceResult(error=invalid_input(f"could not read slice: {e}"))
