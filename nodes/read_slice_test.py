import json

from gen.messages_pb2 import ReadSliceRequest, SliceDim, SliceOutputFormat
from nodes.read_slice import read_slice
from nodes._helpers import MAX_SLICE_ELEMENTS
from nodes._test_fixtures import (
    FakeAxiomContext,
    highly_compressible_h5_bytes,
    sample_h5_bytes,
)


def test_read_slice_golden_full_2d_json():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/measurements/temp",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == ""
    assert list(result.shape) == [2, 3]
    assert result.total_elements == 6
    assert result.dtype == "float64"
    assert json.loads(result.json) == [[20.5, 21.0, 21.5], [22.0, 22.5, 23.0]]


def test_read_slice_golden_full_2d_csv():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/measurements/temp",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_CSV,
        ),
    )
    assert result.error.code == ""
    rows = result.csv.strip().split("\r\n")
    assert rows[0] == "row,col_0,col_1,col_2"
    assert rows[1] == "0,20.5,21.0,21.5"
    assert rows[2] == "1,22.0,22.5,23.0"


def test_read_slice_golden_partial_1d_slice_with_step():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/counts",
            slice=[SliceDim(start=10, stop=20, step=2)],
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == ""
    assert json.loads(result.json) == [10, 12, 14, 16, 18]
    assert list(result.shape) == [5]


def test_read_slice_golden_vlen_string_dataset_decoded_to_text():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/names",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == ""
    assert json.loads(result.json) == ["alice", "bob", "carol"]


def test_read_slice_golden_vlen_string_csv():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/names",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_CSV,
        ),
    )
    assert result.error.code == ""
    rows = result.csv.strip().split("\r\n")
    assert rows == ["index,value", "0,alice", "1,bob", "2,carol"]


def test_read_slice_rejects_csv_for_true_3d_slice():
    ax = FakeAxiomContext()
    # /counts is 1-D so build a synthetic 3-D case via the cube-shaped
    # highly-compressible fixture reused with a small, real shape.
    data = highly_compressible_h5_bytes((2, 2, 2), (1, 1, 1))
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=data,
            dataset_path="/zeros",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_CSV,
        ),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_read_slice_squeezes_size_one_axis_for_csv():
    ax = FakeAxiomContext()
    data = highly_compressible_h5_bytes((1, 3, 4), (1, 3, 4))
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=data,
            dataset_path="/zeros",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_CSV,
        ),
    )
    assert result.error.code == ""
    assert list(result.shape) == [1, 3, 4]
    rows = result.csv.strip().split("\r\n")
    assert rows[0] == "row,col_0,col_1,col_2,col_3"
    assert len(rows) == 4  # header + 3 data rows


def test_read_slice_guard_fires_from_shape_metadata_not_file_size():
    """The decompression-bomb guard: a ~1 KB on-disk file with a huge
    nominal dataset shape (nothing ever written past the fill value) must
    be rejected by the pre-read element-count check, not silently
    materialized."""
    ax = FakeAxiomContext()
    data = highly_compressible_h5_bytes((10_000_000,), (1000,))
    assert len(data) < 4096  # the file itself is tiny
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=data,
            dataset_path="/zeros",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == "TOO_LARGE"
    assert result.json == ""


def test_read_slice_allows_a_narrow_slice_of_the_same_huge_dataset():
    ax = FakeAxiomContext()
    data = highly_compressible_h5_bytes((10_000_000,), (1000,))
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=data,
            dataset_path="/zeros",
            slice=[SliceDim(start=0, stop=10)],
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == ""
    assert json.loads(result.json) == [0.0] * 10


def test_read_slice_element_cap_boundary_is_exact():
    ax = FakeAxiomContext()
    data = highly_compressible_h5_bytes((MAX_SLICE_ELEMENTS + 1,), (1000,))
    at_cap = read_slice(
        ax,
        ReadSliceRequest(
            data=data,
            dataset_path="/zeros",
            slice=[SliceDim(start=0, stop=MAX_SLICE_ELEMENTS)],
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert at_cap.error.code == ""
    over_cap = read_slice(
        ax,
        ReadSliceRequest(
            data=data,
            dataset_path="/zeros",
            slice=[SliceDim(start=0, stop=MAX_SLICE_ELEMENTS + 1)],
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert over_cap.error.code == "TOO_LARGE"


def test_read_slice_error_path_not_found():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/nope",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == "NOT_FOUND"


def test_read_slice_error_path_names_a_group():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/measurements",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_read_slice_error_path_too_many_slice_dims():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            dataset_path="/counts",
            slice=[SliceDim(), SliceDim()],
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_read_slice_error_path_missing_output_format():
    ax = FakeAxiomContext()
    result = read_slice(
        ax, ReadSliceRequest(data=sample_h5_bytes(), dataset_path="/counts")
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_read_slice_error_path_missing_dataset_path():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=sample_h5_bytes(),
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_read_slice_error_path_empty_input():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=b"",
            dataset_path="/counts",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == "INVALID_INPUT"


def test_read_slice_error_path_malformed_input():
    ax = FakeAxiomContext()
    result = read_slice(
        ax,
        ReadSliceRequest(
            data=b"garbage",
            dataset_path="/counts",
            output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
        ),
    )
    assert result.error.code == "INVALID_INPUT"


def test_read_slice_is_deterministic():
    ax = FakeAxiomContext()
    req = ReadSliceRequest(
        data=sample_h5_bytes(),
        dataset_path="/counts",
        slice=[SliceDim(start=0, stop=5)],
        output_format=SliceOutputFormat.SLICE_OUTPUT_FORMAT_JSON,
    )
    r1 = read_slice(ax, req)
    r2 = read_slice(ax, req)
    assert r1.json == r2.json
