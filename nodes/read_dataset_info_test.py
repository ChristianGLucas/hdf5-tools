from gen.messages_pb2 import ReadDatasetInfoRequest
from nodes.read_dataset_info import read_dataset_info
from nodes._test_fixtures import FakeAxiomContext, sample_h5_bytes


def test_read_dataset_info_golden_contiguous_dataset():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=sample_h5_bytes(), dataset_path="/measurements/temp")
    )
    assert result.error.code == ""
    assert result.dtype == "float64"
    assert list(result.shape) == [2, 3]
    assert list(result.chunks) == []  # not chunked (default contiguous storage)
    assert result.compression == ""
    assert result.compression_opts == -1
    assert result.shuffle is False
    assert result.fletcher32 is False
    assert result.nbytes == 2 * 3 * 8  # 6 float64 elements
    assert result.num_attrs == 0


def test_read_dataset_info_golden_chunked_compressed_dataset():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=sample_h5_bytes(), dataset_path="/counts")
    )
    assert result.error.code == ""
    assert result.dtype == "int32"
    assert list(result.shape) == [100]
    assert list(result.chunks) == [10]
    assert result.compression == "gzip"
    assert result.compression_opts == 4
    assert result.shuffle is True
    assert result.nbytes == 100 * 4


def test_read_dataset_info_golden_path_without_leading_slash():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=sample_h5_bytes(), dataset_path="counts")
    )
    assert result.error.code == ""
    assert result.dtype == "int32"


def test_read_dataset_info_error_path_not_found():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=sample_h5_bytes(), dataset_path="/does/not/exist")
    )
    assert result.error.code == "NOT_FOUND"


def test_read_dataset_info_error_path_names_a_group():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=sample_h5_bytes(), dataset_path="/measurements")
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_read_dataset_info_error_path_missing_dataset_path():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=sample_h5_bytes(), dataset_path="")
    )
    assert result.error.code == "INVALID_ARGUMENT"


def test_read_dataset_info_error_path_empty_input():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=b"", dataset_path="/counts")
    )
    assert result.error.code == "INVALID_INPUT"


def test_read_dataset_info_error_path_malformed_input():
    ax = FakeAxiomContext()
    result = read_dataset_info(
        ax, ReadDatasetInfoRequest(data=b"garbage", dataset_path="/counts")
    )
    assert result.error.code == "INVALID_INPUT"
