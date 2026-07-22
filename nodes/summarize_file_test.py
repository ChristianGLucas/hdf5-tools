from gen.messages_pb2 import SummarizeFileRequest
from nodes.summarize_file import summarize_file
from nodes._test_fixtures import (
    FakeAxiomContext,
    corrupted_mid_file_h5_bytes,
    empty_h5_bytes,
    sample_h5_bytes,
)


def test_summarize_file_golden():
    ax = FakeAxiomContext()
    data = sample_h5_bytes()
    result = summarize_file(ax, SummarizeFileRequest(data=data))
    assert result.error.code == ""
    # Hand-known layout (see _test_fixtures.sample_h5_bytes docstring):
    # root + "measurements" = 2 groups; "temp", "counts", "names" = 3
    # datasets; deepest path is /measurements/temp = depth 2.
    assert result.num_groups == 2
    assert result.num_datasets == 3
    assert result.max_depth == 2
    assert result.root_num_attrs == 2  # institution, version
    assert set(result.top_level_names) == {"measurements", "counts", "names"}
    assert result.file_size_bytes == len(data)
    assert "/" in result.libver_bounds


def test_summarize_file_golden_empty_file():
    ax = FakeAxiomContext()
    result = summarize_file(ax, SummarizeFileRequest(data=empty_h5_bytes()))
    assert result.error.code == ""
    assert result.num_groups == 1
    assert result.num_datasets == 0
    assert result.max_depth == 0
    assert result.top_level_names == []


def test_summarize_file_error_path_empty_input():
    ax = FakeAxiomContext()
    result = summarize_file(ax, SummarizeFileRequest(data=b""))
    assert result.error.code == "INVALID_INPUT"


def test_summarize_file_error_path_malformed_input():
    ax = FakeAxiomContext()
    result = summarize_file(ax, SummarizeFileRequest(data=b"definitely not hdf5"))
    assert result.error.code == "INVALID_INPUT"


def test_summarize_file_error_path_late_exception_during_traversal():
    """Regression test for the late-exception traceback-leak bug class: a
    file that OPENS fine but raises RuntimeError mid-traversal must come
    back as a structured error, not an unhandled crash."""
    ax = FakeAxiomContext()
    result = summarize_file(ax, SummarizeFileRequest(data=corrupted_mid_file_h5_bytes()))
    assert result.error.code == "INVALID_INPUT"
    assert result.error.message != ""


def test_summarize_file_is_deterministic():
    ax = FakeAxiomContext()
    data = sample_h5_bytes()
    r1 = summarize_file(ax, SummarizeFileRequest(data=data))
    r2 = summarize_file(ax, SummarizeFileRequest(data=data))
    assert r1.num_groups == r2.num_groups
    assert r1.num_datasets == r2.num_datasets
    assert list(r1.top_level_names) == list(r2.top_level_names)
