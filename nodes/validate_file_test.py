from gen.messages_pb2 import ValidateFileRequest
from nodes.validate_file import validate_file
from nodes._helpers import MAX_INPUT_BYTES
from nodes._test_fixtures import FakeAxiomContext, empty_h5_bytes, sample_h5_bytes


def test_validate_file_golden_valid():
    ax = FakeAxiomContext()
    result = validate_file(ax, ValidateFileRequest(data=sample_h5_bytes()))
    assert result.error.code == ""
    assert result.valid is True
    assert result.detail == ""
    # HDF5 reports superblock version as a small non-negative integer.
    assert result.superblock_version.isdigit()


def test_validate_file_golden_empty_file_group_only():
    ax = FakeAxiomContext()
    result = validate_file(ax, ValidateFileRequest(data=empty_h5_bytes()))
    assert result.error.code == ""
    assert result.valid is True


def test_validate_file_rejects_malformed_bytes():
    ax = FakeAxiomContext()
    result = validate_file(ax, ValidateFileRequest(data=b"this is not an HDF5 file at all"))
    assert result.error.code == ""
    assert result.valid is False
    assert result.detail != ""


def test_validate_file_error_path_empty_input():
    ax = FakeAxiomContext()
    result = validate_file(ax, ValidateFileRequest(data=b""))
    assert result.error.code == "INVALID_INPUT"


def test_validate_file_error_path_oversized_input():
    ax = FakeAxiomContext()
    oversized = b"\x89HDF\r\n\x1a\n" + b"0" * (MAX_INPUT_BYTES + 1)
    result = validate_file(ax, ValidateFileRequest(data=oversized))
    assert result.error.code == "TOO_LARGE"


def test_validate_file_is_deterministic():
    ax = FakeAxiomContext()
    data = sample_h5_bytes()
    r1 = validate_file(ax, ValidateFileRequest(data=data))
    r2 = validate_file(ax, ValidateFileRequest(data=data))
    assert r1.valid == r2.valid
    assert r1.superblock_version == r2.superblock_version
