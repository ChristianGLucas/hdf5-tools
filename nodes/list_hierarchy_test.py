from gen.messages_pb2 import ListHierarchyRequest
from nodes.list_hierarchy import list_hierarchy
from nodes._test_fixtures import (
    FakeAxiomContext,
    corrupted_mid_file_h5_bytes,
    sample_h5_bytes,
)


def test_list_hierarchy_golden():
    ax = FakeAxiomContext()
    result = list_hierarchy(ax, ListHierarchyRequest(data=sample_h5_bytes()))
    assert result.error.code == ""
    assert result.truncated is False
    assert result.total_entries_available == 5  # root, measurements, temp, counts, names
    by_path = {e.path: e for e in result.entries}
    assert by_path["/"].kind == "group"
    assert by_path["/"].num_attrs == 2
    assert by_path["/measurements"].kind == "group"
    assert by_path["/measurements"].num_attrs == 1
    assert by_path["/measurements/temp"].kind == "dataset"
    assert by_path["/measurements/temp"].dtype == "float64"
    assert list(by_path["/measurements/temp"].shape) == [2, 3]
    assert by_path["/counts"].kind == "dataset"
    assert by_path["/counts"].dtype == "int32"
    assert list(by_path["/counts"].shape) == [100]
    assert by_path["/names"].kind == "dataset"
    assert list(by_path["/names"].shape) == [3]


def test_list_hierarchy_respects_max_entries_and_reports_truncation():
    ax = FakeAxiomContext()
    result = list_hierarchy(
        ax, ListHierarchyRequest(data=sample_h5_bytes(), max_entries=2)
    )
    assert result.error.code == ""
    assert len(result.entries) == 2
    assert result.total_entries_available == 5
    assert result.truncated is True


def test_list_hierarchy_error_path_empty_input():
    ax = FakeAxiomContext()
    result = list_hierarchy(ax, ListHierarchyRequest(data=b""))
    assert result.error.code == "INVALID_INPUT"


def test_list_hierarchy_error_path_malformed_input():
    ax = FakeAxiomContext()
    result = list_hierarchy(ax, ListHierarchyRequest(data=b"nope"))
    assert result.error.code == "INVALID_INPUT"
    assert result.entries == []


def test_list_hierarchy_error_path_late_exception_during_traversal():
    ax = FakeAxiomContext()
    result = list_hierarchy(
        ax, ListHierarchyRequest(data=corrupted_mid_file_h5_bytes())
    )
    assert result.error.code == "INVALID_INPUT"


def test_list_hierarchy_is_deterministic():
    ax = FakeAxiomContext()
    data = sample_h5_bytes()
    r1 = list_hierarchy(ax, ListHierarchyRequest(data=data))
    r2 = list_hierarchy(ax, ListHierarchyRequest(data=data))
    assert [e.path for e in r1.entries] == [e.path for e in r2.entries]
