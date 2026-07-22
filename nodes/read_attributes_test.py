import json

from gen.messages_pb2 import ReadAttributesRequest
from nodes.read_attributes import read_attributes
from nodes._helpers import MAX_ATTR_VALUE_JSON_BYTES, MAX_ATTRIBUTES
from nodes._test_fixtures import (
    FakeAxiomContext,
    h5_bytes_with_many_attributes,
    h5_bytes_with_oversized_attribute,
    sample_h5_bytes,
)


def test_read_attributes_golden_root():
    ax = FakeAxiomContext()
    result = read_attributes(ax, ReadAttributesRequest(data=sample_h5_bytes(), path="/"))
    assert result.error.code == ""
    assert result.total_attributes_available == 2
    assert result.truncated is False
    by_name = {a.name: a for a in result.attributes}
    assert json.loads(by_name["institution"].value_json) == "Axiom Labs"
    assert json.loads(by_name["version"].value_json) == 3
    assert all(not a.value_truncated for a in result.attributes)


def test_read_attributes_golden_empty_path_means_root():
    ax = FakeAxiomContext()
    result = read_attributes(ax, ReadAttributesRequest(data=sample_h5_bytes(), path=""))
    assert result.error.code == ""
    assert result.total_attributes_available == 2


def test_read_attributes_golden_group():
    ax = FakeAxiomContext()
    result = read_attributes(
        ax, ReadAttributesRequest(data=sample_h5_bytes(), path="/measurements")
    )
    assert result.error.code == ""
    assert result.total_attributes_available == 1
    assert json.loads(result.attributes[0].value_json) == "celsius"


def test_read_attributes_golden_dataset_with_no_attributes():
    ax = FakeAxiomContext()
    result = read_attributes(
        ax, ReadAttributesRequest(data=sample_h5_bytes(), path="/measurements/temp")
    )
    assert result.error.code == ""
    assert result.total_attributes_available == 0
    assert result.attributes == []


def test_read_attributes_error_path_not_found():
    ax = FakeAxiomContext()
    result = read_attributes(
        ax, ReadAttributesRequest(data=sample_h5_bytes(), path="/nope")
    )
    assert result.error.code == "NOT_FOUND"


def test_read_attributes_caps_count_and_reports_truncation():
    ax = FakeAxiomContext()
    total = MAX_ATTRIBUTES + 25
    result = read_attributes(
        ax, ReadAttributesRequest(data=h5_bytes_with_many_attributes(total), path="/")
    )
    assert result.error.code == ""
    assert result.total_attributes_available == total
    assert len(result.attributes) == MAX_ATTRIBUTES
    assert result.truncated is True


def test_read_attributes_truncates_oversized_value():
    ax = FakeAxiomContext()
    data = h5_bytes_with_oversized_attribute(MAX_ATTR_VALUE_JSON_BYTES * 3)
    result = read_attributes(ax, ReadAttributesRequest(data=data, path="/x"))
    assert result.error.code == ""
    assert len(result.attributes) == 1
    assert result.attributes[0].value_truncated is True
    assert len(result.attributes[0].value_json.encode("utf-8")) < MAX_ATTR_VALUE_JSON_BYTES


def test_read_attributes_error_path_empty_input():
    ax = FakeAxiomContext()
    result = read_attributes(ax, ReadAttributesRequest(data=b"", path="/"))
    assert result.error.code == "INVALID_INPUT"


def test_read_attributes_error_path_malformed_input():
    ax = FakeAxiomContext()
    result = read_attributes(ax, ReadAttributesRequest(data=b"garbage", path="/"))
    assert result.error.code == "INVALID_INPUT"
