from gen.messages_pb2 import AttributeEntry, ReadAttributesRequest, ReadAttributesResult
from gen.axiom_context import AxiomContext
from nodes._helpers import (
    NotFoundError,
    attr_value_to_json,
    invalid_input,
    not_found,
    open_h5_temp,
    resolve_object,
    to_json_text,
)


def read_attributes(ax: AxiomContext, input: ReadAttributesRequest) -> ReadAttributesResult:
    """Read every attribute attached at one path in an HDF5 file — the file
    root ("/" or ""), any group, or any dataset — as name/dtype/JSON-
    encoded-value triples. A path that does not exist in the file returns a
    structured NOT_FOUND error rather than crashing.
    """
    try:
        with open_h5_temp(input.data) as h5f:
            try:
                obj = resolve_object(h5f, input.path)
            except NotFoundError as e:
                return ReadAttributesResult(error=not_found(str(e)))

            names = list(obj.attrs.keys())
            total = len(names)

            entries = []
            for name in names:
                try:
                    raw = obj.attrs[name]
                except Exception as e:
                    # A malformed/unsupported attribute encoding (e.g. an
                    # exotic committed datatype) should not sink the whole
                    # response — report it as an unreadable value instead.
                    entries.append(
                        AttributeEntry(
                            name=name,
                            dtype="",
                            value_json="null",
                            value_truncated=True,
                        )
                    )
                    continue

                dtype = getattr(raw, "dtype", None)
                json_val = attr_value_to_json(raw)
                text = to_json_text(json_val)

                entries.append(
                    AttributeEntry(
                        name=name,
                        dtype=str(dtype) if dtype is not None else "",
                        value_json=text,
                        value_truncated=False,
                    )
                )

            return ReadAttributesResult(
                attributes=entries,
                total_attributes_available=total,
                truncated=len(entries) < total,
            )
    except OSError as e:
        return ReadAttributesResult(error=invalid_input(f"not a valid HDF5 file: {e}"))
    except Exception as e:
        return ReadAttributesResult(
            error=invalid_input(f"could not read attributes: {e}")
        )
