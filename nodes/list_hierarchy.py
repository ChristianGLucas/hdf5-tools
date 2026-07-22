import h5py

from gen.messages_pb2 import HierarchyEntry, ListHierarchyRequest, ListHierarchyResult
from gen.axiom_context import AxiomContext
from nodes._helpers import (
    DEFAULT_HIERARCHY_ENTRIES,
    MAX_HIERARCHY_ENTRIES,
    check_input_size,
    invalid_input,
    open_h5_temp,
)


def list_hierarchy(ax: AxiomContext, input: ListHierarchyRequest) -> ListHierarchyResult:
    """Walk an HDF5 file's full group/dataset tree in one depth-first pass
    and report every node found: its path, whether it is a group or a
    dataset, and — for datasets — dtype, shape, and attribute count. The
    root group is always included first. Capped at max_entries (default
    500, hard cap 2000) with a truncated flag and the true total entry
    count when the file has more nodes than were returned. Malformed
    input returns a structured error rather than crashing.
    """
    size_err = check_input_size(input.data)
    if size_err is not None:
        return ListHierarchyResult(error=size_err)

    limit = input.max_entries if input.max_entries > 0 else DEFAULT_HIERARCHY_ENTRIES
    limit = min(limit, MAX_HIERARCHY_ENTRIES)

    try:
        with open_h5_temp(input.data) as h5f:
            all_paths = ["/"]
            h5f.visititems(lambda name, obj: all_paths.append("/" + name))
            total = len(all_paths)

            entries = []
            for path in all_paths[:limit]:
                obj = h5f["/"] if path == "/" else h5f[path[1:]]
                if isinstance(obj, h5py.Dataset):
                    entries.append(
                        HierarchyEntry(
                            path=path,
                            kind="dataset",
                            dtype=str(obj.dtype),
                            shape=list(obj.shape),
                            num_attrs=len(obj.attrs),
                        )
                    )
                else:
                    entries.append(
                        HierarchyEntry(
                            path=path,
                            kind="group",
                            num_attrs=len(obj.attrs),
                        )
                    )

            return ListHierarchyResult(
                entries=entries,
                total_entries_available=total,
                truncated=len(entries) < total,
            )
    except OSError as e:
        return ListHierarchyResult(error=invalid_input(f"not a valid HDF5 file: {e}"))
    except Exception as e:
        return ListHierarchyResult(
            error=invalid_input(f"could not read file structure: {e}")
        )
