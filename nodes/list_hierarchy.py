import h5py

from gen.messages_pb2 import HierarchyEntry, ListHierarchyRequest, ListHierarchyResult
from gen.axiom_context import AxiomContext
from nodes._helpers import invalid_input, open_h5_temp


def list_hierarchy(ax: AxiomContext, input: ListHierarchyRequest) -> ListHierarchyResult:
    """Walk an HDF5 file's full group/dataset tree in one depth-first pass
    and report every node found: its path, whether it is a group or a
    dataset, and — for datasets — dtype, shape, and attribute count. The
    root group is always included first. Returns every entry unless the
    caller passes max_entries > 0, in which case the walk is truncated to
    that many entries (with a truncated flag and the true total entry
    count). Malformed input returns a structured error rather than
    crashing.
    """
    limit = input.max_entries if input.max_entries > 0 else None

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
