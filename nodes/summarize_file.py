import h5py

from gen.messages_pb2 import SummarizeFileRequest, SummarizeFileResult
from gen.axiom_context import AxiomContext
from nodes._helpers import invalid_input, open_h5_temp


def summarize_file(ax: AxiomContext, input: SummarizeFileRequest) -> SummarizeFileResult:
    """Report a quick, cheap top-level layout summary of an HDF5 file: total
    group and dataset counts anywhere in the tree, the deepest path depth,
    the immediate children of the root group, the number of attributes
    attached to the root group, the library's reported format-
    compatibility bounds, and the supplied file's byte size. Useful before
    paying for a full ListHierarchy walk on a large or deep file.
    Malformed input returns a structured error rather than crashing.
    """
    try:
        with open_h5_temp(input.data) as h5f:
            num_groups = 1  # root group
            num_datasets = 0
            max_depth = 0

            def visitor(name, obj):
                nonlocal num_groups, num_datasets, max_depth
                depth = name.count("/") + 1
                max_depth = max(max_depth, depth)
                if isinstance(obj, h5py.Dataset):
                    num_datasets += 1
                elif isinstance(obj, h5py.Group):
                    num_groups += 1

            h5f.visititems(visitor)

            low, high = h5f.libver
            return SummarizeFileResult(
                num_groups=num_groups,
                num_datasets=num_datasets,
                max_depth=max_depth,
                root_num_attrs=len(h5f.attrs),
                top_level_names=list(h5f.keys()),
                libver_bounds=f"{low}/{high}",
                file_size_bytes=len(input.data),
            )
    except OSError as e:
        return SummarizeFileResult(error=invalid_input(f"not a valid HDF5 file: {e}"))
    except Exception as e:
        # h5py can raise other exception types while lazily walking a
        # corrupt tree (RuntimeError from libhdf5, KeyError on a broken
        # link, ...); treat all of them as a structured INVALID_INPUT
        # rather than letting an unexpected type escape as a crash.
        return SummarizeFileResult(error=invalid_input(f"could not read file structure: {e}"))
