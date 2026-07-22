import h5py

from gen.messages_pb2 import ValidateFileRequest, ValidateFileResult
from gen.axiom_context import AxiomContext
from nodes._helpers import check_input_size, open_h5_temp


def validate_file(ax: AxiomContext, input: ValidateFileRequest) -> ValidateFileResult:
    """Cheaply check whether bytes are a structurally well-formed HDF5 file
    (the HDF5 library can open it) within the package's documented size
    cap, without walking the tree or reading any data. Returns valid=false
    with a human-readable detail for anything unparseable — never a crash
    or raised error; a hard input-level failure (e.g. the raw payload
    itself is oversized) is reported via the separate structured error
    field instead.
    """
    size_err = check_input_size(input.data)
    if size_err is not None:
        return ValidateFileResult(error=size_err)

    try:
        with open_h5_temp(input.data) as h5f:
            # get_version() returns (superblock, freelist, symbol_table,
            # shared_header_message) format versions; we report only the
            # superblock's, which is the file-format-compatibility number
            # HDF5 tooling actually calls "the superblock version".
            superblock_version = str(h5f.id.get_create_plist().get_version()[0])
            return ValidateFileResult(
                valid=True,
                superblock_version=superblock_version,
            )
    except OSError as e:
        return ValidateFileResult(valid=False, detail=f"not a valid HDF5 file: {e}")
    except Exception as e:
        # h5py/libhdf5 can raise a variety of exception types (not just
        # OSError) on truncated/corrupt input encountered lazily while
        # parsing the superblock — treat all of them as "not valid" rather
        # than letting an unexpected type escape as a crash.
        return ValidateFileResult(valid=False, detail=f"not a valid HDF5 file: {e}")
