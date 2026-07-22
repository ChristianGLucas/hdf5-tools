# christiangeorgelucas/hdf5-tools

Composable [Axiom](https://axiom.co) nodes for deterministic, offline inspection and
bounded extraction of HDF5 (`.h5`/`.hdf5`) scientific/hierarchical data files, wrapping
[h5py](https://www.h5py.org/) (BSD-3-Clause) — the reference Python binding to the HDF5
C library (itself BSD-style-licensed), which owns the canonical HDF5 reader.

Built for the Axiom marketplace. Every node is a pure bytes-in/struct-out transform: the
caller-supplied file bytes are written to a private temp file for the call and removed
before returning (h5py/libhdf5 need a real file path) — no persistent state, network, or
secrets.

## Nodes

- **ValidateFile** — a cheap structural well-formedness check (can the HDF5 library open
  these bytes at all), without walking the tree or reading any data.
- **SummarizeFile** — a quick top-level layout summary: group/dataset counts, tree depth,
  root attributes, and the library's format-compatibility bounds.
- **ListHierarchy** — walk the full group/dataset tree in one pass: path, kind, and (for
  datasets) dtype/shape/attribute-count.
- **ReadDatasetInfo** — one dataset's storage layout without reading its data: dtype,
  shape, maxshape, chunking, compression + opts, shuffle/fletcher32, fill value, and
  logical byte size.
- **ReadAttributes** — every attribute attached at a file/group/dataset path, as
  name/dtype/JSON-encoded-value triples.
- **ReadSlice** — extract a rectangular slice of one dataset as JSON (any rank)
  or CSV (1-D/2-D).

## Design

Every node is a pure input-to-output transform: it validates domain correctness (is
this actually a well-formed HDF5 file? does this path/argument make sense against the
file's real structure?) and returns a structured error rather than crashing on anything
malformed. No node imposes its own payload-size, element-count, or other resource bound
— sizing and resource containment (including guarding against a small file that decodes
to an enormous array) are the Axiom platform's job, not the package's.

## License

MIT (see `LICENSE`).
