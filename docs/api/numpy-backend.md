# NumPy backend (AMICA_NumPy)

The legacy NumPy reference implementation, retained as an oracle and for its
command-line interface. It carries the same parity fixes as the PyTorch backend,
plus baralpha and outlier rejection (`do_reject`), which mirrors the PyTorch
backend's `good_idx` sample-dropping mechanism (issue #123).

`AMICA_NumPy(params_file=...)` and the `AMICA_NumPy.from_params_file` classmethod
(renamed from `from_json_file`, issue #304) accept both pamica's JSON schema and
the literal Fortran `input.param` text format, content-sniffed the same way as the
PyTorch wrapper's `AMICA.from_params_file` -- see
[Parameter files](../guides/validation.md#parameter-files).
The NumPy CLI (`python -m pamica.numpy_impl.cli`) accepts both formats too, through
the same shared reader.
Settings the file carries that this backend does not consume are named in one
`logger.warning` rather than silently dropped.

This backend implements only the generalized-Gaussian source density
(`pdftype=0`); `AMICA_NumPy(pdftype=...)` with any other value raises
`NotImplementedError` at construction instead of silently doing nothing -- see
the backend-differences table in
[amica-differences.md](../guides/amica-differences.md).

::: pamica.AMICA_NumPy
