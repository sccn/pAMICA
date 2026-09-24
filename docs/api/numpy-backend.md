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

Its defaults come from the bundled `pamica/numpy_impl/params.json`, read by the constructor,
and match the PyTorch and MLX backends' ([Default settings](../guides/amica-differences.md#default-settings-issue-354)):
since issue #354 a default fit runs without Newton and stops at `max_iter=100`,
where it used to turn Newton on and run 2000 iterations.
A `params_file` replaces the bundled file, and a keyword argument overrides either.

It writes files only when given an `outdir` (keyword, params file, or the CLI's `--outdir`, default `output`):
there it writes its `out.txt` log, its `writestep` checkpoints and its final results in the Fortran `amicaout` layout.
Without one (the default) a fit writes nothing, like the PyTorch and MLX backends.

This backend implements only the generalized-Gaussian source density
(`pdftype=0`); `AMICA_NumPy(pdftype=...)` with any other value raises
`NotImplementedError` at construction instead of silently doing nothing.
It also has no best-iterate safeguard (`keep_best`), Mutual Information Reduction (MIR) diagnostic,
`variance_order` or per-sample scoring accessors, and a second `fit` on the same instance continues from the first;
see the [backend-differences table](../guides/amica-differences.md#backend-differences).

`AMICA_NumPy(**kwargs)` rejects a keyword argument it does not recognize
(issue #346), before any other construction runs.
An unrecognized name raises `TypeError`,
with a `difflib`-based "did you mean" suggestion when a close match exists.
An option implemented on the PyTorch backend (`AMICATorchNG`) but not this one
(for example `keep_best`, `device`, `dtype`, the kurtosis-switch schedule)
also raises `TypeError`, naming the option and pointing to `AMICA(backend='torch')`.
`n_models`/`n_mix` (`AMICATorchNG`'s own spelling of this backend's
`num_models`/`num_mix`) get a message naming the correct spelling instead;
`n_channels` gets its own message: this backend, like the `AMICA` wrapper,
infers it from the data passed to `fit()`.
Every offending keyword is named in one error, however many different kinds
are mixed in the same call.
The three settings this backend spells differently from `AMICATorchNG`
(`min_nd`/`maxdecs`/`share_iter`) are accepted under either spelling,
the same as a params file already resolves them;
passing both spellings of the same setting at once raises `TypeError`.

`files`, `data_dim` and `field_dim` -- data-location metadata `fit()` reads
when called with no data -- are also accepted as keyword arguments directly,
with the same meaning as in a params file, not only through `params_file=...`
(issue #346).
Setting both `params_file` and a keyword argument to a *different* value for
the same one of these three raises `TypeError`; the same value from both is
not a conflict.

::: pamica.AMICA_NumPy
