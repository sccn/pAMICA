# Testing

## Real data only

Correctness tests use real sample EEG and the reference Fortran binary shipped in
`pamica/sample_data/`. Mocks, stubs, and synthetic data are not used as the
basis for correctness tests: no test is better than a fake passing test.

## Running the suite

```bash
uv run pytest                    # full suite
uv run pytest --cov              # with coverage
uv run pytest pamica/tests/torch_tests/   # PyTorch-vs-Fortran parity tests
uv run pytest -m "not slow"      # skip the slow tests, as CI does
AMICA_RUN_FORTRAN=1 uv run pytest  # also run the tests that execute the native binary
```

Tests that run the Fortran binary, including the element-wise oracles that seed it from a pamica state
(`pamica/tests/native_oracle.py`), are opt-in with `AMICA_RUN_FORTRAN=1`;
the weekly macOS workflow runs them.

## Working directory

The suite runs from a temporary working directory, not the repository:
a session-scoped autouse fixture in `pamica/tests/conftest.py` switches to one per session (per worker under `pytest-xdist`),
so anything a test writes to a relative path lands there instead of in the repository.
A test that needs a repository path resolves it from `__file__`
(for example `Path(__file__).resolve().parent.parent / "sample_data"`),
or, when a relative path inside a file must resolve against the repository root
(such as `sample_params.json`'s `files`), changes into the root explicitly with `monkeypatch.chdir`.
Coverage reports still land in the repository root.

## Layout

- `pamica/tests/`: end-to-end and interface tests, and the cross-backend tests every shared behavior needs.
- `pamica/tests/torch_tests/`: natural-gradient backend parity, PDF families,
  component sharing, float32 stability, and edge cases.
- `pamica/tests/mlx_tests/`: MLX backend tests (Apple Silicon).
- `pamica/tests/mne_tests/`: the MNE wrapper, including the end-to-end workflow test.
- `validate_implementations.py`: cross-implementation validation harness
  (Hungarian component matching against Fortran; `--backend` selects torch,
  numpy, mlx, a list, or `all`).
