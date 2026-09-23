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
```

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

- `pamica/tests/` — end-to-end and interface tests.
- `pamica/tests/torch_tests/` — natural-gradient backend parity, PDF families,
  component sharing, float32 stability, and edge cases.
- `pamica/tests/mlx_tests/` — MLX backend tests (Apple Silicon).
- `validate_implementations.py` — cross-implementation validation harness
  (Hungarian component matching against Fortran; `--backend` selects torch,
  numpy, mlx, a list, or `all`).
