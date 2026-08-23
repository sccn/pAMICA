# CI/CD Workflow Standards

## Purpose: Automated Quality Gates
**Why CI/CD?** Catch issues before users do.
**Think:** Every pipeline failure is a production bug prevented.
**Goal:** Fast feedback, high confidence, zero surprises.

## pAMICA CI Topology (actual, issue #246)

The generic template below stays as guidance for other projects; this project's
real workflows live in `.github/workflows/` and diverge from it in ways worth
recording:

- **`ci.yml`** -- `pull_request` (any branch) plus `push` to `main` (release
  branch) and `dev` (integration/default branch), so post-merge drift on `dev`
  is caught, not just PR-time state (concurrency cancels a superseded run on
  the same ref). Jobs: `lint` (ruff) -> `typecheck` (ty) -> `test` (Linux,
  `-m "not slow"`, `--cov-fail-under=80`, builds the dependency-free native
  AMICA binary for engine E2E tests) -> `test-macos` (Apple Silicon, `--extra
  mlx --extra mne`, asserts a real MLX GPU device, builds the same native
  binary via Accelerate, `-m "not slow"`, `--no-cov`) -> `test-mne` (Linux,
  `--extra mne`) -> `build` (sdist/wheel import matrix, Python 3.12/3.13).
  `typecheck`/`lint` gate every test job via `needs:`.
- **`weekly-macos-slow.yml`** -- schedule-only (Sunday cron) plus manual
  `workflow_dispatch`, never on `push`/`pull_request`, so it cannot block or
  slow a PR. Runs the full suite with no `-m` filter on macOS (Apple Silicon):
  `@pytest.mark.slow` tests, plus the Fortran-parity tests gated on
  `PAMICA_NATIVE_BINARY` (native `native/build.sh` shim, arm64) and
  `AMICA_RUN_FORTRAN=1`, plus `test_fortran_adapter.py`'s
  `AMICA_FORTRAN_BIN`-gated tests (a *second*, separately built native
  binary, `benchmarks/fortran/build_amica.sh`, real mpif90 as a single-rank
  singleton). Neither native build needs Rosetta -- both compile
  `amica15.f90`/`funmod2.f90` from source for the runner's own arch; only the
  legacy bundled `pamica/sample_data/amica15mac` fixture is x86_64-only, and
  nothing in CI runs it.
- **`release-binaries.yml`**, **`publish.yml`**, **`auto-tag.yml`**,
  **`auto-bump-dev.yml`**, **`sync-dev.yml`**, **`docs.yml`**, **`typos.yml`**,
  **`draft-pdf.yml`** each own one concern (native-binary release assets, PyPI
  publish, version tagging, dev version bump, post-release dev sync, MkDocs
  deploy, spell-check, JOSS paper PDF) and trigger independently -- see each
  file's header comment for its exact trigger and rationale.

## Essential Workflows (generic template)

### 1. Testing (`test.yml`)
**Triggers:** `on: [push, pull_request]` to main branches
**Jobs (in order):**
- **Lint:** `ruff check` / `biome check` (fails fast)
- **Type Check:** `ty` / `tsc --noEmit`
- **Test:** Real tests only, matrix for versions
- **Build:** Verify compilation if applicable
- **Coverage:** Optional reporting to Codecov

### 2. Documentation (`docs.yml`)
**Triggers:** `on: push: branches: [main]`
**Jobs:** Build with MkDocs -> Deploy to GitHub Pages

### 3. Release (`release.yml`)
**Triggers:** Tag creation or manual
**Jobs:** Build -> Create release -> Publish packages

## Python Example (UV)
```yaml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
    - run: uv sync --dev
    - run: uv run ruff check .
    - run: uv run ruff format --check .

  test:
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      matrix: { python-version: ['3.12', '3.13'] }
    steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
      with: { python-version: '${{ matrix.python-version }}' }
    - run: uv sync --dev
    - run: uv run pytest --cov=src
```

## JavaScript/TypeScript Example (Bun)
```yaml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: oven-sh/setup-bun@v2
    - run: bun install
    - run: bun run biome check .

  test:
    needs: lint
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: oven-sh/setup-bun@v2
    - run: bun install
    - run: bun test
```

## Key Practices (Think About Pipeline Flow)
- **Pin versions:** `actions/checkout@v4` (reproducibility)
- **Cache deps:** UV and Bun both have built-in caching; use `setup-uv` and `setup-bun` actions
- **Fail fast:** Lint -> Type Check -> Test -> Build -> Deploy (catch cheap failures first)
- **Matrix testing:** Test all supported versions
- **Secrets:** Never commit credentials; use GitHub Secrets
- **Conditional:** Deploy only from protected branches

## Pipeline Philosophy
**Fast feedback:** Developers should know in <5 min
**Clear failures:** Error messages should guide fixes
**No surprises:** If it passes CI, it works in production

**Ask yourself:**
- Will this catch real issues?
- Is the feedback loop fast enough?
- Are we testing what actually matters?
