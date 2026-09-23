"""The first Newton iterations against the native reference binary (issue #335).

Opt-in (``AMICA_RUN_FORTRAN=1``) like the other binary-driven tests. The
binary is the pinned v0.3.3 release asset from :mod:`pamica.native.resolver`
(``PAMICA_NATIVE_BINARY`` overrides it), run single-threaded so its own
reduction order is fixed.

The binary is seeded with pamica's own initialization through its ``indir``/
``load_*`` inputs (``A`` in the reference's layout is the stored block
transposed, the issue #24 mapping), so the two sides start from the same state
and differ only in their arithmetic. Newton switches on at ``newt_start=3`` and
``doscaling`` is off, which isolates the Newton schedule from the ``doscaling``
axis fixed in issue #333. Over 6 iterations (four Newton M-steps) the
log-likelihood trajectory and the mixing matrix of both float64 backends match
the reference to round-off:

========  ======================  ======================
backend   max abs LL deviation    max abs ``A`` deviation
========  ======================  ======================
PyTorch   5.4e-11 (1.2e-3 before) 4.2e-10 (3.1e-2 before)
NumPy     2.1e-12 (1.2e-3 before) 1.1e-10 (3.1e-2 before)
========  ======================  ======================

"Before" is the same run before issue #335, when Newton switched on one
iteration late. The deviations grow with the iteration count on this
recording, Newton or not: one mixture component's shape sits at ``rho=1``,
where the location update divides by ``|y|`` and amplifies round-off by
orders of magnitude per iteration. That is why the comparison stops at the
first Newton iterations rather than running to convergence.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica.native import resolver
from pamica.native.engine import _DEFAULT_PARAMS, _render_param
from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
PINNED_BINARY = "v0.3.3"
NEWT_START = 3
N_ITER = 6
LL_ATOL = 1e-9
A_ATOL = 1e-8

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("AMICA_RUN_FORTRAN") != "1",
        reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
    ),
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
]

# One configuration for all three runs, in the torch spelling; the binary's and
# NumPy's spellings are derived from it below.
_CONFIG: dict[str, Any] = dict(
    block_size=512,
    lrate=0.05,
    lratefact=0.5,
    rholrate=0.05,
    rholratefact=0.5,
    rho0=1.5,
    minrho=1.0,
    maxrho=2.0,
    invsigmin=0.0,
    invsigmax=100.0,
    newt_start=NEWT_START,
    newtrate=1.0,
    newt_ramp=10,
    maxdecs=3,
    doscaling=False,
)


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    # The whole record: the binary NaNs on a short slice at block size 512
    # (issue #292).
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _torch(do_newton: bool = True) -> AMICATorchNG:
    return AMICATorchNG(
        n_channels=NW,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        keep_best=False,
        do_newton=do_newton,
        **_CONFIG,
    )


def _write_f64(path: Path, arr) -> None:
    np.asarray(arr, dtype="<f8").flatten(order="F").tofile(path)


def _read_f64(path: Path, shape=None) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f8")
    return arr if shape is None else arr.reshape(shape, order="F")


def _seeded_reference_run(X: np.ndarray, workdir: Path) -> dict[str, np.ndarray]:
    """Run the binary for ``N_ITER`` iterations from pamica's own seeded
    initialization; return its log-likelihood trajectory and mixing matrix."""
    init = _torch()
    init._preprocess(X)
    init._initialize_parameters()
    assert init.A is not None and init.mean is not None
    indir = workdir / "init"
    indir.mkdir()
    _write_f64(indir / "A", init.A.numpy().T)  # reference layout (issue #24)
    _write_f64(indir / "mean", init.mean.numpy().reshape(-1))
    for name, value in (
        ("mu", init.mu),
        ("sbeta", init.beta),
        ("rho", init.rho),
        ("alpha", init.alpha),
        ("gm", init.gm),
        ("c", init.c),
    ):
        assert value is not None
        _write_f64(indir / name, value.numpy())

    X.astype(np.float32).ravel(order="F").tofile(workdir / "data.fdt")
    params = dict(_DEFAULT_PARAMS)
    params.update(
        {k: v for k, v in _CONFIG.items() if k not in ("maxdecs", "doscaling")}
    )
    params.update(
        data_dim=NW,
        field_dim=X.shape[1],
        num_models=1,
        num_mix_comps=NMIX,
        max_iter=N_ITER,
        max_threads=1,
        pcakeep=NW,
        do_newton=1,
        max_decs=_CONFIG["maxdecs"],
        doscaling=int(_CONFIG["doscaling"]),
        write_LLt=0,
        writestep=10**6,
        indir=str(indir),
        load_mean=1,
        load_sphere=0,
        load_A=1,
        load_mu=1,
        load_beta=1,
        load_rho=1,
        load_alpha=1,
        load_gm=1,
        load_c=1,
    )
    (workdir / "out").mkdir()
    # ``files`` must come first: amica15.f90 stops if it parses other keys first.
    param = {"files": "./data.fdt", "outdir": "./out/", **params}
    (workdir / "input.param").write_text(_render_param(param))
    proc = subprocess.run(
        [str(resolver.resolve(PINNED_BINARY)), "input.param"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "Starting Newton" in proc.stdout, "the reference never switched Newton on"
    ll = _read_f64(workdir / "out" / "LL")
    return {"LL": ll[:N_ITER], "A": _read_f64(workdir / "out" / "A", (NW, NW))}


@pytest.fixture(scope="module")
def reference(X, tmp_path_factory) -> dict[str, np.ndarray]:
    return _seeded_reference_run(X, tmp_path_factory.mktemp("native_newton"))


def test_torch_first_newton_iterations_match_the_reference(X, reference):
    m = _torch().fit(X, max_iter=N_ITER, verbose=False)
    assert len(m.ll_history) == N_ITER and m.n_newton_fallbacks == 0
    ll_err = np.abs(np.asarray(m.ll_history) - reference["LL"]).max()
    a_err = np.abs(m.get_mixing_matrix(0) - reference["A"]).max()
    assert ll_err < LL_ATOL, f"LL deviates by {ll_err:.2e}"
    assert a_err < A_ATOL, f"A deviates by {a_err:.2e}"


def test_numpy_first_newton_iterations_match_the_reference(X, reference, tmp_path):
    init = _torch()
    init._preprocess(X)
    init._initialize_parameters()
    m = AMICA_NumPy(
        num_models=1,
        num_mix=NMIX,
        seed=SEED,
        max_iter=N_ITER,
        use_tqdm=False,
        do_opt_block=False,
        writestep=10**7,
        outdir=str(tmp_path / "out"),
        do_newton=True,
        max_decs=_CONFIG["maxdecs"],
        **{k: v for k, v in _CONFIG.items() if k != "maxdecs"},
    )
    # Start from the same initialization the binary was seeded with: the NumPy
    # backend draws its own from a different generator, and keeps any parameter
    # already set when it initializes.
    for name in ("A", "mu", "beta", "rho", "alpha", "gm", "c"):
        value = getattr(init, name)
        assert value is not None
        setattr(m, name, value.numpy().copy())
    m.fit(X)
    assert m.comp_list is not None and m.A is not None
    assert len(m.ll) == N_ITER
    ll_err = np.abs(np.asarray(m.ll) - reference["LL"]).max()
    a_err = np.abs(m.A[:, m.comp_list[:, 0]].T - reference["A"]).max()
    assert ll_err < LL_ATOL, f"LL deviates by {ll_err:.2e}"
    assert a_err < A_ATOL, f"A deviates by {a_err:.2e}"


def test_the_window_compared_is_newton_sensitive(X, reference):
    """Control: the same seeded run without Newton leaves the reference by
    orders of magnitude more than the tolerance above, from the first iteration
    a Newton step can reach (index ``NEWT_START``), so the agreement above is a
    statement about the Newton schedule rather than about a window Newton never
    touched."""
    m = _torch(do_newton=False).fit(X, max_iter=N_ITER, verbose=False)
    dev = np.abs(np.asarray(m.ll_history) - reference["LL"])
    assert dev[:NEWT_START].max() < LL_ATOL  # the shared natural-gradient prefix
    assert dev[NEWT_START] > 1e4 * LL_ATOL
