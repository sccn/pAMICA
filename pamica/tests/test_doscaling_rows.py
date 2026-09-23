"""``doscaling`` rescales components, not stored columns (issue #333, epic #324
Phase 7).

pamica stores each model's mixing block transposed relative to the reference
(issue #24 convention, ADR 0006), so source ``i`` of model ``h`` is a ROW of its
block, with its density at ``mu[:, comp_list[i, h]]``; since issue #334 (ADR
0007) ``A`` holds one component per row, ``A[comp_list[i, h], :]``. The
reference's ``doscaling`` (amica15.f90:1843-1851) divides each component's
mixing vector by its norm and multiplies/divides that component's ``mu``/``sbeta``
by it: an exact change of scale that leaves the log-likelihood unchanged. Before
this fix every backend normalized stored COLUMNS instead, which is not a change
of scale of any component and moved every default fit off the reference's
trajectory from the first iteration.

Pinned here, cross-backend per ``.rules/backend_parity.md`` (PyTorch and NumPy
always run; MLX checks skip individually without MLX or an Apple GPU):

1. one doscaling pass, through each backend's own ``_rescale_components``, is an
   exact change of scale on natural-gradient and Newton states alike: the
   log-likelihood moves by float round-off only and every component row ends
   at unit norm, while the old column rule on the same state moves the
   log-likelihood by far more (so these tests catch the bug);
2. the three backends rescale one shared state identically, including a
   permuted ``comp_list`` and a merged one, where the shared component row is
   rescaled once (issue #334); a zero- or NaN-norm row is left untouched, as in
   the reference;
3. ``doscaling=False`` is byte-identical to the pre-fix code: the package at
   commit ``fb13d76`` is loaded from git and each backend fitted in the same
   process, its ``A`` mapped onto the component rows issue #334 introduced;
4. ``scalestep``, a pamica extension the reference parses but never reads,
   counts iterations from 1, so its default of 1 is the reference's
   every-iteration rescale, and every constructor rejects a ``scalestep`` that
   is not an integer >= 1 while the rescale is on;
5. (opt-in, ``AMICA_RUN_FORTRAN=1``) the native reference binary, seeded from
   pamica's initialization, matches pamica's ``A``/``mu``/``sbeta`` after one
   and three iterations to float64 round-off.

Real bundled sample EEG only, no synthetic data or mocks (``.rules/testing.md``).
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from pamica import AMICA_NumPy
from pamica.component_layout import rows_from_legacy_columns
from pamica.tests.pre_change import load_pre_change_package
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
# The fitting slice for the always-on tests: long enough for real dynamics,
# short enough that a few iterations on every backend stay fast.
N_SLICE = 8192

# A float64 log-likelihood (per sample and channel, about -3.4) changed by a
# pure rescale moves by round-off only: measured 0 to 4.4e-16 (1 ULP) on the
# sample. The float32 MLX likelihood moves by float32 round-off: measured
# 3.5e-9 to 6e-8 (a float32 ULP at 3.4 is 2.4e-7).
LL_TOL_F64 = 1e-13
LL_TOL_F32 = 1e-6
# The old stored-column rule on the same 3-iteration states moved the
# log-likelihood by 1.0e-4 to 1.1e-4 (natural gradient) and 2.1e-4 to 2.2e-4
# (with Newton steps), a compensated rescale of the wrong vectors: at least ten
# times this bound and a hundred times the float32 tolerance.
COLUMN_RULE_MIN_DLL = 1e-5

# A real state reached WITH Newton steps: Newton from the second iteration
# (0-based iteration 1). A positive-definite Newton step ramps lrate past the
# natural-gradient ceiling (the lrate default, 0.1), which the setup asserts;
# measured on every backend, one and two models: 0.3 after 3 iterations.
NEWTON: Dict[str, Any] = dict(do_newton=True, newt_start=1)
NG_LRATE_CEILING = 0.1

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_slice() -> np.ndarray:
    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return X.astype(np.float64)[:, :N_SLICE]


def _mlx_core():
    """The MLX backend module, or skip this test (never the whole module)."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core


def _row_norm_dev(A: np.ndarray, comp_list: np.ndarray) -> float:
    """Largest ``| ||A[k, :]|| - 1 |`` over every component ``k`` a model uses
    (one component per row, issue #334)."""
    norms = np.linalg.norm(A[np.unique(comp_list), :], axis=1)
    return float(np.abs(norms - 1.0).max())


def _column_rule(
    A: np.ndarray, mu: np.ndarray, beta: np.ndarray, comp_list: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pre-#333 rule: normalize every stored column of the pre-#334
    component-column layout, scaling ``mu``/``beta`` of the component id that
    indexes the column to match -- a compensated rescale of vectors that are not
    components. Takes and returns component-row arrays (an unmerged
    ``comp_list``); the column layout is rebuilt block by block and converted
    back with the persistence conversion."""
    A_cols = np.empty((A.shape[1], A.shape[0]), dtype=A.dtype)
    for h in range(comp_list.shape[1]):
        A_cols[:, comp_list[:, h]] = A[comp_list[:, h], :]
    scale = np.sqrt(np.sum(A_cols**2, axis=0))
    A_rows = rows_from_legacy_columns(A_cols / scale, comp_list, owner="test")
    return A_rows, mu * scale, beta / scale


def _mean_ll(lht: np.ndarray, n_channels: int) -> float:
    """Per-sample-per-channel log-likelihood from ``Lht`` (models x samples)."""
    lht = np.asarray(lht, dtype=np.float64)
    return float(logsumexp(lht, axis=0).sum() / (lht.shape[1] * n_channels))


# --- 1. one pass is an exact change of scale, per backend -------------------
# Each state is a real short unscaled fit, natural gradient only or with Newton
# steps (the Newton update moves A differently, so its rows are another test).
newton_param = pytest.mark.parametrize("newton", [False, True], ids=["ng", "newton"])


@newton_param
@pytest.mark.parametrize("n_models", [1, 2])
def test_torch_rescale_is_an_exact_change_of_scale(real_slice, n_models, newton):
    m = AMICATorchNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        doscaling=False,
        keep_best=False,
        **(NEWTON if newton else {}),
    )
    m.fit(real_slice, max_iter=3, verbose=False)
    assert (m.lrate > NG_LRATE_CEILING) == newton, "test setup: Newton steps"
    assert m.A is not None and m.mu is not None and m.beta is not None
    assert m.comp_list is not None
    comp_list = m.comp_list.numpy()
    saved = {k: getattr(m, k).clone() for k in ("A", "mu", "beta")}
    ll0 = _mean_ll(m.model_loglik(real_slice), NW)
    # Unscaled rows drift well away from unit norm: the pass has work to do.
    assert _row_norm_dev(m.A.numpy(), comp_list) > 1e-2

    m._rescale_components()
    m._update_unmixing_matrices()
    assert abs(_mean_ll(m.model_loglik(real_slice), NW) - ll0) <= LL_TOL_F64
    assert _row_norm_dev(m.A.numpy(), comp_list) <= 1e-14

    # The pre-#333 rule (normalize stored columns) on the same state.
    A0, mu0, beta0 = (saved[k].numpy() for k in ("A", "mu", "beta"))
    cols = _column_rule(A0, mu0, beta0, comp_list)
    m.A, m.mu, m.beta = (torch.from_numpy(x) for x in cols)
    m._update_unmixing_matrices()
    assert abs(_mean_ll(m.model_loglik(real_slice), NW) - ll0) > COLUMN_RULE_MIN_DLL


@newton_param
@pytest.mark.parametrize("n_models", [1, 2])
def test_numpy_rescale_is_an_exact_change_of_scale(
    real_slice, n_models, newton, tmp_path
):
    m = AMICA_NumPy(
        num_models=n_models,
        num_mix=NMIX,
        max_iter=3,
        seed=SEED,
        use_tqdm=False,
        outdir=str(tmp_path),
        do_opt_block=False,
        block_size=8192,
        doscaling=False,
        writestep=10000,
        **(NEWTON if newton else {}),
    )
    m.fit(real_slice)
    assert (m.lrate > NG_LRATE_CEILING) == newton, "test setup: Newton steps"
    assert m.A is not None and m.mu is not None and m.beta is not None
    assert m.comp_list is not None
    comp_list = m.comp_list
    A0, mu0, beta0 = m.A.copy(), m.mu.copy(), m.beta.copy()
    ll0 = m._get_updates_and_likelihood()["ll"]
    assert _row_norm_dev(A0, comp_list) > 1e-2

    m._rescale_components()
    m._update_unmixing_matrices()
    assert abs(m._get_updates_and_likelihood()["ll"] - ll0) <= LL_TOL_F64
    assert m.A is not None
    assert _row_norm_dev(m.A, comp_list) <= 1e-14

    m.A, m.mu, m.beta = _column_rule(A0, mu0, beta0, comp_list)
    m._update_unmixing_matrices()
    assert abs(m._get_updates_and_likelihood()["ll"] - ll0) > COLUMN_RULE_MIN_DLL


@newton_param
@pytest.mark.parametrize("n_models", [1, 2])
def test_mlx_rescale_is_an_exact_change_of_scale(real_slice, n_models, newton):
    mlx_core = _mlx_core()
    mx = mlx_core.mx
    m = mlx_core.AMICAMLXNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        doscaling=False,
        keep_best=False,
        **(NEWTON if newton else {}),
    )
    m.fit(real_slice, max_iter=3, verbose=False)
    assert (m.lrate > NG_LRATE_CEILING) == newton, "test setup: Newton steps"
    comp_list = np.array(m.comp_list)
    saved = {k: getattr(m, k) for k in ("A", "mu", "beta")}
    ll0 = _mean_ll(m.model_loglik(real_slice), NW)
    assert _row_norm_dev(np.array(m.A, dtype=np.float64), comp_list) > 1e-2

    m._rescale_components()
    m._update_unmixing_matrices()
    assert abs(_mean_ll(m.model_loglik(real_slice), NW) - ll0) <= LL_TOL_F32
    assert _row_norm_dev(np.array(m.A, dtype=np.float64), comp_list) <= 1e-6

    # MLX rebinds (never mutates) these arrays, so the saved ones are intact.
    A0, mu0, beta0 = (np.array(saved[k]) for k in ("A", "mu", "beta"))
    cols = _column_rule(A0, mu0, beta0, comp_list)
    m.A, m.mu, m.beta = (mx.array(x) for x in cols)
    m._update_unmixing_matrices()
    assert abs(_mean_ll(m.model_loglik(real_slice), NW) - ll0) > COLUMN_RULE_MIN_DLL


# --- 2. one state, three backends, identical rescale ------------------------
State = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture(scope="module")
def unscaled_state(real_slice) -> State:
    """A real two-model state: ``A``, ``mu``, ``beta`` and ``comp_list`` of a
    short unscaled fit, before any rescale."""
    t = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        doscaling=False,
        keep_best=False,
    )
    t.fit(real_slice, max_iter=3, verbose=False)
    assert t.A is not None and t.mu is not None and t.beta is not None
    assert t.comp_list is not None
    return (
        t.A.numpy().copy(),
        t.mu.numpy().copy(),
        t.beta.numpy().copy(),
        t.comp_list.numpy().copy(),
    )


@pytest.fixture(
    scope="module",
    params=["disjoint", "permuted", "merged"],
)
def two_model_state(request, unscaled_state) -> State:
    """``permuted`` shuffles each model's ``comp_list`` column, so the rows of
    a block are neither contiguous nor in order (a general index pattern for
    every backend's gather and scale). ``merged`` plants a share merge (model
    1's source 3 folded onto model 0's source 5 component, the way
    ``identify_shared_comps`` folds a pair): the shared component row then sits
    in both blocks and must be rescaled exactly once, and the merged-away row
    not at all (issue #334).
    """
    A, mu, beta, comp_list = unscaled_state
    comp_list = comp_list.copy()
    if request.param == "permuted":
        rng = np.random.default_rng(0)
        for h in range(comp_list.shape[1]):
            comp_list[:, h] = rng.permutation(comp_list[:, h])
    elif request.param == "merged":
        comp_list[3, 1] = comp_list[5, 0]
    return A, mu, beta, comp_list


def _rescaled(backend: str, state: State) -> State:
    """``backend``'s own ``_rescale_components`` applied to a copy of
    ``state`` (cast to float32 for MLX), as numpy arrays."""
    A, mu, beta, comp_list = state
    if backend == "torch":
        t = AMICATorchNG(
            n_channels=NW, n_models=2, n_mix=NMIX, device="cpu", dtype=torch.float64
        )
        t.A, t.mu, t.beta = (
            torch.from_numpy(np.array(x, np.float64)) for x in (A, mu, beta)
        )
        t.comp_list = torch.from_numpy(comp_list.copy())
        t._rescale_components()
        assert t.A is not None and t.mu is not None and t.beta is not None
        return t.A.numpy(), t.mu.numpy(), t.beta.numpy(), comp_list
    if backend == "numpy":
        n = AMICA_NumPy(num_models=2, num_mix=NMIX, use_tqdm=False)
        n.A, n.mu, n.beta = (np.array(x, np.float64) for x in (A, mu, beta))
        n.comp_list = comp_list.copy()
        n._rescale_components()
        assert n.A is not None and n.mu is not None and n.beta is not None
        return n.A, n.mu, n.beta, comp_list
    mlx_core = _mlx_core()
    mx = mlx_core.mx
    x = mlx_core.AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX)
    x.A, x.mu, x.beta = (mx.array(np.asarray(a, np.float32)) for a in (A, mu, beta))
    x.comp_list = mx.array(comp_list)
    x._rescale_components()
    return np.array(x.A), np.array(x.mu), np.array(x.beta), comp_list


def test_numpy_rescales_like_torch(two_model_state):
    # Same float64 arithmetic; only the sum-of-squares association may differ.
    got = _rescaled("numpy", two_model_state)
    want = _rescaled("torch", two_model_state)
    for g, w in zip(got[:3], want[:3]):
        np.testing.assert_allclose(g, w, rtol=1e-14, atol=0)


def test_mlx_rescales_like_torch(two_model_state):
    got = _rescaled("mlx", two_model_state)
    # The float64 pass on the same float32-cast state is MLX's reference.
    A, mu, beta, comp_list = two_model_state
    f32 = tuple(a.astype(np.float32) for a in (A, mu, beta))
    want = _rescaled("torch", (*f32, comp_list))
    for g, w in zip(got[:3], want[:3]):
        np.testing.assert_allclose(g, w, rtol=1e-6, atol=0)


@pytest.mark.parametrize("bad", [0.0, np.nan], ids=["zero", "nan"])
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_a_zero_or_nan_norm_row_is_left_untouched(unscaled_state, backend, bad):
    """The zero-norm guard (the reference's ``Anrmk > 0``): a component row
    whose norm is zero, or NaN (``NaN > 0`` is false), keeps its row and its
    ``mu``/``beta`` columns bit for bit, while every other row is rescaled
    exactly as it is without the planted row (rows are independent)."""
    A, mu, beta, comp_list = unscaled_state
    h, i = 1, 7  # source 7 of model 1: a block other than the first
    k = comp_list[i, h]
    planted = A.copy()
    planted[k, :] = bad  # component k is row k (issue #334)
    dtype = np.float32 if backend == "mlx" else np.float64
    before = tuple(np.asarray(x, dtype) for x in (planted, mu, beta))

    got_A, got_mu, got_beta, _ = _rescaled(backend, (planted, mu, beta, comp_list))
    clean_A, clean_mu, clean_beta, _ = _rescaled(backend, unscaled_state)

    rows = np.zeros(A.shape, dtype=bool)
    rows[k, :] = True
    assert got_A[rows].tobytes() == before[0][rows].tobytes()
    assert got_mu[:, k].tobytes() == before[1][:, k].tobytes()
    assert got_beta[:, k].tobytes() == before[2][:, k].tobytes()

    others = np.arange(mu.shape[1]) != k
    assert got_A[~rows].tobytes() == clean_A[~rows].tobytes()
    assert got_mu[:, others].tobytes() == clean_mu[:, others].tobytes()
    assert got_beta[:, others].tobytes() == clean_beta[:, others].tobytes()
    # ...and those rows are unit norm, so the rescale really ran on them.
    unit = 1e-6 if backend == "mlx" else 1e-14
    norms = np.linalg.norm(got_A.astype(np.float64), axis=1)
    others_rows = np.arange(A.shape[0]) != k
    assert np.abs(norms[others_rows] - 1.0).max() <= unit


# --- 3. doscaling=False is byte-identical to the pre-fix code ---------------
# The epic #324 head this phase branched from: the last commit with the stored-
# column rule. A later phase that deliberately changes the doscaling=False
# trajectory moves this pin to its own base commit.
_PRE_FIX_COMMIT = "fb13d76de145419da9c89db94438d60aafbff444"


@pytest.fixture(scope="module")
def pre_fix(tmp_path_factory) -> Any:
    """The pamica package at ``_PRE_FIX_COMMIT``, imported beside the live one
    (``pamica/tests/pre_change.py``): both then fit in one process on one
    machine, so the comparison is exact without recorded constants."""
    return load_pre_change_package(
        _PRE_FIX_COMMIT, "pamica_pre333", tmp_path_factory.mktemp("pre333")
    )


def _pre_fix_class(pre_fix: Any, backend: str) -> Any:
    if backend == "torch":
        return pre_fix.torch_impl.core.AMICATorchNG
    if backend == "numpy":
        return pre_fix.numpy_impl.core.AMICA
    return importlib.import_module(f"{pre_fix.__name__}.mlx_impl.core").AMICAMLXNG


def _fit_unscaled(
    cls: Any, backend: str, n_models: int, X: np.ndarray, outdir: Path
) -> Dict[str, np.ndarray]:
    """A short ``doscaling=False`` fit of ``cls``; its trajectory and every
    fitted array, as float64/int numpy arrays, ``A`` as component rows (a
    pre-#334 class's is converted the way a pre-#334 save is)."""
    if backend == "numpy":
        model = cls(
            num_models=n_models,
            num_mix=NMIX,
            max_iter=6,
            seed=SEED,
            use_tqdm=False,
            outdir=str(outdir),
            do_opt_block=False,
            block_size=8192,
            doscaling=False,
            writestep=10000,
        )
        model.fit(X)
        ll = model.ll
    else:
        kwargs: Dict[str, Any] = dict(
            n_channels=NW, n_models=n_models, n_mix=NMIX, seed=SEED, doscaling=False
        )
        if backend == "torch":
            kwargs.update(device="cpu", dtype=torch.float64)
        model = cls(**kwargs)
        model.fit(X, max_iter=6, verbose=False)
        ll = model.ll_history
    out = {"ll": np.asarray(ll, dtype=np.float64)}
    for k in ("A", "W", "mu", "beta", "alpha", "rho", "gm", "c", "comp_list"):
        v = getattr(model, k)
        out[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
    if out["A"].shape[0] != out["comp_list"].size:
        out["A"] = rows_from_legacy_columns(out["A"], out["comp_list"], owner="test")
    return out


@pytest.mark.parametrize("n_models", [1, 2])
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_doscaling_off_is_byte_identical_to_the_pre_fix_code(
    pre_fix, real_slice, backend, n_models, tmp_path
):
    """``doscaling=False`` never enters the rescale, so this fix must leave its
    trajectory and every fitted array bit for bit where the pre-fix code put
    them, on every backend."""
    new_cls: Any
    if backend == "mlx":
        new_cls = _mlx_core().AMICAMLXNG
    else:
        new_cls = AMICATorchNG if backend == "torch" else AMICA_NumPy
    old_cls = _pre_fix_class(pre_fix, backend)

    old = _fit_unscaled(old_cls, backend, n_models, real_slice, tmp_path / "old")
    new = _fit_unscaled(new_cls, backend, n_models, real_slice, tmp_path / "new")
    changed = sorted(k for k in old if not np.array_equal(old[k], new[k]))
    assert not changed, f"doscaling=False results changed: {changed}"


# --- 4. scalestep cadence, counted from 1 -----------------------------------
def _fit_rows_dev(backend: str, scalestep: int, max_iter: int, X, tmp_path) -> float:
    """Largest component-row norm deviation after a real two-model fit of
    ``max_iter`` iterations (``keep_best`` off, so it is the last iterate)."""
    if backend == "numpy":
        n = AMICA_NumPy(
            num_models=2,
            num_mix=NMIX,
            max_iter=max_iter,
            seed=SEED,
            use_tqdm=False,
            outdir=str(tmp_path / f"s{scalestep}_k{max_iter}"),
            do_opt_block=False,
            block_size=4096,
            scalestep=scalestep,
            writestep=10000,
        )
        n.fit(X)
        assert n.A is not None and n.comp_list is not None
        return _row_norm_dev(n.A, n.comp_list)
    kwargs: Dict[str, Any] = dict(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        block_size=4096,
        keep_best=False,
        scalestep=scalestep,
    )
    if backend == "torch":
        m = AMICATorchNG(device="cpu", dtype=torch.float64, **kwargs)
        m.fit(X, max_iter=max_iter, verbose=False)
        assert m.A is not None and m.comp_list is not None
        return _row_norm_dev(m.A.numpy(), m.comp_list.numpy())
    m = _mlx_core().AMICAMLXNG(**kwargs)
    m.fit(X, max_iter=max_iter, verbose=False)
    return _row_norm_dev(np.array(m.A, dtype=np.float64), np.array(m.comp_list))


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
@pytest.mark.parametrize("scalestep", [1, 3])
def test_scalestep_counts_iterations_from_one(real_slice, backend, scalestep, tmp_path):
    """The reference rescales every iteration: it parses ``scalestep`` but never
    reads it (amica15.f90:1843, :3686). pamica keeps ``scalestep`` as an
    extension counted from 1 like the reference's other cadences, so the rescale
    runs on 1-based iterations ``scalestep``, ``2*scalestep``, ...; the default 1
    is the reference.

    Observed on real fits of 1 to 6 iterations: right after a rescale iteration
    every component row has unit norm, and one unscaled iteration already moves
    the rows by 1.7e-2 to 9.5e-2 (measured).
    """
    if backend == "mlx":
        _mlx_core()
    unit = 1e-6 if backend == "mlx" else 1e-14
    X = real_slice[:, :4096]
    for k in range(1, 7):
        dev = _fit_rows_dev(backend, scalestep, k, X, tmp_path)
        if k % scalestep == 0:
            assert dev <= unit, f"1-based iteration {k}: rows not rescaled ({dev})"
        else:
            assert dev > 1e-3, f"1-based iteration {k}: rows rescaled ({dev})"


def _construct(backend: str, outdir: Path, **kwargs: Any) -> Any:
    if backend == "numpy":
        return AMICA_NumPy(
            num_models=1, num_mix=NMIX, use_tqdm=False, outdir=str(outdir), **kwargs
        )
    if backend == "torch":
        return AMICATorchNG(n_channels=NW, n_mix=NMIX, device="cpu", **kwargs)
    return _mlx_core().AMICAMLXNG(n_channels=NW, n_mix=NMIX, **kwargs)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_every_backend_rejects_an_invalid_scalestep(backend, bad, tmp_path):
    """A zero ``scalestep`` used to surface as a bare ``ZeroDivisionError``
    mid-fit on every backend; each constructor now rejects anything but an
    integer >= 1, with one message, while the rescale is on."""
    with pytest.raises(ValueError, match="scalestep must be an integer >= 1"):
        _construct(backend, tmp_path, doscaling=True, scalestep=bad)
    # With the rescale off, scalestep is never read, so it is not validated.
    off = _construct(backend, tmp_path, doscaling=False, scalestep=bad)
    assert off.scalestep == bad


# --- 5. seeded native reference oracle (opt-in) -----------------------------
# The reference's input.param optimizer (natural gradient only here: the Newton
# start is epic #324 Phase 9, issue #335) with the block size pinned on both
# sides. pamica keyword -> reference keyword where the names differ.
_OPT: Dict[str, Any] = dict(
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
    do_newton=False,
    newt_start=50,
    newtrate=1.0,
    newt_ramp=10,
)
_REF_OPT: Dict[str, Any] = {
    **{k: v for k, v in _OPT.items() if k != "do_newton"},
    "do_newton": 0,
    "do_opt_block": 0,
    "do_reject": 0,
    "share_comps": 0,
    "doscaling": 1,
    "scalestep": 1,
}
# Tolerances per iteration count. The reference itself carries round-off that
# the ill-conditioned exact-EM mu update amplifies (mu of a low-mass mixture
# component), so each bound sits at the doscaling-OFF noise floor with margin.
# Measured maxima (doscaling on / off), each taken over both backends (PyTorch
# and NumPy) and both model counts, which is why they are wider than the
# single-configuration figures in the changelog and PR (PyTorch, one model:
# A 5.0e-16, mu 7.8e-11, sbeta 1.1e-14 after 1 iteration):
#   1 iteration:  A 2.2e-15 / 2.4e-15, mu 7.8e-11 / 8.0e-11, sbeta 2.4e-14
#   3 iterations: A 6.1e-12 / 3.1e-11, mu 3.6e-8 / 1.3e-8, sbeta 2.8e-10
# The column rule this replaced was off by A 7.2e-5, mu 6.6e-5, sbeta 8.6e-5
# after 1 iteration and 1.4e-3 / 6.7e-3 / 2.2e-3 after 3.
_ORACLE_TOL = {
    1: {"A": 1e-13, "mu": 1e-9, "sbeta": 1e-12, "LL": 1e-12},
    3: {"A": 1e-10, "mu": 1e-6, "sbeta": 1e-8, "LL": 1e-11},
}


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
@pytest.mark.parametrize("n_models", [1, 2])
def test_doscaling_matches_the_seeded_reference(n_models, tmp_path):
    """Seed the reference from pamica's initialization, run 1 and 3 iterations
    with ``doscaling`` on, and compare ``A``/``mu``/``sbeta`` element by element.

    The two-model case seeds ``comp_list`` through ``load_comp_list`` (the
    default one; ``c`` then starts at the reference's own zero, as pamica's
    does), so the helper's comp_list path is exercised for Phase 8.
    """
    from pamica.tests.native_oracle import (
        reference_mixing,
        run_seeded_reference,
        seed_from_torch,
    )

    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    X = X.astype(np.float64)

    def torch_model() -> AMICATorchNG:
        return AMICATorchNG(
            n_channels=NW,
            n_models=n_models,
            n_mix=NMIX,
            seed=SEED,
            device="cpu",
            dtype=torch.float64,
            keep_best=False,
            **_OPT,
        )

    init = torch_model()
    init._preprocess(X)
    init._initialize_parameters()
    state = seed_from_torch(init)
    assert init.comp_list is not None and init.sphere is not None
    comp_list = init.comp_list.numpy()
    if n_models > 1:
        state = dataclasses.replace(state, comp_list=comp_list)

    for k, tol in _ORACLE_TOL.items():
        ref = run_seeded_reference(
            state,
            DATA_FILE,
            tmp_path / f"k{k}",
            n_samples=FIELD,
            max_iter=k,
            num_models=n_models,
            **_REF_OPT,
        )
        if n_models > 1:
            assert "reading comp_list" in ref.stdout
            np.testing.assert_array_equal(ref.comp_list, comp_list)
        # Same data, same symmetric ZCA sphere (the reference computes its own).
        assert np.abs(ref.S - init.sphere.numpy()).max() < 1e-12
        # The reference's components are unit norm, as pamica's rows now are.
        assert np.abs(np.linalg.norm(ref.A, axis=0) - 1.0).max() < 1e-14

        t = torch_model()
        t.fit(X, max_iter=k, verbose=False)
        assert t.A is not None and t.mu is not None and t.beta is not None
        with tempfile.TemporaryDirectory() as td:
            n = AMICA_NumPy(
                num_models=n_models,
                num_mix=NMIX,
                max_iter=k,
                seed=SEED,
                use_tqdm=False,
                outdir=td,
                do_opt_block=False,
                writestep=10000,
                **_OPT,
            )
            n.fit(X)
        fits: Dict[str, tuple] = {
            "torch": (t.A.numpy(), t.mu.numpy(), t.beta.numpy(), t.ll_history),
            "numpy": (n.A, n.mu, n.beta, n.ll),
        }
        for name, (A, mu, beta, ll) in fits.items():
            errs = {
                "A": np.abs(reference_mixing(A) - ref.A).max(),
                "mu": np.abs(mu - ref.mu).max(),
                "sbeta": np.abs(beta - ref.sbeta).max(),
                "LL": np.abs(np.asarray(ll) - ref.LL).max(),
            }
            over = {q: e for q, e in errs.items() if not e <= tol[q]}
            assert not over, f"{name}, {k} iteration(s): {over} (bounds {tol})"
            assert _row_norm_dev(A, comp_list) <= 1e-14
