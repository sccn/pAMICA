"""Every drawn initial component has unit norm, as in the reference (issue #341,
epic #324 Phase 12).

The reference draws each model's block of ``A`` as ``0.01 * (0.5 - u)``, sets
its diagonal to one and divides every component (a column of its ``A``, a row
of pamica's, ADR 0007) by its Euclidean norm (amica15.f90:805-823); its restart
after a non-finite likelihood redraws the same way (:1026-1044). A loaded ``A``
is used as is (:793-802), and ``fix_init`` starts from the identity. Before
this change pamica drew ``I + 0.01 * (0.5 - u)``, whose components have norms
within about 5e-3 of one, so the first E-step, and with ``doscaling`` off the
whole fit, ran on a scale the reference never uses.

Pinned here, cross-backend per ``.rules/backend_parity.md`` (PyTorch and NumPy
always run; MLX checks skip individually without MLX or an Apple GPU):

1. the shared draw (:func:`pamica.initialization.initial_mixing`) is the
   reference's recipe, written out below loop for loop from amica15.f90, and
   its normalization refuses a zero or non-finite component instead of
   returning NaN;
2. every backend starts every fit (one and two models, a single fit and each
   of best-of-two restarts) from that draw: unit-norm components, the same
   bits in PyTorch and NumPy and their float32 cast in MLX, unchanged up to
   the first update; the same checks fail on the pre-change code (commit
   ``027cb07``, loaded from git), whose components are off by more than 1e-3;
3. a supplied or loaded ``A`` is used bit for bit (a NumPy ``A`` set before
   ``fit``, a NumPy refit, a PyTorch ``state_dict``, an MLX save): only a
   drawn ``A`` is normalized;
4. the NumPy restart after a non-finite likelihood redraws unit-norm
   components from the running generator (the sanctioned error-injection
   pattern of ``.rules/testing.md`` poisons one real likelihood);
5. (opt-in, ``AMICA_RUN_FORTRAN=1``) the native binary's own drawn
   initialization has unit-norm components with the recipe's shape, and the
   binary, loaded with pamica's draw BEFORE its normalization, normalizes it
   to pamica's initial ``A`` to float64 round-off.

A bit-level oracle of the whole initialization is not possible: the binary
draws from gfortran's ``random_number`` and pamica from NumPy's
``RandomState``, and the binary normalizes only a draw of its own (a loaded
``A`` is used as is, and ``fix_init`` draws nothing). Oracle 5 therefore checks
the two halves separately: the binary's draw path normalizes (on its own
draw), and its normalization arithmetic, which is the same expression in its
initialization (:818-819) and in its per-iteration rescale (:1845-1847),
reproduces pamica's on pamica's draw, reached through ``load_A`` with the
``A`` update off so that the rescale is the only change to ``A``.

Real bundled sample EEG only, no synthetic data or mocks
(``.rules/testing.md``). The only instrumentation is a call-recording spy on
the real ``_initialize_parameters`` and ``_update_parameters``, which calls
straight through and never substitutes a result.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from pamica import AMICA_NumPy
from pamica.initialization import (
    draw_initial_block,
    initial_mixing,
    normalize_components,
)
from pamica.tests.pre_change import load_pre_change_package
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
N_SLICE = 4096
BACKENDS = ["torch", "numpy", "mlx"]

# The epic #324 head this phase branched from: the last commit whose backends
# drew I + 0.01 * (0.5 - u) without normalizing it.
PRE_CHANGE_COMMIT = "027cb0731dbaaa8e7a5795166af95ba7ff52c2cd"

# Unit-norm bounds. Measured over 500 seeds (two models of 32): the float64
# rows are within 4.4e-16 (2 units in the last place) of norm one, and their
# float32 cast (MLX) within 3.0e-8 (a quarter of a float32 unit); the pre-change
# draw's rows sit 1e-4 to 5e-3 away.
UNIT_TOL = {np.dtype(np.float64): 1e-15, np.dtype(np.float32): 1.2e-7}
# A pre-change row deviates by at least this much somewhere in every draw (the
# largest deviation over its 32 rows is 4e-3 to 5e-3 on the seeds used here).
PRE_CHANGE_MIN_DEV = 1e-3

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def X() -> np.ndarray:
    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data.astype(np.float64)[:, :N_SLICE]


@pytest.fixture(scope="module")
def pre(tmp_path_factory) -> Any:
    """The package before this change, imported beside the live one."""
    return load_pre_change_package(
        PRE_CHANGE_COMMIT, "pamica_pre341", tmp_path_factory.mktemp("pre341")
    )


def _mlx_core():
    """The MLX backend module, or skip this test (never the whole module)."""
    mlx_core = pytest.importorskip(
        "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
    )
    mx = mlx_core.mx
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("no Apple GPU")
    return mlx_core


def _live(backend: str) -> Any:
    if backend == "torch":
        return AMICATorchNG
    if backend == "numpy":
        return AMICA_NumPy
    return _mlx_core().AMICAMLXNG


def _pre_change(backend: str, pre: Any) -> Any:
    if backend == "torch":
        return pre.torch_impl.core.AMICATorchNG
    if backend == "numpy":
        return pre.numpy_impl.core.AMICA
    _mlx_core()  # skip without MLX, as for the live class
    return importlib.import_module(f"{pre.__name__}.mlx_impl.core").AMICAMLXNG


def _np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.array(value)


def _reference_recipe(rng: np.random.RandomState, n: int, n_models: int) -> np.ndarray:
    """amica15.f90:805-823 written out loop for loop, NumPy's generator in
    place of gfortran's and components as rows (ADR 0007): per model, one
    ``n x n`` draw ``0.01 * (0.5 - u)``; then per component, the diagonal
    entry set to one and the component divided by ``sqrt(sum(a * a))``."""
    blocks = []
    for _ in range(n_models):
        a = 0.01 * (0.5 - rng.rand(n, n))
        for i in range(n):
            a[i, i] = 1.0
            a[i, :] = a[i, :] / np.sqrt(np.sum(a[i, :] * a[i, :]))
        blocks.append(a)
    return np.vstack(blocks)


def _row_norm_dev(A: np.ndarray) -> float:
    norms = np.linalg.norm(np.asarray(A, dtype=np.float64), axis=1)
    return float(np.abs(norms - 1.0).max())


# --- 1. the shared draw is the reference's recipe -----------------------------
@pytest.mark.parametrize("n_models", [1, 2, 3])
def test_the_shared_draw_is_the_reference_recipe(n_models):
    """Bit for bit, over several seeds, including the generator's position
    afterward (``n * n`` values per model, so the ``mu`` and ``beta`` draws
    that follow are the ones they always were). With ``fix_init`` the draw is
    the identity and consumes nothing (amica15.f90:806-810)."""
    for seed in (0, 7, SEED):
        got_rng, want_rng = np.random.RandomState(seed), np.random.RandomState(seed)
        A = initial_mixing(got_rng, NW, n_models)
        assert A.shape == (n_models * NW, NW) and A.dtype == np.float64
        np.testing.assert_array_equal(A, _reference_recipe(want_rng, NW, n_models))
        assert _row_norm_dev(A) <= UNIT_TOL[A.dtype]
        assert got_rng.rand() == want_rng.rand()

        rng, untouched = np.random.RandomState(seed), np.random.RandomState(seed)
        fixed = initial_mixing(rng, NW, n_models, fix_init=True)
        np.testing.assert_array_equal(fixed, np.vstack([np.eye(NW)] * n_models))
        assert rng.rand() == untouched.rand()


@pytest.mark.parametrize("bad", [0.0, np.nan], ids=["zero", "nan"])
def test_normalizing_a_zero_or_nan_component_raises(bad):
    """The reference's normalization has no zero-norm guard because its draw
    cannot need one (the unit diagonal keeps every norm at least 1). A direct
    call on another block must not divide by a zero or NaN norm silently."""
    block = draw_initial_block(np.random.RandomState(SEED), NW)
    block[5, :] = bad
    with pytest.raises(ValueError, match=r"component row\(s\) \[5\]"):
        normalize_components(block)


# --- 2. every backend starts every fit from that draw -------------------------
Record = Tuple[Optional[int], np.ndarray, np.ndarray]


def _record_starts(model: Any, monkeypatch) -> List[list]:
    """Spy on the real ``_initialize_parameters`` and ``_update_parameters``:
    for each initialization, ``[seed, A after it, A at the entry of the next
    update]``. Both spies call straight through."""
    records: List[list] = []
    initialize, update = model._initialize_parameters, model._update_parameters

    def initialize_spy(*args, **kwargs):
        result = initialize(*args, **kwargs)
        records.append([model.seed, _np(model.A), None])
        return result

    def update_spy(*args, **kwargs):
        if records and records[-1][2] is None:
            records[-1][2] = _np(model.A)
        return update(*args, **kwargs)

    monkeypatch.setattr(model, "_initialize_parameters", initialize_spy)
    monkeypatch.setattr(model, "_update_parameters", update_spy)
    return records


def _fit_recording(
    cls: Any,
    backend: str,
    n_models: int,
    n_restarts: int,
    X: np.ndarray,
    tmp_path: Path,
    monkeypatch,
) -> List[list]:
    """A real two-iteration fit of ``cls`` from ``SEED`` with the start spies."""
    if backend == "numpy":
        model = cls(
            num_models=n_models,
            num_mix=NMIX,
            max_iter=2,
            seed=SEED,
            n_restarts=n_restarts,
            use_tqdm=False,
            outdir=str(tmp_path),
            do_opt_block=False,
            block_size=N_SLICE,
            writestep=10**6,
        )
        records = _record_starts(model, monkeypatch)
        model.fit(X)
        return records
    kwargs: Dict[str, Any] = dict(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        n_restarts=n_restarts,
        block_size=N_SLICE,
    )
    if backend == "torch":
        kwargs.update(device="cpu", dtype=torch.float64)
    model = cls(**kwargs)
    records = _record_starts(model, monkeypatch)
    model.fit(X, max_iter=2, verbose=False)
    return records


def _check_starts(backend: str, n_models: int, n_restarts: int, records) -> None:
    """Each fit started from the reference recipe for its own seed: unit-norm
    components, the float64 recipe's bits (their float32 cast on MLX), and the
    same ``A`` when the first update came."""
    assert [r[0] for r in records] == [SEED + i for i in range(n_restarts)]
    for seed, start, first_update in records:
        assert first_update is not None, "no update ran"
        np.testing.assert_array_equal(start, first_update)
        assert _row_norm_dev(start) <= UNIT_TOL[start.dtype], (
            f"seed {seed}: initial components off unit norm by "
            f"{_row_norm_dev(start):.2e}"
        )
        want = _reference_recipe(np.random.RandomState(seed), NW, n_models)
        if backend == "mlx":
            want = want.astype(np.float32)
        np.testing.assert_array_equal(start, want)


@pytest.mark.parametrize("n_restarts", [1, 2])
@pytest.mark.parametrize("n_models", [1, 2])
@pytest.mark.parametrize("backend", BACKENDS)
def test_every_fit_starts_from_unit_norm_components(
    backend, n_models, n_restarts, X, tmp_path, monkeypatch
):
    """Unit-norm drawn components before the first update, on every backend,
    for a single fit and for each of best-of-two restarts. The recipe check
    also makes the backends agree: PyTorch and NumPy start from the same
    float64 bits, MLX from their float32 cast."""
    records = _fit_recording(
        _live(backend), backend, n_models, n_restarts, X, tmp_path, monkeypatch
    )
    _check_starts(backend, n_models, n_restarts, records)


@pytest.mark.parametrize("n_models", [1, 2])
@pytest.mark.parametrize("backend", BACKENDS)
def test_the_pre_change_code_fails_the_start_checks(
    pre, backend, n_models, X, tmp_path, monkeypatch
):
    """The control: the code before this change drew ``I + 0.01 * (0.5 - u)``
    and never normalized it, so the same checks fail on it, on the norm."""
    records = _fit_recording(
        _pre_change(backend, pre), backend, n_models, 2, X, tmp_path, monkeypatch
    )
    for _, start, _ in records:
        assert _row_norm_dev(start) > PRE_CHANGE_MIN_DEV
    with pytest.raises(AssertionError, match="off unit norm"):
        _check_starts(backend, n_models, 2, records)


# --- 3. a supplied or loaded A is used as is ----------------------------------
@pytest.fixture(scope="module")
def fitted_unscaled(X) -> Dict[str, np.ndarray]:
    """A real ``A`` whose components are far from unit norm: three iterations
    of a two-model fit with ``doscaling`` off."""
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
    t.fit(X, max_iter=3, verbose=False)
    A = _np(t.A)
    assert _row_norm_dev(A) > 1e-2, "setup: the fitted components are unit norm"
    return {"A": A, "comp_list": _np(t.comp_list)}


def _numpy_model(outdir: Path, **kwargs: Any) -> Any:
    params: Dict[str, Any] = dict(
        num_models=2,
        num_mix=NMIX,
        max_iter=1,
        seed=SEED,
        use_tqdm=False,
        outdir=str(outdir),
        do_opt_block=False,
        block_size=N_SLICE,
        writestep=10**6,
    )
    params.update(kwargs)
    return AMICA_NumPy(**params)


def test_numpy_uses_an_A_supplied_before_fit_as_is(
    fitted_unscaled, X, tmp_path, monkeypatch
):
    """The NumPy backend initializes a parameter only while it is ``None``, so
    an ``A`` set before ``fit`` is its starting point, like the reference's
    loaded ``A`` (amica15.f90:793-802): used bit for bit, not normalized."""
    supplied = fitted_unscaled["A"]
    model = _numpy_model(tmp_path)
    model.A = supplied.copy()
    records = _record_starts(model, monkeypatch)
    model.fit(X)
    assert len(records) == 1
    _, start, first_update = records[0]
    assert start.tobytes() == supplied.tobytes()
    assert first_update.tobytes() == supplied.tobytes()


def test_numpy_refit_continues_from_its_fitted_A_as_is(X, tmp_path, monkeypatch):
    """A second ``fit`` on the same NumPy instance starts from the first fit's
    final ``A`` (``docs/guides/amica-differences.md``, issue #312), which
    ``doscaling`` off leaves far from unit norm: used bit for bit."""
    model = _numpy_model(tmp_path, max_iter=3, doscaling=False)
    model.fit(X)
    assert model.A is not None
    fitted = model.A.copy()
    assert _row_norm_dev(fitted) > 1e-2
    records = _record_starts(model, monkeypatch)
    model.fit(X)
    assert len(records) == 1
    assert records[0][1].tobytes() == fitted.tobytes()


def test_a_torch_state_dict_restores_A_as_is(X):
    """A PyTorch ``state_dict`` restores the fitted ``A`` bit for bit."""
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
    t.fit(X, max_iter=3, verbose=False)
    fitted = _np(t.A)
    assert _row_norm_dev(fitted) > 1e-2
    restored = AMICATorchNG.from_state_dict(t.state_dict(), device="cpu")
    assert _np(restored.A).tobytes() == fitted.tobytes()


def test_an_mlx_save_restores_A_as_is(X, tmp_path):
    """An MLX save, through ``state_dict`` and through the file, restores the
    fitted ``A`` bit for bit."""
    cls = _mlx_core().AMICAMLXNG
    m = cls(n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, doscaling=False)
    m.fit(X, max_iter=3, verbose=False)
    fitted = _np(m.A)
    assert _row_norm_dev(fitted) > 1e-2
    assert _np(cls.from_state_dict(m.state_dict()).A).tobytes() == fitted.tobytes()
    path = tmp_path / "model.npz"
    m.save(str(path))
    assert _np(cls.load(str(path)).A).tobytes() == fitted.tobytes()


# --- 4. the NumPy restart after a non-finite likelihood -----------------------
def _nan_on_iteration(base: Any, nan_iter: int) -> Any:
    """``base`` with its real E-step likelihood poisoned on one 0-based
    iteration: the sanctioned error-injection pattern (``.rules/testing.md``,
    as ``test_schedule_gates.py``'s ``_NaNOnIteration``). Everything else, the
    restart included, is the real code on real data."""

    class NaNOnIteration(base):
        def _get_updates_and_likelihood(self):
            updates = super()._get_updates_and_likelihood()
            if self.iter == nan_iter:
                updates["ll"] = float("nan")
            return updates

    return NaNOnIteration


def _numpy_restart_starts(
    cls: Any, n_models: int, X: np.ndarray, tmp_path: Path, monkeypatch
) -> List[list]:
    """The starts of a real NumPy fit whose second likelihood is NaN, inside
    the restart window: the initial draw and the restart's redraw."""
    model = _nan_on_iteration(cls, 1)(
        num_models=n_models,
        num_mix=NMIX,
        max_iter=4,
        seed=SEED,
        use_tqdm=False,
        outdir=str(tmp_path),
        do_opt_block=False,
        block_size=N_SLICE,
        writestep=10**6,
        restartiter=10,
        maxrestarts=3,
    )
    records = _record_starts(model, monkeypatch)
    model.fit(X)
    assert model.numrestarts == 1
    assert len(records) == 2, "setup: the NaN did not restart the fit"
    return records


@pytest.mark.parametrize("n_models", [1, 2])
def test_the_numpy_restart_redraws_unit_norm_components(
    n_models, X, tmp_path, monkeypatch
):
    """The redraw continues the running generator past the first fit's ``A``,
    ``mu`` and ``beta`` draws, and normalizes like the initial draw
    (amica15.f90:1039-1040)."""
    records = _numpy_restart_starts(AMICA_NumPy, n_models, X, tmp_path, monkeypatch)
    (_, first, _), (_, redraw, redraw_update) = records
    assert redraw_update is not None
    np.testing.assert_array_equal(redraw, redraw_update)
    assert _row_norm_dev(redraw) <= UNIT_TOL[redraw.dtype]
    assert not np.array_equal(redraw, first)

    rng = np.random.RandomState(SEED)
    np.testing.assert_array_equal(first, _reference_recipe(rng, NW, n_models))
    n_comps = NW * n_models
    for _ in range(n_comps):
        rng.rand(NMIX)  # mu, one component at a time
    rng.rand(NMIX, n_comps)  # beta
    np.testing.assert_array_equal(redraw, _reference_recipe(rng, NW, n_models))


def test_the_pre_change_numpy_restart_redrew_unnormalized_components(
    pre, X, tmp_path, monkeypatch
):
    """The control: the pre-change redraw is off unit norm."""
    records = _numpy_restart_starts(
        pre.numpy_impl.core.AMICA, 2, X, tmp_path, monkeypatch
    )
    assert _row_norm_dev(records[1][1]) > PRE_CHANGE_MIN_DEV


# --- 5. the native reference binary (opt-in) ----------------------------------
run_fortran = pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)


def _off_diagonal_ratios(A: np.ndarray, n: int) -> np.ndarray:
    """``a[j] / a[i]`` over every component ``a`` (a row) with its own
    diagonal entry ``a[i]`` (``i`` = its index within its model), ``j != i``:
    the draw before normalization, whose off-diagonal entries lie in
    (-0.005, 0.005]."""
    ratios = []
    for k, a in enumerate(A):
        i = k % n
        assert np.argmax(np.abs(a)) == i, f"component {k}: diagonal not largest"
        ratios.append(np.delete(a, i) / a[i])
    return np.concatenate(ratios)


@run_fortran
@pytest.mark.parametrize("seed", [1, SEED])
@pytest.mark.parametrize("n_models", [1, 2])
def test_the_reference_draws_unit_norm_components(n_models, seed, tmp_path):
    """The binary's own drawn initialization (``max_iter=0`` writes it before
    any iteration): every component has unit norm, its diagonal entry is its
    largest, and the other entries relative to it lie within the draw's
    half-width 0.005. pamica's draw has the same shape (measured on the
    binary: norms within 4.4e-16 of one, ratios up to 0.004998)."""
    from pamica.tests.native_oracle import reference_mixing, run_drawn_reference

    ref = run_drawn_reference(
        DATA_FILE,
        tmp_path,
        nw=NW,
        n_samples=FIELD,
        num_models=n_models,
        num_mix=NMIX,
        seed=seed,
    )
    assert ref.LL.size == 0  # no iteration ran
    # The reference layout's columns are components: pamica's rows.
    components = reference_mixing(ref.A)
    assert _row_norm_dev(components) <= UNIT_TOL[np.dtype(np.float64)]
    half_width = 0.01 * 0.5
    assert np.abs(_off_diagonal_ratios(components, NW)).max() <= half_width

    ours = initial_mixing(np.random.RandomState(seed), NW, n_models)
    assert np.abs(_off_diagonal_ratios(ours, NW)).max() <= half_width


@run_fortran
@pytest.mark.parametrize("n_models", [1, 2])
def test_the_reference_normalizes_pamicas_draw_as_pamica_does(n_models, tmp_path):
    """The binary loaded with pamica's draw BEFORE normalization (``load_A``,
    used as is) and every other parameter of pamica's initialization, with the
    ``A`` update off, rescales each component in its first iteration with the
    same expression its initialization normalizes with (amica15.f90:1845-1847
    and :818-819): the result is pamica's initial ``A`` to float64 round-off
    (measured: 4.4e-16, two units in the last place, for one and two models),
    while the loaded draw was off unit norm by 1.8e-4."""
    from pamica.tests.native_oracle import (
        reference_mixing,
        run_seeded_reference,
        seed_from_torch,
    )

    full = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    init = AMICATorchNG(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
    )
    init._preprocess(full.astype(np.float64))
    init._initialize_parameters()
    rng = np.random.RandomState(SEED)
    drawn = np.vstack([draw_initial_block(rng, NW) for _ in range(n_models)])
    assert _row_norm_dev(drawn) > 1e-5, "setup: the draw is already unit norm"
    state = dataclasses.replace(seed_from_torch(init), A=reference_mixing(drawn))

    ref = run_seeded_reference(
        state,
        DATA_FILE,
        tmp_path,
        n_samples=FIELD,
        max_iter=1,
        num_models=n_models,
        update_A=0,
        doscaling=1,
        do_newton=0,
    )
    assert init.A is not None
    err = np.abs(ref.A - reference_mixing(_np(init.A))).max()
    assert err <= UNIT_TOL[np.dtype(np.float64)], f"off by {err:.2e}"
