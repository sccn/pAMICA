"""Components are rows of ``A`` in every backend (issue #334, epic #324 Phase 8,
ADR 0007).

Before this change every array backend stored ``A`` as ``(n, n_comps)`` and took
model ``h``'s block as ``A[:, comp_list[:, h]]``, whose ROWS are the model's
components (issue #24 convention), while the component ids in ``comp_list``
indexed stored COLUMNS. ``share_comps`` therefore compared and tied the wrong
vectors: its metric measured stored columns and its merge tied one sphered
channel's loadings across two models. ``A`` is now ``(n_comps, n)`` with one
component per row, so a component id names the same component everywhere, as in
the reference (whose ``A(:, k)`` is our row ``k``).

Pinned here, cross-backend per ``.rules/backend_parity.md`` (PyTorch and NumPy
always run; MLX checks skip individually without MLX or an Apple GPU):

1. every configuration without a merge is byte-identical to the pre-change code
   (the package at commit ``0930c0e`` is loaded from git): one, two and three
   models, ``doscaling`` and Newton on and off, every ``pdftype``,
   ``do_reject``, several blocks, a ``keep_best`` restore and best-of-two
   restarts, each on every backend that supports it; the old ``A`` maps onto
   the new one through the conversion the persistence layer uses;
2. the semantics: (a) sources grouped by a real merge share one component map;
   (b) a planted exact duplicate is merged at the default and at a strict
   threshold, and the merge leaves the log-likelihood, ``transform`` and the
   maps unchanged; (c) the metric compares exactly the sensor maps
   ``get_sensor_mixing_matrix`` returns; (d) the pre-change column semantics
   fails (a) to (c), so none of them is vacuous;
3. the ``A`` the EEGLAB export writes is the reference's layout for several
   models, and single-model exports are unchanged byte for byte;
4. (opt-in, ``AMICA_RUN_FORTRAN=1``) the native reference binary, seeded with a
   MERGED ``comp_list`` through ``load_comp_list``, matches the PyTorch and
   NumPy updates from that state to float64 round-off; its own scan never
   merges; and an early mass merge collapses its update exactly as it
   collapses pamica's.

Persistence of the new layout and conversion of old saves are in
``test_component_rows_persistence.py``. Real bundled sample EEG only, no
synthetic data or mocks (``.rules/testing.md``).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from pamica import AMICA_NumPy
from pamica.component_layout import rows_from_legacy_columns
from pamica.numpy_impl.utils import identify_shared_components
from pamica.tests.pre_change import load_pre_change_package
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

# The epic #324 head this phase merges onto: the last commit with the
# component-column layout (Phase 7's doscaling fix and Phase 9's schedule
# gates included, so the two sides differ by this phase alone).
PRE_CHANGE_COMMIT = "0930c0e68ec9e2bbef3d20d51cff4c03029248f2"

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return X.astype(np.float64)


@pytest.fixture(scope="module")
def pre(tmp_path_factory) -> Any:
    """The pamica package at ``PRE_CHANGE_COMMIT``, imported beside the live one."""
    return load_pre_change_package(
        PRE_CHANGE_COMMIT, "pamica_pre334", tmp_path_factory.mktemp("pre334")
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
    """The live backend class."""
    if backend == "torch":
        return AMICATorchNG
    if backend == "numpy":
        return AMICA_NumPy
    return _mlx_core().AMICAMLXNG


def _classes(backend: str, pre: Any) -> Tuple[Any, Any]:
    """``(pre-change class, live class)`` for ``backend``."""
    live = _live(backend)
    if backend == "torch":
        return pre.torch_impl.core.AMICATorchNG, live
    if backend == "numpy":
        return pre.numpy_impl.core.AMICA, live
    old = importlib.import_module(f"{pre.__name__}.mlx_impl.core").AMICAMLXNG
    return old, live


def _np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.array(value)


# --- 1. byte identity of every configuration without a merge ----------------
# A short real slice: every configuration changes the trajectory within a few
# iterations (Newton from the third, rejection from the third), so each one
# exercises its own path.
_BYTE_ID_SAMPLES = 4096
_BYTE_ID_ITERS = 6


_ALL = ("torch", "numpy", "mlx")
# Newton from the second iteration. newtrate at the natural-gradient rate (0.1)
# keeps the ramp saturated: the likelihood falls on the iteration after
# pdftype=1's density switch, and the halved rate is ramped straight back to
# 0.1, so it steps the same whether the decrease response runs before the
# update (the reference's order, issue #339) or after it (the order the
# pre-change code, and this layout comparison, predate). A ramp toward the
# default newtrate 0.5 would move that step, which is not a layout effect.
_NEWTON: Dict[str, Any] = dict(newt_start=2, newtrate=0.1)
# The NumPy backend implements only the generalized-Gaussian density (it raises
# NotImplementedError for any other pdftype; test_numpy_rejects_other_pdftypes
# below) and has no keep_best option, so those configurations run on PyTorch
# and MLX only.
_NOT_NUMPY = ("torch", "mlx")


def _byte_id_configs() -> List[Any]:
    """``(backend, n_models, cfg)`` params, each configuration on every backend
    that supports it (see ``_NOT_NUMPY``)."""
    configs: List[Tuple[str, int, Dict[str, Any], Tuple[str, ...]]] = []
    for n_models in (1, 2):
        for doscaling in (True, False):
            for newton in (False, True):
                for pdftype in (0, 1):
                    cfg = dict(doscaling=doscaling, do_newton=newton, pdftype=pdftype)
                    if newton:
                        cfg.update(_NEWTON)
                    name = (
                        f"m{n_models}-{'scale' if doscaling else 'noscale'}-"
                        f"{'newton' if newton else 'ng'}-pdf{pdftype}"
                    )
                    configs.append(
                        (name, n_models, cfg, _ALL if pdftype == 0 else _NOT_NUMPY)
                    )
    newton: Dict[str, Any] = dict(do_newton=True, **_NEWTON)
    configs += [
        (
            "m2-reject-newton",
            2,
            dict(newton, do_reject=True, rejstart=2, rejint=2),
            _ALL,
        ),
        ("m3-scale-newton-pdf0", 3, newton, _ALL),
        ("m3-noscale-ng-pdf1", 3, dict(doscaling=False, pdftype=1), _NOT_NUMPY),
        # The other fixed density families (n_mix 1 for the single-component
        # sub-Gaussian cosh+, family 4; see _fit_arrays).
        ("m1-ng-pdf2", 1, dict(pdftype=2), _NOT_NUMPY),
        ("m1-ng-pdf3", 1, dict(pdftype=3), _NOT_NUMPY),
        ("m1-ng-pdf4", 1, dict(pdftype=4), _NOT_NUMPY),
        # Four blocks of 1024 samples: the per-block accumulation.
        ("m2-newton-4blocks", 2, dict(newton, block_size=1024), _ALL),
        # The best-iterate safeguard, with a learning rate high enough that the
        # log-likelihood falls, so it restores an earlier snapshot of A. At lrate
        # 0.5 the one decrease is on the last iteration (by 2.4e-2 on PyTorch and
        # MLX), whose update the restore discards, so the iteration order of
        # issue #339 acts on nothing compared here; at 1.5, the value before
        # that issue, the likelihood falls on iterations 2, 4, 5 and 6.
        ("m2-keepbest-restores", 2, dict(keep_best=True, lrate=0.5), _NOT_NUMPY),
        # Best-of-two restarts, which re-initializes A and keeps the winner.
        ("m2-restarts", 2, dict(n_restarts=2), _ALL),
    ]
    return [
        pytest.param(backend, n_models, cfg, id=f"{backend}-{name}")
        for name, n_models, cfg, backends in configs
        for backend in backends
    ]


def test_numpy_rejects_other_pdftypes():
    """The reason the non-GG byte-identity cases run on PyTorch and MLX only."""
    for pdftype in (1, 2, 3, 4):
        with pytest.raises(NotImplementedError, match="pdftype=0"):
            AMICA_NumPy(num_models=1, max_iter=1, use_tqdm=False, pdftype=pdftype)


def _fit_arrays(
    backend: str, cls: Any, n_models: int, cfg: Dict[str, Any], X: np.ndarray, out: Path
) -> Dict[str, np.ndarray]:
    """A short fit of ``cls``: its log-likelihood and gradient-norm record and
    every fitted array, as numpy. ``cfg`` overrides the defaults here (one
    block of the whole slice; ``keep_best`` off on PyTorch and MLX)."""
    n_mix = 1 if cfg.get("pdftype", 0) in (1, 4) else NMIX
    if backend == "numpy":
        params: Dict[str, Any] = dict(
            num_models=n_models,
            num_mix=n_mix,
            max_iter=_BYTE_ID_ITERS,
            seed=SEED,
            use_tqdm=False,
            outdir=str(out),
            do_opt_block=False,
            block_size=_BYTE_ID_SAMPLES,
            writestep=10000,
        )
        params.update(cfg)
        model = cls(**params)
        model.fit(X)
        ll, nd = model.ll, model.nd
    else:
        kwargs: Dict[str, Any] = dict(
            n_channels=NW,
            n_models=n_models,
            n_mix=n_mix,
            seed=SEED,
            block_size=_BYTE_ID_SAMPLES,
            keep_best=False,
        )
        kwargs.update(cfg)
        if backend == "torch":
            kwargs.update(device="cpu", dtype=torch.float64)
        model = cls(**kwargs)
        model.fit(X, max_iter=_BYTE_ID_ITERS, verbose=False)
        ll, nd = model.ll_history, [model._ndtmpsum]
    arrays = {"ll": np.asarray(ll, dtype=np.float64), "nd": np.asarray(nd)}
    names = ["A", "W", "mu", "beta", "alpha", "rho", "gm", "c", "comp_list"]
    if backend != "numpy":
        names.append("pdtype")
        arrays["final_ll"] = np.asarray(model.final_ll_, dtype=np.float64)
    for name in names:
        arrays[name] = _np(getattr(model, name))
    return arrays


@pytest.mark.parametrize("backend, n_models, cfg", _byte_id_configs())
def test_unshared_fits_are_byte_identical_to_the_column_layout(
    pre, real_data, backend, n_models, cfg, tmp_path
):
    """Without a merge every per-model block holds the same ``n x n`` matrix in
    both layouts, so every trajectory and fitted array is bit for bit what the
    pre-change code produced, the old ``A`` mapped onto component rows through
    :func:`pamica.component_layout.rows_from_legacy_columns`.

    The one recorded quantity that moves is the weight-gradient norm
    ``ndtmpsum``, by float round-off: it now sums squares per component row, as
    the reference sums each component's column (``nd(iter,:) =
    sum(dAk*dAk,1)``), where the column layout grouped them per stored column.
    Measured: at most 2.2e-16 relative in float64 and 1.1e-7 (one float32 unit
    in the last place) on MLX. It only feeds the gradient-norm stops, and no
    stop fired in these fits.
    """
    old_cls, new_cls = _classes(backend, pre)
    X = real_data[:, :_BYTE_ID_SAMPLES]
    old = _fit_arrays(backend, old_cls, n_models, cfg, X, tmp_path / "old")
    new = _fit_arrays(backend, new_cls, n_models, cfg, X, tmp_path / "new")

    assert new["A"].shape == (NW * n_models, NW)
    old["A"] = rows_from_legacy_columns(old["A"], old["comp_list"], owner="test")
    changed = sorted(k for k in old if k != "nd" and not np.array_equal(old[k], new[k]))
    assert not changed, f"results changed: {changed}"
    # Six iterations ran, so no stop fired (the premise of the nd note above).
    assert len(new["ll"]) == _BYTE_ID_ITERS
    if cfg.get("keep_best"):
        assert new["final_ll"] != new["ll"][-1], "setup: keep_best restored nothing"
    eps = np.finfo(np.float32 if backend == "mlx" else np.float64).eps
    np.testing.assert_allclose(new["nd"], old["nd"], rtol=float(2 * eps), atol=0)


# --- 2. semantics ----------------------------------------------------------------
# A shared recipe whose one scan merges a few pairs on the short slice: both
# models start near the identity, so their components stay near-collinear for
# the first iterations. At iteration 8 and comp_thresh 0.99 the scan merges
# three pairs (measured, PyTorch); the column metric on the same trajectory
# would merge five. A partial merge is what makes (d) discriminate: once every
# pair is merged, both layouts hold two identical blocks.
_SHARE = dict(share_comps=True, share_start=8, share_iter=100, comp_thresh=0.99)
_SHARE_SAMPLES = 4096
_SHARE_ITERS = 10


def _share_fit(backend: str, cls: Any, X: np.ndarray, out: Path) -> Any:
    """A real two-model share fit of ``cls`` whose scan at iteration 8 merges."""
    if backend == "numpy":
        model = cls(
            num_models=2,
            num_mix=NMIX,
            max_iter=_SHARE_ITERS,
            seed=SEED,
            use_tqdm=False,
            outdir=str(out),
            do_opt_block=False,
            block_size=_SHARE_SAMPLES,
            writestep=10000,
            share_start=_SHARE["share_start"],
            share_int=_SHARE["share_iter"],
            share_comps=True,
            comp_thresh=_SHARE["comp_thresh"],
        )
        model.fit(X)
        return model
    kwargs: Dict[str, Any] = dict(
        n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, block_size=_SHARE_SAMPLES
    )
    if backend == "torch":
        kwargs.update(device="cpu", dtype=torch.float64)
    model = cls(**kwargs, **_SHARE)
    model.fit(X, max_iter=_SHARE_ITERS, verbose=False)
    return model


def _groups(comp_list: np.ndarray) -> List[List[Tuple[int, int]]]:
    """``(model, source)`` groups sharing a component id across models."""
    groups = []
    for k in np.unique(comp_list):
        src, mdl = np.where(comp_list == k)
        if np.unique(mdl).size >= 2:
            groups.append([(int(h), int(i)) for i, h in zip(src, mdl)])
    return groups


def _mixing(model: Any, h: int) -> np.ndarray:
    """Model ``h``'s sphered-space mixing matrix, one source per column, for a
    live or pre-change model of any backend (NumPy has no accessor)."""
    if hasattr(model, "get_mixing_matrix"):
        return np.asarray(model.get_mixing_matrix(h), dtype=np.float64)
    A, comp_list = _np(model.A), _np(model.comp_list)
    if A.shape[0] == comp_list.size:  # component rows
        return A[comp_list[:, h], :].T
    return A[:, comp_list[:, h]].T  # the pre-change column layout


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_sources_grouped_by_a_merge_share_one_component_map(
    pre, real_data, backend, tmp_path
):
    """(a) Sources a merge groups use ONE component, so their mixing vectors
    and sensor maps are identical, and so are their densities. (d) The column
    semantics fails this: its grouped sources keep different maps."""
    old_cls, new_cls = _classes(backend, pre)
    X = real_data[:, :_SHARE_SAMPLES]
    new = _share_fit(backend, new_cls, X, tmp_path / "new")
    comp_list = _np(new.comp_list)
    groups = _groups(comp_list)
    assert groups, "setup: no merge fired"
    for group in groups:
        (h0, i0), rest = group[0], group[1:]
        k = comp_list[i0, h0]
        a0 = _mixing(new, h0)[:, i0]
        s0 = new.get_sensor_mixing_matrix(h0)[:, i0]
        for h, i in rest:
            assert comp_list[i, h] == k
            np.testing.assert_array_equal(_mixing(new, h)[:, i], a0)
            # Same vector through two matrix products of different shape.
            np.testing.assert_allclose(
                new.get_sensor_mixing_matrix(h)[:, i], s0, rtol=1e-12, atol=0
            )

    old = _share_fit(backend, old_cls, X, tmp_path / "old")
    old_groups = _groups(_np(old.comp_list))
    assert old_groups, "setup: the column semantics merged nothing"
    gaps = [
        np.abs(_mixing(old, h)[:, i] - _mixing(old, group[0][0])[:, group[0][1]]).max()
        for group in old_groups
        for h, i in group[1:]
    ]
    assert min(gaps) > 1e-2, f"the column semantics grouped identical maps: {gaps}"


# (b) A planted exact duplicate on a converged state. 150 natural-gradient and
# Newton iterations on the full sample leave the two models' best-matching
# components at |cos| 0.95-0.96 (issue #334), below the default threshold, so
# the planted pair is the only one a scan at 0.99 can merge (asserted).
_PLANT_ITERS = 150


@pytest.fixture(scope="module")
def converged_two_model_state(real_data) -> Dict[str, Any]:
    """A real two-model torch fit (sharing off) and its component-row state."""
    m = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        do_newton=True,
    )
    m.fit(real_data, max_iter=_PLANT_ITERS, verbose=False)
    assert int(m.comp_used.sum()) == m.n_comps, "setup: a merge fired"
    state: Dict[str, Any] = {
        k: _np(getattr(m, k)).copy() for k in ("A", "mu", "beta", "rho", "alpha")
    }
    state.update(
        {
            k: _np(getattr(m, k)).copy()
            for k in ("gm", "c", "comp_list", "mean", "sphere")
        }
    )
    maps = [m.get_sensor_mixing_matrix(h) for h in range(2)]
    maps = [q / np.linalg.norm(q, axis=0) for q in maps]
    cos = np.abs(maps[0].T @ maps[1])
    i, ii = np.unravel_index(int(np.argmax(cos)), cos.shape)
    state["pair"] = (int(i), int(ii))
    state["best_cos"] = float(cos[i, ii])
    state["sldet"] = m.sldet
    return state


def _plant(state: Dict[str, Any]) -> Dict[str, Any]:
    """Make model 1's source ``ii`` exactly model 0's source ``i``: the same
    component mixing vector (row) and density, the state the reference's
    sharing is designed to detect."""
    planted = {
        k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()
    }
    i, ii = state["pair"]
    cl = state["comp_list"]
    ci, cj = int(cl[i, 0]), int(cl[ii, 1])
    planted["A"][cj, :] = planted["A"][ci, :]
    for name in ("mu", "beta", "rho", "alpha"):
        planted[name][:, cj] = planted[name][:, ci]
    planted["ci"], planted["cj"] = ci, cj
    return planted


class _Probe:
    """One backend holding a given state: its scan, log-likelihood, sources
    and maps, through the backend's own methods."""

    def __init__(
        self, backend: str, cls: Any, state: Dict[str, Any], X: np.ndarray, out: Path
    ):
        self.backend, self.X = backend, X
        if backend == "torch":
            m = cls(
                n_channels=NW, n_models=2, n_mix=NMIX, device="cpu", dtype=torch.float64
            )
            for k in ("mu", "beta", "rho", "alpha", "gm", "c", "mean", "sphere"):
                setattr(m, k, torch.from_numpy(np.array(state[k], np.float64)))
            m.A = torch.from_numpy(np.array(state["A"], np.float64))
            m.comp_list = torch.from_numpy(state["comp_list"].copy())
            m.sldet = state["sldet"]
            m.pdtype = torch.zeros((NW, 2), dtype=torch.long)
            m._sphere_pinv = None
        elif backend == "numpy":
            # One real iteration sizes the model and preprocesses the data;
            # every parameter and the preprocessing are then replaced by the
            # given state, so the sphere matches the other backends' exactly.
            m = cls(
                num_models=2,
                num_mix=NMIX,
                max_iter=1,
                seed=SEED,
                use_tqdm=False,
                do_opt_block=False,
                block_size=8192,
                writestep=10**6,
                outdir=str(out),
            )
            m.fit(X)
            m.mean = np.array(state["mean"], np.float64).reshape(-1, 1)
            m.sphere = np.array(state["sphere"], np.float64)
            m.sldet = state["sldet"]
            m.data = m.sphere @ (X - m.mean)
            m._sphere_pinv = None
            for k in ("mu", "beta", "rho", "alpha", "gm", "c", "A"):
                setattr(m, k, np.array(state[k], np.float64))
            m.comp_list = state["comp_list"].copy()
        else:
            mx = _mlx_core().mx
            m = cls(n_channels=NW, n_models=2, n_mix=NMIX)
            for k in ("mu", "beta", "rho", "alpha", "gm", "c", "A"):
                setattr(m, k, mx.array(np.asarray(state[k], np.float32)))
            m.comp_list = mx.array(state["comp_list"].copy())
            m.mean = mx.array(np.asarray(state["mean"], np.float32).reshape(-1, 1))
            m.sphere = mx.array(np.asarray(state["sphere"], np.float32))
            m._sphere_np = np.array(m.sphere, dtype=np.float64)
            m._sphere_pinv = None
            m.sldet = state["sldet"]
            m.pdtype = mx.array(np.zeros((NW, 2), dtype=np.int32))
            used = np.zeros(2 * NW, dtype=bool)
            used[np.unique(state["comp_list"])] = True
            m._comp_used_arr = mx.array(used)
            m._refresh_lgamma_table()
        self.model = m
        self.refresh()

    def refresh(self) -> None:
        m = self.model
        if self.backend == "numpy":
            m.comp_used = np.zeros(2 * NW, dtype=bool)
            m.comp_used[np.unique(m.comp_list)] = True
        elif self.backend == "mlx":
            mx = _mlx_core().mx
            used = np.zeros(2 * NW, dtype=bool)
            used[np.unique(np.array(m.comp_list))] = True
            m._comp_used_arr = mx.array(used)
        m._update_unmixing_matrices()

    @property
    def comp_list(self) -> np.ndarray:
        return _np(self.model.comp_list).copy()

    def set_comp_list(self, comp_list: np.ndarray) -> None:
        m = self.model
        if self.backend == "torch":
            m.comp_list = torch.from_numpy(comp_list.copy())
        elif self.backend == "numpy":
            m.comp_list = comp_list.copy()
        else:
            m.comp_list = _mlx_core().mx.array(comp_list.copy())
        self.refresh()

    def scan(self, thresh: float) -> None:
        self.model.comp_thresh = thresh
        self.model._identify_shared_comps()
        self.refresh()

    def ll(self) -> float:
        m = self.model
        if self.backend == "numpy":
            return float(m._get_updates_and_likelihood()["ll"])
        lht = np.asarray(m.model_loglik(self.X), dtype=np.float64)
        return float(logsumexp(lht, axis=0).sum() / (lht.shape[1] * NW))

    def sources(self, h: int) -> np.ndarray:
        m = self.model
        if self.backend == "numpy":
            return m.transform(self.X)[m.comp_list[:, h], :, h]
        return np.asarray(m.transform(self.X, model_idx=h))

    def maps(self, h: int) -> np.ndarray:
        return np.asarray(self.model.get_sensor_mixing_matrix(h))


@pytest.mark.parametrize("thresh", [0.99, 1.0 - 1e-12], ids=["default", "strict"])
@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_a_planted_duplicate_is_merged_and_the_model_is_unchanged(
    converged_two_model_state, real_data, backend, thresh, tmp_path
):
    """(b) Plant the reference's notion of a shared component: model 1's source
    ``ii`` set to exactly model 0's source ``i`` (mixing vector and density).
    The scan must fold exactly that pair, and the fold must leave the model as
    it was, because the tied parameters were already equal: log-likelihood,
    ``transform`` and the maps are bit for bit unchanged (issue #334 measured
    the column fold at -0.186 on such a pair, the reference's at exactly 0)."""
    planted = _plant(converged_two_model_state)
    ii = planted["pair"][1]
    ci, cj = planted["ci"], planted["cj"]
    probe = _Probe(backend, _live(backend), planted, real_data, tmp_path)
    before = (probe.ll(), probe.sources(1), probe.maps(1))
    cl0 = probe.comp_list

    probe.scan(thresh)
    cl = probe.comp_list
    folded = np.argwhere(cl != cl0)
    assert folded.tolist() == [[ii, 1]], f"expected only the planted pair: {folded}"
    assert cl[ii, 1] == ci and cj not in cl

    after = (probe.ll(), probe.sources(1), probe.maps(1))
    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])
    np.testing.assert_array_equal(after[2], before[2])


def _column_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """A component-row state in the pre-change column layout (the inverse of
    the persistence conversion, valid for an unmerged comp_list)."""
    col = dict(state)
    A_rows, cl = state["A"], state["comp_list"]
    A_cols = np.empty((A_rows.shape[1], A_rows.shape[0]), dtype=A_rows.dtype)
    for h in range(cl.shape[1]):
        A_cols[:, cl[:, h]] = A_rows[cl[:, h], :]
    col["A"] = A_cols
    return col


def test_the_column_semantics_misses_and_breaks_a_planted_duplicate(
    pre, converged_two_model_state, real_data, tmp_path
):
    """(d) The same planted state under the pre-change column semantics: its
    metric does not see the pair at the default threshold, and folding the pair
    the column way changes the log-likelihood (issue #334: -0.186), so test (b)
    distinguishes the two semantics."""
    planted = _plant(converged_two_model_state)
    (i, ii), ci, cj = planted["pair"], planted["ci"], planted["cj"]
    old_cls = pre.torch_impl.core.AMICATorchNG
    old = _Probe("torch", old_cls, _column_state(planted), real_data, tmp_path)
    ll0 = old.ll()
    cl0 = old.comp_list
    old.scan(0.99)
    found = old.comp_list[ii, 1] == old.comp_list[i, 0]
    assert not found, "the column metric found the planted pair"
    cl = cl0.copy()
    cl[cl == cj] = ci
    old.set_comp_list(cl)
    assert abs(old.ll() - ll0) > 1e-2


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_the_metric_compares_the_sensor_maps(
    converged_two_model_state, real_data, backend, tmp_path
):
    """(c) The vector the scan compares for source ``i`` of model ``h`` is
    column ``i`` of ``get_sensor_mixing_matrix(h)``, and the scan's decision is
    the shared kernel's decision on those maps. (d) The pre-change metric
    vectors (de-sphered stored columns) are not the maps at all."""
    probe = _Probe(
        backend, _live(backend), converged_two_model_state, real_data, tmp_path
    )
    vectors = probe.model._component_sensor_maps()
    cl = probe.comp_list
    for h in range(2):
        maps = probe.maps(h)
        # The same vectors through matrix products of different shape.
        scale = np.abs(maps).max()
        np.testing.assert_allclose(
            vectors[:, cl[:, h]], maps, rtol=0, atol=1e-13 * scale
        )

    # The decision on a state where the default threshold merges nothing, at a
    # threshold where it merges several pairs.
    thresh = 0.9
    by_maps = np.empty((NW, 2 * NW))
    for h in range(2):
        by_maps[:, cl[:, h]] = probe.maps(h)
    want, _ = identify_shared_components(by_maps, cl, thresh)
    probe.scan(thresh)
    assert not np.array_equal(want, cl), "setup: nothing merged at 0.9"
    np.testing.assert_array_equal(probe.comp_list, want)

    # The pre-change metric on the same fitted model: stored columns.
    old_state = _column_state(converged_two_model_state)
    spinv = np.linalg.pinv(old_state["sphere"])
    columns = spinv @ old_state["A"]
    columns /= np.linalg.norm(columns, axis=0)
    cos = []
    for h in range(2):
        maps = spinv @ converged_two_model_state["A"][cl[:, h], :].T
        maps /= np.linalg.norm(maps, axis=0)
        cos.append(np.abs(np.sum(columns[:, cl[:, h]] * maps, axis=0)))
    assert np.median(np.concatenate(cos)) < 0.6, "the column metric saw the maps"


# --- 3. the EEGLAB export writes the reference's A ------------------------------
_EXPORT_FILES = (
    "gm",
    "W",
    "S",
    "mean",
    "c",
    "alpha",
    "mu",
    "sbeta",
    "rho",
    "comp_list",
    "LL",
    "LLt",
    "A",
)


def _export(backend: str, model: Any, out: Path) -> Path:
    """The EEGLAB directory of a fitted model (NumPy's fit already wrote it)."""
    if backend != "numpy":
        model.write_amica_output(out)
    return out


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_the_exported_A_is_the_reference_layout(real_data, backend, tmp_path):
    """Two models, a real merge included: the ``A`` file is the reference's
    ``A(nw, num_comps)`` in column-major order, so column ``comp_list[i, h]``
    read that way is source ``i`` of model ``h``, and the bytes are exactly the
    component-row ``A`` in C order. ``load_results`` reads the rows back."""
    from pamica.numpy_impl.data import load_results

    model = _share_fit(backend, _live(backend), real_data[:, :_SHARE_SAMPLES], tmp_path)
    comp_list = _np(model.comp_list)
    assert _groups(comp_list), "setup: no merge fired"
    out = _export(backend, model, tmp_path)
    raw = np.fromfile(out / "A", dtype="<f8")
    A_ref = raw.reshape(NW, 2 * NW, order="F")
    for h in range(2):
        np.testing.assert_array_equal(A_ref[:, comp_list[:, h]], _mixing(model, h))
    A_rows = np.ascontiguousarray(_np(model.A), dtype=np.float64)
    assert raw.tobytes() == A_rows.tobytes()
    np.testing.assert_array_equal(load_results(out)["A"], A_rows)


def _write_single_model(backend: str, cls: Any, X: np.ndarray, out: Path) -> Path:
    if backend == "numpy":
        cls(
            num_models=1,
            num_mix=NMIX,
            max_iter=_BYTE_ID_ITERS,
            seed=SEED,
            use_tqdm=False,
            outdir=str(out),
            do_opt_block=False,
            block_size=_BYTE_ID_SAMPLES,
            writestep=10000,
        ).fit(X)
        return out
    kwargs: Dict[str, Any] = dict(
        n_channels=NW, n_mix=NMIX, seed=SEED, block_size=_BYTE_ID_SAMPLES
    )
    if backend == "torch":
        kwargs.update(device="cpu", dtype=torch.float64)
    model = cls(**kwargs)
    model.fit(X, max_iter=_BYTE_ID_ITERS, verbose=False)
    model.write_amica_output(out)
    return out


@pytest.mark.parametrize("backend", ["torch", "numpy", "mlx"])
def test_single_model_export_is_byte_identical_to_the_pre_change_code(
    pre, real_data, backend, tmp_path
):
    """One model's ``A`` is the same bytes in both layouts, so every file of a
    single-model export, ``A`` included, is what the pre-change code wrote."""
    old_cls, new_cls = _classes(backend, pre)
    X = real_data[:, :_BYTE_ID_SAMPLES]
    old = _write_single_model(backend, old_cls, X, tmp_path / "old")
    new = _write_single_model(backend, new_cls, X, tmp_path / "new")
    for name in _EXPORT_FILES:
        assert (new / name).read_bytes() == (old / name).read_bytes(), name


def test_load_results_refuses_a_pre_change_multi_model_A(pre, real_data, tmp_path):
    """A two-model directory the pre-change code wrote holds its component-
    column ``A`` in C order, which read as component rows does not invert the
    ``W`` beside it: ``load_results`` refuses it with the remedy instead of
    handing the viz helpers scrambled maps. The same directory's other files,
    and a single-model directory, are unaffected."""
    from pamica.numpy_impl.data import load_results

    X = real_data[:, :_BYTE_ID_SAMPLES]
    kwargs = dict(n_channels=NW, n_mix=NMIX, seed=SEED, device="cpu")
    old_cls = pre.torch_impl.core.AMICATorchNG
    two = old_cls(n_models=2, **kwargs)
    two.fit(X, max_iter=_BYTE_ID_ITERS, verbose=False)
    two.write_amica_output(tmp_path / "two")
    with pytest.raises(ValueError, match="does not invert the unmixing W"):
        load_results(tmp_path / "two")

    one = old_cls(n_models=1, **kwargs)
    one.fit(X, max_iter=_BYTE_ID_ITERS, verbose=False)
    one.write_amica_output(tmp_path / "one")
    np.testing.assert_array_equal(load_results(tmp_path / "one")["A"], _np(one.A))


@pytest.mark.parametrize("keep", ["one-value-short", "one-component-short"])
def test_load_results_refuses_a_truncated_A(real_data, tmp_path, keep):
    """A truncated ``A`` file (a copy of a real two-model export) is refused
    by name, whether the lost bytes leave a length that no reshape accepts or
    one short by exactly one component, which would otherwise reshape and fail
    later as a matrix-product error."""
    import shutil

    from pamica.numpy_impl.data import load_results

    model = AMICATorchNG(n_channels=NW, n_models=2, n_mix=NMIX, seed=SEED, device="cpu")
    model.fit(real_data[:, :_BYTE_ID_SAMPLES], max_iter=_BYTE_ID_ITERS, verbose=False)
    model.write_amica_output(tmp_path / "full")
    assert load_results(tmp_path / "full")["A"].shape == (2 * NW, NW)

    shutil.copytree(tmp_path / "full", tmp_path / "cut")
    raw = (tmp_path / "full" / "A").read_bytes()
    lost = 8 if keep == "one-value-short" else 8 * NW
    (tmp_path / "cut" / "A").write_bytes(raw[:-lost])
    with pytest.raises(ValueError, match=r"holds \d+ values, expected 2048"):
        load_results(tmp_path / "cut")


# --- 4. seeded native reference oracle from a MERGED state (opt-in) -------------
# The reference's input.param optimizer, natural gradient only (the Newton start
# is issue #335), with the block size pinned on both sides. ``maxrho`` sits just
# below 2 so no mixture reaches the reference's exact-Gaussian branch
# (``rho == 2``, amica15.f90:1309-1313), whose normalizer is a single-precision
# literal, ``log(dble(1.772453851))``, 3.0e-8 away from ``log(sqrt(pi))``: once a
# warm fit clamps a mixture there, every sample's log-density differs from
# pamica's exact one by that much times its responsibility, which is unrelated
# to the mixing layout and would swamp the round-off this test measures. (The
# Laplace branch, ``rho == 1``, is exact in both.) Whether pamica adopts the
# single-precision value is issue #344.
_OPT: Dict[str, Any] = dict(
    block_size=512,
    lrate=0.05,
    lratefact=0.5,
    rholrate=0.05,
    rholratefact=0.5,
    rho0=1.5,
    minrho=1.0,
    maxrho=1.99,  # below the reference's single-precision rho == 2 branch (#344)
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
    "scalestep": 1,
}
_ORACLE_WARM_ITERS = 100
# Per iteration count: bounds at the no-merge noise floor with margin, the same
# as test_doscaling_rows.py's oracle. Measured maxima over both backends and
# both doscaling settings, from the merged state (the same state with no merge
# in brackets, PyTorch, doscaling on):
#   1 iteration:  A 1.3e-15, mu 2.3e-11, sbeta 1.4e-14, LL 8.9e-16
#                 [A 1.1e-15, mu 2.3e-11, sbeta 1.3e-14, LL 1.8e-15]
#   3 iterations: A 1.5e-12, mu 3.7e-9,  sbeta 1.8e-11, LL 4.7e-14
#                 [A 1.9e-12, mu 1.4e-8,  sbeta 1.9e-10, LL 8.0e-15]
# The column semantics from the same merged state is off by A 0.204 and 0.21 and
# LL 3.5e-4 and 4.3e-4 (doscaling on and off) after 3 iterations.
_MERGED_ORACLE_TOL = {
    1: {"A": 1e-13, "mu": 1e-9, "sbeta": 1e-12, "LL": 1e-13},
    3: {"A": 1e-10, "mu": 1e-6, "sbeta": 1e-8, "LL": 1e-11},
}


@pytest.fixture(scope="module")
def merged_seed(real_data) -> Dict[str, Any]:
    """A real two-model state with one merge applied to ``comp_list`` exactly as
    the scan folds it (the pair whose sensor maps agree best), ``c`` zeroed for
    the reference's ``load_comp_list`` path."""
    warm = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        keep_best=False,
        **_OPT,
    )
    warm.fit(real_data, max_iter=_ORACLE_WARM_ITERS, verbose=False)
    maps = [warm.get_sensor_mixing_matrix(h) for h in range(2)]
    maps = [q / np.linalg.norm(q, axis=0) for q in maps]
    cos = np.abs(maps[0].T @ maps[1])
    i, ii = np.unravel_index(int(np.argmax(cos)), cos.shape)
    default = _np(warm.comp_list).copy()
    merged = default.copy()
    merged[merged == merged[ii, 1]] = merged[i, 0]
    seed: Dict[str, Any] = {
        k: _np(getattr(warm, k)).copy()
        for k in ("A", "mu", "beta", "rho", "alpha", "gm", "mean", "sphere")
    }
    seed.update(default=default, merged=merged, cos=float(cos[i, ii]))
    # No mixture sits at rho == 2, where the reference's normalizer is a
    # single-precision literal (see _OPT).
    assert seed["rho"].max() < 2.0
    return seed


def _seeded_run(
    model: Any, backend: str, seed: Dict[str, Any], X: np.ndarray, k: int, rows: bool
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Run ``k`` production iterations of ``model`` from the merged seed, in
    the reference's load order: ``get_unmixing_matrices`` runs on the DEFAULT
    ``comp_list`` and ``load_comp_list`` replaces it afterwards
    (amica15.f90:825-833), so the first E-step unmixes with the pre-merge
    blocks while indexing densities by the merged ``comp_list``. ``rows`` is
    False for a pre-change class, which gets the seed's ``A`` in its column
    layout. Returns the per-iteration log-likelihood, each model's mixing
    matrix (the reference's ``A(:, comp_list(:, h))``), ``mu`` and ``sbeta``."""
    A = seed["A"]
    if not rows:
        A = _column_state({"A": A, "comp_list": seed["default"]})["A"]
    lls: List[float] = []
    if backend == "torch":
        X_t = model._preprocess(X)
        model._initialize_parameters()
        model.A = torch.from_numpy(A.copy())
        for name in ("mu", "beta", "rho", "alpha", "gm"):
            setattr(model, name, torch.from_numpy(seed[name].copy()))
        model.c = torch.zeros(NW, 2, dtype=torch.float64)
        model.comp_list = torch.from_numpy(seed["default"].copy())
        model._update_unmixing_matrices()
        model.comp_list = torch.from_numpy(seed["merged"].copy())
        n = X_t.shape[1]
        for it in range(k):
            model.iteration = it
            acc = model._accumulate_blocks(X_t)
            lls.append(float(acc["ll"]) / (n * NW))
            model._update_parameters(acc, n)
    else:
        model.fit(X)  # sizes and preprocesses; the state is replaced below
        model.mean = seed["mean"].reshape(-1, 1).copy()
        model.sphere = seed["sphere"].copy()
        model.data = model.sphere @ (X - model.mean)
        model._sphere_pinv = None
        model.A = A.copy()
        for name in ("mu", "beta", "rho", "alpha", "gm"):
            setattr(model, name, seed[name].copy())
        model.c = np.zeros((NW, 2))
        model.lrate, model.rholrate = _OPT["lrate"], _OPT["rholrate"]
        model.rholrate_cap = _OPT["rholrate"]
        model.ll, model.nd = [], []
        model.comp_list = seed["default"].copy()
        model._update_unmixing_matrices()
        model.comp_list = seed["merged"].copy()
        model.comp_used = np.isin(np.arange(2 * NW), model.comp_list)
        for it in range(k):
            model.iter = it
            model._update_parameters(model._get_updates_and_likelihood())
        lls = list(model.ll)
    mixing = [_mixing(model, h) for h in range(2)]
    return np.asarray(lls), mixing, _np(model.mu), _np(model.beta)


def _mixing_error(
    mixing: List[np.ndarray], A_ref: np.ndarray, comp_list: np.ndarray
) -> float:
    """Largest difference between each model's mixing matrix and the
    reference's ``A(:, comp_list(:, h))``."""
    return max(
        float(np.abs(mixing[h] - A_ref[:, comp_list[:, h]]).max())
        for h in range(comp_list.shape[1])
    )


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
@pytest.mark.parametrize("doscaling", [True, False], ids=["scale", "noscale"])
def test_updates_from_a_merged_state_match_the_seeded_reference(
    pre, merged_seed, real_data, doscaling, tmp_path
):
    """The reference's own scan is unrunnable (its ``Spinv2`` is never
    allocated), but its ``load_comp_list`` seeds a merged ``comp_list``, and
    the E-step, the ``gm``-weighted ``dAk/zeta`` average over the models that
    share a component, the ``comp_used`` mask and the M-step then run on it.
    PyTorch and NumPy match that to float64 round-off after one and three
    iterations; the pre-change column semantics does not."""
    from pamica.tests.native_oracle import SeedState, run_seeded_reference

    seed = merged_seed
    used = np.unique(seed["merged"])
    state = SeedState(
        A=seed["A"].T.copy(),
        mean=seed["mean"].reshape(-1),
        mu=seed["mu"],
        sbeta=seed["beta"],
        rho=seed["rho"],
        alpha=seed["alpha"],
        gm=seed["gm"],
        c=np.zeros((NW, 2)),
        comp_list=seed["merged"],
    )
    opt = dict(_OPT, doscaling=doscaling)

    def torch_model(cls: Any) -> Any:
        return cls(
            n_channels=NW,
            n_models=2,
            n_mix=NMIX,
            seed=SEED,
            device="cpu",
            dtype=torch.float64,
            keep_best=False,
            **opt,
        )

    for k, tol in _MERGED_ORACLE_TOL.items():
        ref = run_seeded_reference(
            state,
            DATA_FILE,
            tmp_path / f"k{k}",
            n_samples=FIELD,
            max_iter=k,
            num_models=2,
            doscaling=int(doscaling),
            **_REF_OPT,
        )
        assert "reading comp_list" in ref.stdout
        np.testing.assert_array_equal(ref.comp_list, seed["merged"])
        assert np.abs(ref.S - seed["sphere"]).max() < 1e-12

        numpy_model = AMICA_NumPy(
            num_models=2,
            num_mix=NMIX,
            max_iter=1,
            seed=SEED,
            use_tqdm=False,
            outdir=str(tmp_path / f"np{k}"),
            do_opt_block=False,
            writestep=10**6,
            **opt,
        )
        runs = {
            "torch": _seeded_run(
                torch_model(AMICATorchNG), "torch", seed, real_data, k, rows=True
            ),
            "numpy": _seeded_run(numpy_model, "numpy", seed, real_data, k, rows=True),
        }
        for name, (ll, mixing, mu, sbeta) in runs.items():
            errs = {
                "A": _mixing_error(mixing, ref.A, seed["merged"]),
                "mu": np.abs(mu[:, used] - ref.mu[:, used]).max(),
                "sbeta": np.abs(sbeta[:, used] - ref.sbeta[:, used]).max(),
                "LL": np.abs(ll - ref.LL).max(),
            }
            print(f"merged oracle doscaling={doscaling} {name} k={k}: {errs}")
            over = {q: e for q, e in errs.items() if not e <= tol[q]}
            assert not over, f"{name}, {k} iteration(s): {over} (bounds {tol})"

        if k == 3:
            old = torch_model(pre.torch_impl.core.AMICATorchNG)
            ll_old, mixing, _, _ = _seeded_run(old, "torch", seed, real_data, k, False)
            old_err = _mixing_error(mixing, ref.A, seed["merged"])
            print(
                f"merged oracle doscaling={doscaling} column semantics k=3: "
                f"A {old_err:.3g}, LL {np.abs(ll_old - ref.LL).max():.3g}"
            )
            assert old_err > 1e-2


# --- 5. early mass merges behave the same in the reference (opt-in) -------------
# The kind of recipe the sharing tests used before issue #334 made the metric
# compare true component maps: 4096 samples, pamica's default optimizer, an
# early scan at a loose threshold. The first scan (iteration 11) merges 25
# components and the second model's gm falls from 0.53 to 6.0e-4 within two
# iterations; by the second scan (iteration 22, which merges nothing more) it
# is 2e-22, and the fit goes non-finite at iteration 25. Issue #345 moved the
# A-freeze to the reference's iterations (mod(iter, share_iter) <= 5), so
# share_start is a multiple of share_iter here, which puts each window on its
# scan iteration; the earlier recipe (seed 7, scans at 8 and 18) now goes
# non-finite at iteration 18, before its second scan. _COLLAPSE_REF spells pamica's
# defaults out for the binary, with no further scans, no A-freeze and no
# convergence stops, so both sides run the same updates from the same state.
_COLLAPSE_SAMPLES = 4096
_COLLAPSE: Dict[str, Any] = dict(
    n_channels=NW,
    n_models=2,
    n_mix=NMIX,
    seed=20,
    device="cpu",
    dtype=torch.float64,
    block_size=1024,
    share_comps=True,
    share_start=11,
    share_iter=11,
    comp_thresh=0.9,
)
_COLLAPSE_REF: Dict[str, Any] = dict(
    block_size=1024,
    do_opt_block=0,
    lrate=0.1,
    lratefact=0.5,
    minlrate=1e-12,
    rholrate=0.05,
    rholratefact=0.1,
    rho0=1.5,
    minrho=1.0,
    maxrho=2.0,
    invsigmin=1e-4,
    invsigmax=1000.0,
    do_newton=0,
    newt_ramp=10,
    do_reject=0,
    share_comps=0,
    share_start=10**6,
    share_iter=100,
    use_min_dll=0,
    use_grad_norm=0,
)


def _with(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """``base`` with ``overrides`` applied (a copy)."""
    merged: Dict[str, Any] = dict(base)
    merged.update(overrides)
    return merged


def _post_scan(model: AMICATorchNG) -> Dict[str, Any]:
    """A fitted model's state as a seed: the merged ``comp_list``, the default
    one the reference unmixes with first, and ``c`` zeroed (the reference's
    ``load_comp_list`` path reads the file ``c``, so it needs ``c == 0``)."""
    seed = {
        k: _np(getattr(model, k)).copy()
        for k in ("A", "mu", "beta", "rho", "alpha", "gm", "mean")
    }
    seed["merged"] = _np(model.comp_list).copy()
    seed["default"] = np.stack([np.arange(NW), NW + np.arange(NW)], axis=1)
    return seed


def _reference_seed(seed: Dict[str, Any]) -> Any:
    from pamica.tests.native_oracle import SeedState

    return SeedState(
        A=seed["A"].T.copy(),
        mean=seed["mean"].reshape(-1),
        mu=seed["mu"],
        sbeta=seed["beta"],
        rho=seed["rho"],
        alpha=seed["alpha"],
        gm=seed["gm"],
        c=np.zeros((NW, 2)),
        comp_list=seed["merged"],
    )


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
def test_the_reference_scan_never_merges(real_data, tmp_path):
    """The reference's similarity weights by ``Spinv2``, which is never
    allocated: its scan runs, every similarity is NaN, and nothing merges even
    at ``comp_thresh=0``, where any finite similarity would merge."""
    from pamica.tests.native_oracle import run_seeded_reference

    init = AMICATorchNG(**_with(_COLLAPSE, share_comps=False))
    init._preprocess(real_data[:, :_COLLAPSE_SAMPLES])
    init._initialize_parameters()
    seed = _post_scan(init)
    ref = run_seeded_reference(
        _reference_seed(seed),
        DATA_FILE,
        tmp_path,
        n_samples=_COLLAPSE_SAMPLES,
        max_iter=3,
        num_models=2,
        **_with(_COLLAPSE_REF, share_comps=1, share_start=2, comp_thresh=0.0),
    )
    assert "Number of unique components" in ref.stdout, "the scan did not run"
    np.testing.assert_array_equal(ref.comp_list, seed["default"])
    assert "Identifying component" not in ref.stdout


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
def test_an_early_mass_merge_collapses_the_reference_too(real_data, tmp_path):
    """From pamica's state right after each early scan, the reference's update
    loses the second model's responsibility exactly as pamica's does: after the
    first scan both drop its ``gm`` from above 0.3 to below 0.01 within two
    iterations, and after the second both drive it to zero in one iteration and
    go non-finite in the next (the binary reports NaN and reinitializes). So
    the collapse is the algorithm on models that have not separated, not a
    defect of the port."""
    from pamica.tests.native_oracle import run_seeded_reference

    X = real_data[:, :_COLLAPSE_SAMPLES]

    def continuation(seed: Dict[str, Any], k: int) -> Tuple[np.ndarray, float]:
        """``k`` iterations from ``seed`` in the reference's load order (see
        ``_seeded_run``, whose accessors refuse the non-finite state this
        reaches), with no scan and no A-freeze, like ``_COLLAPSE_REF``."""
        model = AMICATorchNG(**_with(_COLLAPSE, share_comps=False))
        X_t = model._preprocess(X)
        model._initialize_parameters()
        for name in ("A", "mu", "beta", "rho", "alpha", "gm"):
            setattr(model, name, torch.from_numpy(seed[name].copy()))
        model.c = torch.zeros(NW, 2, dtype=torch.float64)
        model.comp_list = torch.from_numpy(seed["default"].copy())
        model._update_unmixing_matrices()
        model.comp_list = torch.from_numpy(seed["merged"].copy())
        lls = []
        for it in range(k):
            model.iteration = it
            acc = model._accumulate_blocks(X_t)
            lls.append(float(acc["ll"]) / (X_t.shape[1] * NW))
            model._update_parameters(acc, X_t.shape[1])
        return np.asarray(lls), float(_np(model.gm)[1])

    first = AMICATorchNG(**_COLLAPSE)
    # The scan runs on the last iteration of each fit.
    first.fit(X, max_iter=_COLLAPSE["share_start"], verbose=False)
    seed = _post_scan(first)
    merges = 2 * NW - len(np.unique(seed["merged"]))
    assert merges >= 20, f"setup: the first scan merged only {merges}"
    assert seed["gm"][1] > 0.3, "setup: the second model had already collapsed"
    ref = run_seeded_reference(
        _reference_seed(seed),
        DATA_FILE,
        tmp_path / "first",
        n_samples=_COLLAPSE_SAMPLES,
        max_iter=2,
        num_models=2,
        **_COLLAPSE_REF,
    )
    lls, gm2 = continuation(seed, 2)
    print(f"after the first scan: gm2 reference {ref.gm[1]:.3e}, torch {gm2:.3e}")
    assert np.isfinite(lls).all() and np.isfinite(ref.LL).all()
    assert gm2 < 0.01 and ref.gm[1] < 0.01
    np.testing.assert_allclose(gm2, ref.gm[1], rtol=1e-2)

    second = AMICATorchNG(**_COLLAPSE)
    second.fit(
        X, max_iter=_COLLAPSE["share_start"] + _COLLAPSE["share_iter"], verbose=False
    )
    assert second.stop_reason == "max_iter", "the fit collapsed before its second scan"
    seed = _post_scan(second)
    ref = run_seeded_reference(
        _reference_seed(seed),
        DATA_FILE,
        tmp_path / "second1",
        n_samples=_COLLAPSE_SAMPLES,
        max_iter=1,
        num_models=2,
        **_COLLAPSE_REF,
    )
    lls, gm2 = continuation(seed, 1)
    assert ref.gm[1] == 0.0 and gm2 == 0.0
    with pytest.raises(RuntimeError, match="Reinitiali"):
        run_seeded_reference(
            _reference_seed(seed),
            DATA_FILE,
            tmp_path / "second2",
            n_samples=_COLLAPSE_SAMPLES,
            max_iter=2,
            num_models=2,
            **_COLLAPSE_REF,
        )
    lls, _ = continuation(seed, 2)
    assert np.isfinite(lls[0]) and not np.isfinite(lls[1])
