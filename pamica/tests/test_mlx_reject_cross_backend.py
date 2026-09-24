"""``do_reject`` and ``mir()`` cross-backend agreement: MLX vs PyTorch --
issue #123's ``AMICATorchNG`` mechanism and issue #137, epic #278 Phase
3/#289.

Cross-backend by design, so it lives in ``pamica/tests/`` rather than
``pamica/tests/mlx_tests/`` (``.rules/backend_parity.md``: a test that pins
two backends against each other belongs outside any one backend's
subdirectory), the same placement as
``test_mlx_sharing_cross_backend.py``/``test_mlx_transform_cross_backend.py``.
The MLX-only mechanics of both features stay in
``pamica/tests/mlx_tests/test_mlx_reject.py`` and
``pamica/tests/mlx_tests/test_mlx_mir.py``; this is the ONE new
cross-backend file the phase plan calls for, so both anti-drift agreement
pins ride here together.

1. **do_reject agreement**: same real data, same config, ``AMICATorchNG``
   (float64) and ``AMICAMLXNG`` (float32) must make the same rejection
   decision on every sample that is not within float32 rounding of the
   threshold. This is the load-bearing evidence for the phase's design
   decision: MLX reads the rejection statistic FROM the LLt stash instead of
   a second forward pass (the NumPy backend's design, pre-empting
   AMICATorchNG's open follow-up #298) -- if that were not mathematically
   equivalent to torch's own ``_sample_ll`` forward-pass statistic, the two
   backends would drift apart on which samples they drop. Exact agreement of
   the whole rejected set is not a property of float32, so the check is per
   pass and explains every disagreement by the rounding measured on that
   pass (issue #333 moved the trajectories enough to flip one borderline
   sample of the earlier exact-set pin).

2. **do_reject x pdftype=1 agreement**: a PR review regression
   (``_choose_pdfs`` was called with the FULL sphered dataset on MLX,
   ``X_t``, instead of the do_reject-restricted good set ``X_use`` AMICATorchNG
   passes -- so the kurtosis-based family decision silently saw the
   outliers torch excludes). Fixed to pass ``X_use``; pinned here against a
   float64 torch twin, same seed/data/config, so a regression shows up as a
   per-source ``pdtype`` mismatch, not just a shrinking-size assertion.

3. **mir() agreement**: from ONE real fitted MLX state copied into a
   float64 ``AMICATorchNG`` twin (the ``_torch_twin`` pattern from
   ``test_mlx_sharing_cross_backend.py``), both backends' ``mir()`` must
   agree to float32 precision.

MLX is an optional Apple-Silicon backend, so the module self-skips via
``importorskip`` plus an Apple-GPU guard; PyTorch always runs. Real bundled
sample EEG only, no synthetic data or mocks (``.rules/testing.md``).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica.torch_impl.core import AMICATorchNG

mx = pytest.importorskip("mlx.core", reason="MLX not installed (Apple Silicon only)")
mlx_core = pytest.importorskip(
    "pamica.mlx_impl.core", reason="MLX not installed (Apple Silicon only)"
)
AMICAMLXNG = mlx_core.AMICAMLXNG

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


def _real_data(n_samples: int = 4096) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data[:, :n_samples].astype(np.float64)


# --- do_reject: the same decisions, up to float32 rounding at the threshold -
class _RecordingTorch(AMICATorchNG):
    """Records the input of each rejection pass, then runs the real pass.

    A pass-through spy, not a stub: ``_reject_outliers`` itself runs unchanged
    on the real statistic, so the fit is exactly the production fit. It only
    keeps what the backend decided on: the good set before the pass (sample
    indices) and the per-sample log-likelihood over it, in the same order.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.passes: list[tuple[np.ndarray, np.ndarray]] = []

    def _reject_outliers(self, ll_vec: torch.Tensor) -> None:
        assert self.good_idx is not None
        self.passes.append((self.good_idx.numpy().copy(), ll_vec.numpy().copy()))
        super()._reject_outliers(ll_vec)


class _RecordingMLX(AMICAMLXNG):
    """The MLX twin of :class:`_RecordingTorch` (float32 statistic)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.passes: list[tuple[np.ndarray, np.ndarray]] = []

    def _reject_outliers(self, ll_vec: np.ndarray) -> None:
        assert self.good_idx is not None
        self.passes.append((np.array(self.good_idx), np.array(ll_vec).copy()))
        super()._reject_outliers(ll_vec)


def _torch_threshold(ll: np.ndarray, rejsig: float) -> float:
    """``mean - rejsig * std`` (population std) in float64, with the same
    torch operations as ``AMICATorchNG._reject_outliers``."""
    t = torch.from_numpy(ll)
    mean = t.mean()
    std = torch.sqrt((t.pow(2).mean() - mean.pow(2)).clamp_min(0.0))
    return float(mean - rejsig * std)


def _mlx_threshold(ll: np.ndarray, rejsig: float) -> float:
    """The same threshold with the float32 numpy operations of
    ``AMICAMLXNG._reject_outliers``."""
    mean = float(ll.mean())
    std = float(np.sqrt(max(float(np.mean(ll**2) - mean**2), 0.0)))
    return mean - rejsig * std


# A float32 statistic can flip a decision only for samples within its own
# rounding of the threshold, so the band that explains a disagreement is
# measured per pass (below). These caps are what make a REAL divergence fail
# instead of being absorbed into a band it inflated: an orientation or
# trajectory bug makes the two per-sample log-likelihoods differ by the order
# of their spread, and a threshold bug moves the threshold by a sizable
# fraction of it. Measured over seeds 0-5, 7 and 8, one and two models,
# rejsig 2.0, 2.5 and 3.0 (Apple M4 Pro): the band was at most 3.6e-2 of the
# per-sample log-likelihood's standard deviation, held at most 14 samples
# (0.36% of a pass), and at most 1 sample ended up rejected by one backend and
# kept by the other (the band's worst sample, a two-model fit's second pass).
_MAX_BAND_OVER_STD = 0.1
_MAX_BORDERLINE_FRACTION = 0.01
_MAX_DISAGREEMENTS = 3


def _rejection_violations(
    m64: _RecordingTorch, m32: _RecordingMLX, rejsig64: float, rejsig32: float
) -> list[str]:
    """Every way the two backends' rejections differ by more than float32
    rounding at the threshold (empty when they agree up to that rounding)."""
    assert m64.good_idx is not None and m32.good_idx is not None
    final64 = m64.good_idx.numpy()
    final32 = np.array(m32.good_idx)
    problems: list[str] = []
    if len(m64.passes) != len(m32.passes):
        return [f"pass counts differ: {len(m64.passes)} vs {len(m32.passes)}"]
    explained: set[int] = set()
    for p, ((g64, l64), (g32, l32)) in enumerate(zip(m64.passes, m32.passes)):
        after64 = m64.passes[p + 1][0] if p + 1 < len(m64.passes) else final64
        after32 = m32.passes[p + 1][0] if p + 1 < len(m32.passes) else final32
        t64 = _torch_threshold(l64, rejsig64)
        t32 = _mlx_threshold(l32, rejsig32)
        # The thresholds recomputed here are the ones each backend applied.
        for name, g, ll, t, after in (
            ("torch", g64, l64, t64, after64),
            ("mlx", g32, l32, t32, after32),
        ):
            if not np.array_equal(g[ll >= t], after):
                problems.append(f"pass {p}: {name} did not apply mean - rejsig*std")

        common, i64, i32 = np.intersect1d(g64, g32, return_indices=True)
        x64 = l64[i64]
        diff = np.abs(x64 - l32[i32].astype(np.float64))
        # Measured on this pass: the largest float32-vs-float64 difference of
        # a per-sample log-likelihood, plus the resulting threshold difference.
        band = float(diff.max()) + abs(t64 - t32)
        spread = float(np.std(x64))
        if band > _MAX_BAND_OVER_STD * spread:
            problems.append(
                f"pass {p}: the backends' statistics differ by {band:.3g}, "
                f"{band / spread:.2g} of their spread (cap {_MAX_BAND_OVER_STD})"
            )
        borderline = np.abs(x64 - t64) <= band
        if borderline.mean() > _MAX_BORDERLINE_FRACTION:
            problems.append(
                f"pass {p}: {int(borderline.sum())} of {common.size} samples lie "
                f"within {band:.3g} of the threshold"
            )
        disagree = (l64[i64] >= t64) != (l32[i32] >= t32)
        # Every decision outside the band is identical on both backends.
        if (disagree & ~borderline).any():
            far = common[disagree & ~borderline]
            problems.append(f"pass {p}: samples {far.tolist()} disagree off the band")
        explained |= set(common[disagree].tolist())

    rejected_by_one = set(final64.tolist()) ^ set(final32.tolist())
    if not rejected_by_one <= explained:
        problems.append(
            f"samples {sorted(rejected_by_one - explained)} differ in the final "
            "good sets without a disagreeing pass decision"
        )
    if len(rejected_by_one) > _MAX_DISAGREEMENTS:
        problems.append(
            f"{len(rejected_by_one)} samples rejected by one backend only "
            f"(cap {_MAX_DISAGREEMENTS})"
        )
    return problems


def _reject_fits(
    n_models: int, seed: int, rejsig32: float = 2.0, seed32: int | None = None
) -> tuple[_RecordingTorch, _RecordingMLX]:
    X = _real_data()
    kwargs: dict[str, Any] = dict(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        block_size=BLOCK,
        do_reject=True,
        rejstart=2,
        rejint=3,
        maxrej=2,
        keep_best=True,
    )
    m32 = _RecordingMLX(
        **kwargs, rejsig=rejsig32, seed=seed if seed32 is None else seed32
    )
    m32.fit(X, max_iter=10, verbose=False)
    m64 = _RecordingTorch(
        **kwargs, rejsig=2.0, seed=seed, device="cpu", dtype=torch.float64
    )
    m64.fit(X, max_iter=10, verbose=False)
    return m64, m32


@pytest.mark.parametrize("seed", [7, 8])
@pytest.mark.parametrize("n_models", [1, 2])
def test_reject_decisions_agree_up_to_float32_rounding(n_models, seed):
    """Both backends make the same rejection decision on every sample, except
    samples within float32 rounding of the threshold.

    Exact agreement of the rejected SET is not a property of the backends: the
    float32 and float64 per-sample log-likelihoods differ by up to ~1e-1 by
    the second pass (iteration 5), and a sample that close to the threshold
    can land on either side. It held at rejsig=2.0 on seed 7 only until issue
    #333 moved the trajectories, after which one two-model sample (of 357
    rejected) flipped. So each pass is checked on its own terms
    (:func:`_rejection_violations`): the recomputed threshold is the one each
    backend applied; every disagreement lies within the band measured on that
    pass (the largest per-sample difference plus the threshold difference),
    so every decision outside it is identical; the band is narrow and holds
    few samples; and at most a few samples end up rejected by only one
    backend. Seed 8's two-model fit is the widest band measured (14
    borderline samples, one disagreement). A threshold or trajectory
    divergence fails it (``test_reject_agreement_check_catches_a_divergence``).
    """
    m64, m32 = _reject_fits(n_models, seed)
    assert len(m64.passes) == 2, "test setup: both rejection passes should run"
    assert m64.good_idx is not None
    assert m64.good_idx.numel() < 4096, "test setup: torch rejected nothing"
    assert m32.numrej == m64.numrej
    assert not _rejection_violations(m64, m32, 2.0, 2.0)


@pytest.mark.parametrize(
    "divergence", ["threshold", "trajectory"], ids=["threshold", "trajectory"]
)
def test_reject_agreement_check_catches_a_divergence(divergence):
    """The rounding allowance above cannot absorb a real divergence: a
    threshold 0.2 standard deviations off (MLX at rejsig=2.2), or a
    different trajectory (MLX from another seed, standing in for an
    orientation or update bug), each violates it."""
    if divergence == "threshold":
        m64, m32 = _reject_fits(2, 7, rejsig32=2.2)
        problems = _rejection_violations(m64, m32, 2.0, 2.2)
    else:
        m64, m32 = _reject_fits(2, 7, seed32=8)
        problems = _rejection_violations(m64, m32, 2.0, 2.0)
    assert problems, "the agreement check accepted a real divergence"


def test_reject_x_pdftype1_kurtosis_switch_matches_across_backends():
    """Regression: _choose_pdfs must see the do_reject-restricted good set
    (X_use), not the full dataset (X_t) -- verified against a float64 torch
    twin from identical seed/data/config. Before the fix this mismatched 1
    of 32 per-source pdtype decisions on this exact config (the switcher's
    kurtosis estimate over the full set, contaminated by the outliers
    do_reject had already dropped by the time the switch ran, disagreed
    with torch's estimate over the good set); after the fix the decisions
    are bit-for-bit identical.

    rejstart=2/rejint=2 fires the first rejection at iteration 2, strictly
    before kurt_start=3's first switch at iteration 3, so the switcher's
    FIRST call already sees a shrunken good set on both backends -- the
    scenario the bug needed.

    The good sets are checked with the same float32 band as
    ``test_reject_decisions_agree_up_to_float32_rounding``
    (:func:`_rejection_violations`), not an equal rejected count, which a
    single borderline sample on other hardware would break. This config's
    band is far inside those caps: over seeds 0-5, 7 and 8, and over 12
    copies of the MLX input multiplied by (1 + eps * z) at seed 2 (eps the
    float32 machine epsilon, z standard normal), it stayed within 1.2e-4 of
    the per-sample log-likelihood's spread, held at most 1 sample, and no
    decision differed. The pdtype decisions stay exact because the switch
    has margin: at seed 2 the smallest |kurtosis| any switch saw is 1.24e-2,
    560 times the largest float32-vs-float64 kurtosis difference over those
    perturbations (2.2e-5).
    """
    X = _real_data()
    kwargs: dict[str, Any] = dict(
        n_channels=NW,
        n_models=1,
        n_mix=1,
        pdftype=1,
        seed=2,
        block_size=BLOCK,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=2,
        maxrej=3,
        kurt_start=3,
        num_kurt=5,
        kurt_int=1,
        keep_best=False,
    )

    mlx_model = _RecordingMLX(**kwargs)
    mlx_model.fit(X, max_iter=15, verbose=False)

    torch_kwargs: dict[str, Any] = dict(kwargs, device="cpu", dtype=torch.float64)
    torch_model = _RecordingTorch(**torch_kwargs)
    torch_model.fit(X, max_iter=15, verbose=False)

    assert mlx_model.numrej > 0, "test setup: no rejection fired"
    assert mlx_model.n_kurt_done > 0, "test setup: the switcher never ran"
    assert not _rejection_violations(torch_model, mlx_model, 2.0, 2.0)

    assert torch_model.pdtype is not None
    mlx_pdtype = np.array(mlx_model.pdtype)
    torch_pdtype = torch_model.pdtype.numpy()
    np.testing.assert_array_equal(
        mlx_pdtype,
        torch_pdtype,
        err_msg=(
            "per-source pdtype decision diverged from the torch twin -- "
            "_choose_pdfs is likely seeing a different sample set again"
        ),
    )


# --- mir(): numeric agreement from one shared fitted state ------------------
def _torch_twin(model, sphere_np: np.ndarray):
    """A float64 AMICATorchNG holding ``model``'s exact fitted state
    (mirrors test_mlx_sharing_cross_backend.py's ``_torch_twin``)."""
    dtype = torch.float64
    ng = AMICATorchNG(
        n_channels=model.n_channels,
        n_models=model.n_models,
        n_mix=model.n_mix,
        device="cpu",
        dtype=dtype,
        block_size=model.block_size,
        seed=model.seed,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c"):
        value = np.array(getattr(model, name)).astype(np.float64)
        setattr(ng, name, torch.from_numpy(value).to(dtype))
    ng.comp_list = torch.from_numpy(np.array(model.comp_list).astype(np.int64))
    ng.sphere = torch.from_numpy(sphere_np.copy()).to(dtype)
    ng.mean = torch.from_numpy(np.array(model.mean).astype(np.float64)).to(dtype)
    ng._update_unmixing_matrices()
    return ng


def test_mir_matches_torch_on_identical_parameters():
    """From one real fitted MLX state, the float64 torch twin's mir()
    agrees with the MLX backend's own mir() to float32 precision (the
    twin runs the SAME composition -- W_fort @ sphere -- but in float64
    arithmetic)."""
    X = _real_data()
    m = AMICAMLXNG(n_channels=NW, n_models=2, n_mix=NMIX, seed=3, block_size=BLOCK)
    m.fit(X, max_iter=6, verbose=False)

    ng = _torch_twin(m, m._sphere_np)

    for idx in range(2):
        mlx_mir, mlx_var = m.mir(X, model_idx=idx)
        torch_mir, torch_var = ng.mir(X, model_idx=idx)
        # float32-native MLX vs a float64 recompute of the same
        # composition: agreement to float32 precision, not bit-exact.
        assert mlx_mir == pytest.approx(torch_mir, rel=1e-4)
        assert mlx_var == pytest.approx(torch_var, rel=1e-3)
