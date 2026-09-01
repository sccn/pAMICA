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
   (float64) and ``AMICAMLXNG`` (float32) must reject the SAME sample set.
   This is the load-bearing evidence for the phase's design decision: MLX
   reads the rejection statistic FROM the LLt stash instead of a second
   forward pass (the NumPy backend's design, pre-empting AMICATorchNG's
   open follow-up #298) -- if that were not mathematically equivalent to
   torch's own ``_sample_ll`` forward-pass statistic, the two backends
   would drift apart on which samples they drop. They do not, at a
   generous margin (measured: identical rejected-sample sets at rejsig in
   {2.0, 2.5, 3.0}, single- and 2-model), so the config below is chosen
   with clear headroom rather than right at a borderline threshold (the
   Phase 1/2 flakiness lesson).

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


# --- do_reject: the same sample set, at a config with a clear margin -------
@pytest.mark.parametrize("n_models", [1, 2])
def test_reject_same_sample_set_across_backends(n_models):
    """rejsig=2.0 (not right at any observed borderline): both backends
    reject exactly the same sample indices, from the same seed/data/
    schedule. Measured to hold at rejsig in {2.0, 2.5, 3.0} during
    development; 2.0 is kept as the assertion (generous margin, not tuned
    to pass)."""
    X = _real_data()
    kwargs: dict[str, Any] = dict(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=7,
        block_size=BLOCK,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
        keep_best=True,
    )

    mlx_model = AMICAMLXNG(**kwargs)
    mlx_model.fit(X, max_iter=10, verbose=False)

    torch_kwargs: dict[str, Any] = dict(kwargs, device="cpu", dtype=torch.float64)
    torch_model = AMICATorchNG(**torch_kwargs)
    torch_model.fit(X, max_iter=10, verbose=False)

    assert mlx_model.good_idx is not None and torch_model.good_idx is not None
    mlx_set = set(np.array(mlx_model.good_idx).tolist())
    torch_set = set(torch_model.good_idx.numpy().tolist())

    n_total = X.shape[1]
    assert len(mlx_set) < n_total, "test setup: MLX rejected nothing"
    assert len(torch_set) < n_total, "test setup: torch rejected nothing"
    assert mlx_model.numrej == torch_model.numrej
    assert mlx_set == torch_set, (
        f"rejected-sample sets diverged: {len(mlx_set ^ torch_set)} samples "
        "differ between backends"
    )


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

    mlx_model = AMICAMLXNG(**kwargs)
    mlx_model.fit(X, max_iter=15, verbose=False)

    torch_kwargs: dict[str, Any] = dict(kwargs, device="cpu", dtype=torch.float64)
    torch_model = AMICATorchNG(**torch_kwargs)
    torch_model.fit(X, max_iter=15, verbose=False)

    assert mlx_model.numrej > 0, "test setup: no rejection fired"
    assert mlx_model.n_kurt_done > 0, "test setup: the switcher never ran"
    assert mlx_model.good_idx is not None and torch_model.good_idx is not None
    assert int(mlx_model.good_idx.size) == int(torch_model.good_idx.numel()), (
        "test setup: the two backends rejected a different number of samples"
    )

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
