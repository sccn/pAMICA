"""LLt stash on the MLX backend -- issue #157, epic #278 Phase 3/#289 (porting
``AMICATorchNG``'s issue #157/#297 semantics and ``pamica/tests/test_llt_stash.py``).

The E-step's per-sample log-likelihood is stashed block by block as
``_accumulate_blocks(stash_llt=True)`` runs, never recomputed by a second
forward pass at write time. The invariant that pins this down is

    Lt.sum() / (n_good_samples * n_channels) == final_ll_

Apple-Silicon only, real sample EEG (no synthetic/mock), same module guards as
``test_mlx_backend.py``. Every numeric comparison here runs INSIDE one test
(same process, same machine): MLX float32 is bit-reproducible run-to-run on
one machine but NOT across GPU models (``test_mlx_transform.py``'s
``_NOOP_PIN_*`` module comment), so no cross-machine literal is recorded here
-- values are always compared against a quantity computed in the same test.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
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


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :4096]


def _model(**kwargs: Any) -> AMICAMLXNG:
    params: dict[str, Any] = dict(n_channels=NW, n_mix=NMIX, block_size=BLOCK)
    params.update(kwargs)
    return AMICAMLXNG(**params)


def _llt_invariant(Lt: np.ndarray, n_good: int, nw: int) -> float:
    """The mean per-sample-per-channel log-likelihood, matching what
    ``final_ll_`` reports -- see ``pamica/tests/test_llt_stash.py``'s
    identically-named helper."""
    return float(Lt.sum()) / (n_good * nw)


# float32 accumulation across ~4000 samples * up to 2 models: measured
# residuals for the invariant are ~1e-9 to 1e-7 in practice, so 1e-5 is
# generous headroom while staying well below a real trajectory step (~1e-2
# or more between iterations on this data, and ~1.7e-4 for the smallest
# genuine keep_best overshoot this suite forces -- see
# _NOT_VACUOUS_THRESHOLD below, which is deliberately a SEPARATE, looser
# bound for distinguishing "the invariant held" from "the two sides
# happened to coincide").
_LL_TOL = 1e-5
_NOT_VACUOUS_THRESHOLD = 5e-5


def test_llt_invariant_holds_for_a_default_fit(real_data):
    m = _model(n_models=2, seed=42, keep_best=False)
    m.fit(real_data, max_iter=8, verbose=False)
    assert m._llt_lt is not None and m._llt_lht is not None and m.final_ll_ is not None
    assert m._llt_lht.shape == (2, real_data.shape[1])
    inv = _llt_invariant(m._llt_lt, m._llt_lt.size, NW)
    assert abs(inv - m.final_ll_) <= _LL_TOL
    # Not a vacuous match against a neighboring iterate.
    assert abs(m.ll_history[-1] - m.ll_history[-2]) > _NOT_VACUOUS_THRESHOLD


def test_llt_is_the_logsumexp_of_lht(real_data):
    """Lt really is the model-wise log-sum-exp of Lht, not a separately
    drifting quantity (the definitional LLt identity)."""
    m = _model(n_models=2, seed=42, keep_best=False)
    m.fit(real_data, max_iter=6, verbose=False)
    assert m._llt_lt is not None and m._llt_lht is not None
    np.testing.assert_allclose(
        m._llt_lt,
        np.log(np.exp(m._llt_lht.astype(np.float64)).sum(axis=0)),
        rtol=0,
        atol=1e-4,
    )


def test_llt_stashed_at_one_model_too(real_data):
    m = _model(n_models=1, seed=1, keep_best=False)
    m.fit(real_data, max_iter=5, verbose=False)
    assert m._llt_lht is not None and m._llt_lt is not None and m.final_ll_ is not None
    assert m._llt_lht.shape == (1, real_data.shape[1])
    inv = _llt_invariant(m._llt_lt, m._llt_lt.size, NW)
    assert abs(inv - m.final_ll_) <= _LL_TOL


def test_stashed_llt_is_one_m_step_behind_the_written_parameters(real_data):
    """A fit of N iterations stashes the E-step of a fit of N-1 iterations'
    parameters -- bit for bit -- matching the reference ordering (issue
    #157) and ``pamica/tests/test_llt_stash.py``'s torch-backend pin."""
    m8 = _model(n_models=2, seed=3, keep_best=False)
    m8.fit(real_data, max_iter=8, verbose=False)
    m7 = _model(n_models=2, seed=3, keep_best=False)
    m7.fit(real_data, max_iter=7, verbose=False)
    assert len(m8.ll_history) == 8 and len(m7.ll_history) == 7
    assert m8._llt_lht is not None

    lht_pre = m7.model_loglik(real_data)
    # float32-vs-float32 recompute through model_loglik takes a slightly
    # different path than the stash (see test_mlx_scoring.py), so this is a
    # tolerance match, not bit equality (unlike the torch backend's version).
    np.testing.assert_allclose(lht_pre, m8._llt_lht, rtol=0, atol=1e-3)
    # Not vacuous: the M-step in between genuinely moves the per-sample LL.
    assert np.abs(m8.model_loglik(real_data) - m8._llt_lht).max() > 1e-3


def test_llt_stash_is_none_before_any_estep_ran():
    """The LLt stash starts unset and stays unset until an E-step actually
    runs, rather than an all-zero one that ``load_rej`` would misread as
    "every sample rejected".

    Previously exercised via ``fit(max_iter=0)`` ("a model that never
    completed an E-step"), but PR #318 review item 5 made ``max_iter=0`` an
    upfront ``ValueError`` (``max_iter must be >= 1``) rather than a
    completed zero-iteration fit, so that path is no longer reachable
    through :meth:`fit`. The construction-time invariant checked here still
    holds; ``write_amica_output``'s behavior on a model actually left with
    no stash (but otherwise fitted) is covered separately by
    ``test_from_state_dict_model_warns_and_omits_llt`` (in
    ``test_mlx_export.py``), whose ``state_dict`` round trip drops the
    stash on an otherwise-usable model."""
    m = _model(n_models=1, seed=42)
    assert m._llt_lht is None and m._llt_lt is None


# --- keep_best (issue #51) interaction ---------------------------------
# Reuses the aggressive-Newton recipe test_mlx_keepbest.py measured to
# genuinely overshoot on this backend (module docstring there): n_models=2,
# seed=0, block_size=1024, do_newton=True, newt_start=1, lrate=0.5,
# use_min_dll=True, min_dll=1e-4, maxincs=2, use_grad_norm=False, max_iter=60.
_FORCED_RESTORE_KWARGS: dict[str, Any] = dict(
    n_models=2,
    n_mix=NMIX,
    seed=0,
    block_size=BLOCK,
    do_newton=True,
    newt_start=1,
    lrate=0.5,
    use_min_dll=True,
    min_dll=1e-4,
    maxincs=2,
    use_grad_norm=False,
)


def test_keep_best_restore_rolls_the_llt_stash_back(real_data):
    """A best-iterate restore takes the stashed LLt back with the
    parameters: the exported LLt belongs to the RESTORED iterate, not the
    discarded last one (port of ``pamica/tests/test_llt_stash.py``'s
    torch-backend pin of the same name)."""
    m = AMICAMLXNG(n_channels=NW, **_FORCED_RESTORE_KWARGS)
    m.fit(real_data, max_iter=60, verbose=False)
    if m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS:
        pytest.skip("aggressive run ended degenerate; not the case under test")
    assert m.final_ll_ is not None
    if np.isclose(m.ll_history[-1], m.final_ll_):
        pytest.skip("run was monotone; keep_best restore did not fire")

    assert m._llt_lt is not None and m._llt_lht is not None
    inv = _llt_invariant(m._llt_lt, m._llt_lt.size, NW)
    # The stash is the restored iterate's E-step ...
    assert abs(inv - m.final_ll_) <= _LL_TOL
    # ... and demonstrably not the discarded last iterate's.
    assert abs(inv - m.ll_history[-1]) > _NOT_VACUOUS_THRESHOLD
