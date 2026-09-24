"""Exception-state consistency for mid-loop invariant raises -- PR #318 review.

Three sites in ``AMICAMLXNG`` can raise ``RuntimeError`` from INSIDE
``_fit_once``'s iteration body, after that iteration's ``_update_parameters``
has already reassigned ``A``/``mu``/``beta``/``rho``/``alpha``/``gm``/``c`` to
the new iterate's values: the issue #274 condition-number guard (in
``_update_unmixing_matrices``), ``_choose_pdfs``'s pdtype-invariant guard, and
``_pinv_sphere``'s non-finite-sphere guard (reached from
``_identify_shared_comps`` under ``share_comps``). Before this fix, none of
them set ``stop_reason``, which stayed ``"max_iter"`` from before the loop.
The single-restart ``fit()`` path (the default, ``n_restarts=1``) has no
``try``/``except`` around ``_fit_once``, so the exception propagates straight
to the caller -- leaving the instance holding a genuinely inconsistent mix of
new and stale parameters, but reporting a HEALTHY ``stop_reason``. Every
``state_dict()``/``write_amica_output()`` degenerate-fit refusal keys off
``stop_reason in _DEGENERATE_STOP_REASONS``, so a caller that catches the
exception (e.g. to try a different config) and then persists the model anyway
would silently write out that inconsistent state.

The fix sets ``self.stop_reason = restarts.ERROR_STOP_REASON`` immediately
BEFORE each of these three raises, so the instance is always left in a state
every downstream guard recognizes as degenerate, regardless of whether the
caller catches the exception.

Uses the sanctioned error-injection subclass pattern (``.rules/testing.md``):
a subclass that runs the real, unmodified fit for N genuine iterations, then
corrupts state to force the target guard's raise on a specific later
iteration -- exercising the ACTUAL uncaught-exception propagation path through
``fit()``'s single-restart branch, not a standalone method call on
hand-manipulated state (that mechanics-only coverage is
``test_mlx_inv_guard.py``'s job). Real sample EEG throughout.
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica import restarts  # noqa: E402
from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
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


class _SingularAAfterIteration(AMICAMLXNG):
    """Forces model 0's ``A`` to become exactly singular (a duplicated
    column) at ``corrupt_at``, so the issue #274 condition-number guard's
    raise fires from INSIDE a real ``fit()`` call. Every iteration up to
    ``corrupt_at`` runs the untouched production ``_update_unmixing_matrices``
    on real data."""

    corrupt_at: int = 10**9  # effectively never, unless overridden

    def _update_unmixing_matrices(self):
        if self.iteration == self.corrupt_at:
            a_np = np.array(self.A)
            a_np[:, 1] = a_np[:, 0]  # exact duplicate column -> singular
            self.A = mx.array(a_np)
        super()._update_unmixing_matrices()


def test_274_guard_raise_leaves_the_instance_degenerate(tmp_path):
    m = _SingularAAfterIteration(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    m.corrupt_at = 5

    with pytest.raises(RuntimeError, match="Singular unmixing matrix"):
        m.fit(_real_data(), max_iter=30, verbose=False)

    # The raise propagated all the way to this test (the single-restart
    # fit() path really does have no try/except), and the instance itself
    # -- not just the exception -- now reports degenerate.
    assert m.stop_reason == restarts.ERROR_STOP_REASON
    assert m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS

    with pytest.raises(RuntimeError, match="degenerate"):
        m.state_dict()
    with pytest.raises(RuntimeError, match="degenerate"):
        m.write_amica_output(tmp_path / "should_not_be_written")
    assert not (tmp_path / "should_not_be_written").exists()


def test_274_guard_raise_under_multi_restart_also_leaves_it_degenerate(tmp_path):
    """The multi-restart path already caught this via its own except block
    (unchanged here); confirm the two mechanisms agree when the guard's own
    fix also fires -- redundant-but-harmless, not double-reporting anything
    a caller would notice."""
    m = _SingularAAfterIteration(
        n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK, n_restarts=2
    )
    m.corrupt_at = 3
    m.fit(
        _real_data(), max_iter=30, verbose=False
    )  # must NOT raise: caught per-restart

    assert m.stop_reason == restarts.ERROR_STOP_REASON
    assert m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS
    with pytest.raises(RuntimeError, match="degenerate"):
        m.state_dict()


def test_choose_pdfs_invariant_raise_sets_degenerate_stop_reason(monkeypatch):
    """Direct-call variant (.rules/testing.md's injection pattern): the
    "code outside {1, 4}" branch is "currently unreachable" from real
    kurtosis data (per its own comment -- ``_pdtype_from_kurtosis`` can only
    ever emit 1 or 4), so it is exercised by monkeypatching that one
    decision function on a real, genuinely preprocessed/initialized model,
    then calling the real (unmodified) ``_choose_pdfs`` on real data."""
    m = AMICAMLXNG(
        n_channels=NW, n_mix=1, pdftype=1, seed=SEED, block_size=BLOCK, keep_best=False
    )
    x_t = m._preprocess(_real_data())
    m._initialize_parameters()
    m.iteration = 4
    m.stop_reason = "max_iter"
    assert m.pdtype is not None
    invalid_shape = np.array(m.pdtype).shape

    monkeypatch.setattr(
        m, "_pdtype_from_kurtosis", lambda kurt, nsub: np.full(invalid_shape, 99)
    )
    with pytest.raises(RuntimeError, match="adaptive-switcher invariant violated"):
        m._choose_pdfs(x_t)
    assert m.stop_reason == restarts.ERROR_STOP_REASON


def test_pinv_sphere_nonfinite_raise_sets_degenerate_stop_reason():
    """Direct-call variant: a non-finite sphere is itself only reachable from
    already-degenerate input, so this is exercised on hand-built state
    (.rules/testing.md's direct-call injection variant), matching how
    test_mlx_reject.py's ``_reject_outliers`` all-rejected branch is tested."""
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=BLOCK)
    m._preprocess(_real_data())
    m.stop_reason = "max_iter"
    assert m._sphere_np is not None
    m._sphere_np = m._sphere_np.copy()
    m._sphere_np[0, 0] = np.nan
    m._sphere_pinv = None

    with pytest.raises(RuntimeError, match="non-finite values"):
        m._pinv_sphere()
    assert m.stop_reason == restarts.ERROR_STOP_REASON
