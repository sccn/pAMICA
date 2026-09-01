"""Exception-state consistency for mid-loop invariant raises -- PR #318 review.

Two sites in ``AMICATorchNG`` can raise ``RuntimeError`` from INSIDE
``_fit_once``'s iteration body, after that iteration's ``_update_parameters``
has already reassigned ``A``/``mu``/``beta``/``rho``/``alpha``/``gm``/``c`` to
the new iterate's values: ``_update_unmixing_matrices`` (``torch.linalg.inv``
raising ``LinAlgError`` on a singular ``A``) and ``_pinv_sphere``'s
non-finite-sphere guard (reached from ``_identify_shared_comps`` under
``share_comps``). Before this fix, neither set ``stop_reason``, which stayed
``"max_iter"`` from before the loop. The single-restart ``fit()`` path (the
default, ``n_restarts=1``) has no ``try``/``except`` around ``_fit_once``, so
the exception propagates straight to the caller -- leaving the instance
holding a genuinely inconsistent mix of new and stale parameters, but
reporting a HEALTHY ``stop_reason``. Every
``state_dict()``/``write_amica_output()`` degenerate-fit refusal keys off
``stop_reason in _DEGENERATE_STOP_REASONS``, so a caller that catches the
exception and then persists the model anyway would silently write out that
inconsistent state.

The fix sets ``self.stop_reason = restarts.ERROR_STOP_REASON`` immediately
BEFORE each raise. Mirrors
``pamica/tests/mlx_tests/test_mlx_exception_consistency.py`` exactly, adapted
to what actually reproduces a real ``torch.linalg.LinAlgError`` on a real
fitted ``A`` (a duplicated column, MLX's own trigger, does NOT reliably raise
here -- LAPACK's partial pivoting can find a tiny-but-nonzero pivot on a
merely rank-deficient real matrix and return a huge-but-finite inverse
instead, which is precisely the failure mode issue #274's guard exists for on
MLX; torch has no such guard because it does not need one for genuine
exact-zero-pivot singularity -- so this uses a zeroed column, which reliably
zeroes a pivot).

Uses the sanctioned error-injection subclass pattern (``.rules/testing.md``):
every iteration up to the injection point runs the real, unmodified fit.
Real sample EEG throughout.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pamica import restarts
from pamica.torch_impl import AMICATorchNG

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


def _real_data(n_samples: int = 4096) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD)
    return data[:, :n_samples].astype(np.float64)


class _SingularAAfterIteration(AMICATorchNG):
    """Zeroes column 1 of ``A`` at ``corrupt_at`` (a zeroed diagonal pivot
    reliably makes ``torch.linalg.inv`` raise ``LinAlgError``, unlike a
    duplicated column -- see the module docstring), so the raise fires from
    INSIDE a real ``fit()`` call. Every iteration up to ``corrupt_at`` runs
    the untouched production ``_update_unmixing_matrices`` on real data."""

    corrupt_at: int = 10**9  # effectively never, unless overridden

    def _update_unmixing_matrices(self):
        if self.iteration == self.corrupt_at:
            assert self.A is not None
            with torch.no_grad():
                self.A[:, 1] = 0.0
        super()._update_unmixing_matrices()


def test_inv_failure_leaves_the_instance_degenerate(tmp_path):
    m = _SingularAAfterIteration(n_channels=NW, n_mix=NMIX, seed=SEED, device="cpu")
    m.corrupt_at = 5

    with pytest.raises(RuntimeError, match="singular"):
        m.fit(_real_data(), max_iter=30, verbose=False)

    # The raise propagated all the way to this test (the single-restart
    # fit() path really does have no try/except), and the instance itself
    # -- not just the exception -- now reports degenerate.
    assert m.stop_reason == restarts.ERROR_STOP_REASON
    assert m.stop_reason in AMICATorchNG._DEGENERATE_STOP_REASONS

    with pytest.raises(RuntimeError, match="degenerate"):
        m.state_dict()
    with pytest.raises(RuntimeError, match="degenerate"):
        m.write_amica_output(str(tmp_path / "should_not_be_written"))
    assert not (tmp_path / "should_not_be_written").exists()


def test_inv_failure_under_multi_restart_also_leaves_it_degenerate(tmp_path):
    """The multi-restart path already caught this via its own except block
    (unchanged here); confirm the two mechanisms agree when the guard's own
    fix also fires -- redundant-but-harmless, not double-reporting anything
    a caller would notice."""
    m = _SingularAAfterIteration(
        n_channels=NW, n_mix=NMIX, seed=SEED, device="cpu", n_restarts=2
    )
    m.corrupt_at = 3
    m.fit(
        _real_data(), max_iter=30, verbose=False
    )  # must NOT raise: caught per-restart

    assert m.stop_reason == restarts.ERROR_STOP_REASON
    assert m.stop_reason in AMICATorchNG._DEGENERATE_STOP_REASONS
    with pytest.raises(RuntimeError, match="degenerate"):
        m.state_dict()


def test_pinv_sphere_nonfinite_raise_sets_degenerate_stop_reason():
    """Direct-call variant (.rules/testing.md's injection pattern): a
    non-finite sphere is itself only reachable from already-degenerate
    input, so this is exercised on hand-built state rather than driving a
    real fit into it."""
    m = AMICATorchNG(n_channels=NW, n_mix=NMIX, seed=SEED, device="cpu")
    m._preprocess(_real_data())
    m._initialize_parameters()
    m.stop_reason = "max_iter"
    assert m.sphere is not None
    with torch.no_grad():
        m.sphere[0, 0] = float("nan")
    m._sphere_pinv = None

    with pytest.raises(RuntimeError, match="non-finite values"):
        m._pinv_sphere()
    assert m.stop_reason == restarts.ERROR_STOP_REASON
