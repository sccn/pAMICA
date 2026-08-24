"""``LLt`` is written from the E-step's stashed ``logV`` (issue #157).

Both backends used to recompute the per-sample/per-model log-likelihood with a
fresh full-dataset forward pass at write time. They now keep what the training
E-step already computed, exactly as the reference does: ``modloglik``/``loglik``
are allocated once (amica15.f90:2619-2620), filled by every E-step
(amica15.f90:1406-1411) and dumped verbatim by ``write_output``
(amica15.f90:2338-2343).

The user-visible consequence, adopted deliberately (2026-08-23), is the
reference's own one-M-step staleness. Fortran's loop is

    get_updates_and_likelihood  (amica15.f90:996)   <- fills modloglik
    update_params               (amica15.f90:1122)  <- moves W/A
    write_output                (amica15.f90:1126, 1146)

so the ``LLt`` on disk belongs to the parameters as they stood *before* the
M-step whose ``W``/``A`` sit beside it. The invariant that pins this down, and
that the committed reference output satisfies bit for bit, is

    Lt.sum() / (n_good_samples * nw) == LL[-1]

with exactly one exception, itself reference-faithful and pinned here as
behavior: a ``do_reject`` fit that rejects on the same iteration as the write
normalizes ``LL(iter)`` before ``reject_data`` shrinks the good count
(amica15.f90:1770 precedes :1138/:2252), so a small residual remains until the
next E-step re-normalizes.

Real sample EEG only, and the Fortran side of the headline test is the
committed ``sample_data/amicaout`` -- output of the reference binary itself, so
the comparison needs no binary at run time (an opt-in fresh-binary variant is
gated on ``AMICA_RUN_FORTRAN=1``).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from pamica import AMICA_NumPy
from pamica.numpy_impl.data import load_data_file
from pamica.numpy_impl.load import loadmodout
from pamica.torch_impl.core import AMICATorchNG

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
_FDT = SAMPLE_DIR / "eeglab_data.fdt"
_REFERENCE_OUT = SAMPLE_DIR / "amicaout"

pytestmark = pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")

NW = 32
FIELD = 30504
_BLOCK = 1024
_N = 4096


def _real_data(n_samples: int = _N) -> np.ndarray:
    data = load_data_file(str(_FDT), NW, FIELD, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    return _real_data()


def _torch_fit(X, n_models=1, max_iter=8, seed=42, **kwargs):
    m = AMICATorchNG(
        n_channels=X.shape[0],
        n_models=n_models,
        n_mix=3,
        seed=seed,
        device="cpu",
        block_size=_BLOCK,
        **kwargs,
    )
    m.fit(X, max_iter=max_iter, verbose=False)
    return m


def _numpy_fit(X, n_models=1, max_iter=8, **kwargs):
    m = AMICA_NumPy(
        num_models=n_models,
        num_mix=3,
        max_iter=max_iter,
        seed=42,
        use_tqdm=False,
        do_opt_block=False,
        block_size=_BLOCK,
        **kwargs,
    )
    m.fit(X)
    return m


def _llt_invariant(Lt: np.ndarray, n_good: int, nw: int) -> float:
    """The mean per-sample-per-channel log-likelihood Fortran reports as LL.

    Compared against the reported LL with a tight tolerance rather than for bit
    equality: both sides sum the same per-sample values, but the reported LL is
    accumulated block by block during the E-step while this re-sums the whole
    record at once, so they can differ by a summation ULP. The quantity being
    discriminated -- one M-step of movement -- is ~1e-4 here, eight orders of
    magnitude above that, so the tests below pair the tolerance with an
    explicit check that a neighbouring iterate is far outside it.
    """
    return float(Lt.sum()) / (n_good * nw)


_LL_TOL = 1e-12


# --- the invariant, on the reference binary's own output --------------------
@pytest.mark.skipif(
    not (_REFERENCE_OUT / "LLt").exists(), reason="reference amicaout missing"
)
def test_fortran_reference_llt_is_the_last_estep():
    """The committed reference output satisfies the invariant.

    ``sample_data/amicaout`` was produced by the ``amica15mac`` binary. Its
    ``LLt`` sums (over the whole record, divided by ``N*nw``) to the last entry
    of its own ``LL`` trajectory -- i.e. it is the E-step that produced that LL,
    taken before the final ``update_params``. On the machine this was written
    the residual is exactly 0.0; the assertion allows a summation ULP so it
    cannot fail on a different BLAS, and the neighbouring-iterate check below
    shows the tolerance is nowhere near wide enough to match the wrong entry.

    This is the evidence for the write convention issue #157 adopts, measured
    on the reference's own bytes rather than argued from its source.
    """
    out = loadmodout(_REFERENCE_OUT)
    assert out.Lt is not None and out.LL is not None
    nw = out.W.shape[0]
    inv = _llt_invariant(out.Lt, out.Lt.size, nw)
    assert abs(inv - out.LL[-1]) <= _LL_TOL
    # An adjacent iterate is orders of magnitude away, so the match above
    # identifies one specific E-step and not "any recent LL".
    assert abs(out.LL[-1] - out.LL[-2]) > 1e4 * _LL_TOL


@pytest.mark.parametrize("n_models", [1, 2])
def test_written_llt_matches_the_fortran_write_convention(
    real_data, tmp_path, n_models
):
    """Both backends' on-disk LLt satisfies the reference's own invariant.

    Read back through the same ``loadmodout`` the reference fixture is read
    with, so this compares the two implementations' write conventions against
    the binary's and not just their in-memory bookkeeping.
    """
    tm = _torch_fit(real_data, n_models=n_models)
    tdir = tmp_path / f"torch{n_models}"
    tm.write_amica_output(tdir)
    tout = loadmodout(tdir)
    assert tout.Lt is not None
    assert abs(_llt_invariant(tout.Lt, tout.Lt.size, NW) - tout.LL[-1]) <= _LL_TOL
    assert abs(tout.LL[-1] - tout.LL[-2]) > 1e4 * _LL_TOL

    nm = _numpy_fit(
        real_data, n_models=n_models, outdir=str(tmp_path / f"np{n_models}")
    )
    nout = loadmodout(tmp_path / f"np{n_models}")
    assert nout.Lt is not None
    assert (
        abs(_llt_invariant(nout.Lt, nout.Lt.size, nm.data_dim) - nout.LL[-1]) <= _LL_TOL
    )
    assert abs(nout.LL[-1] - nout.LL[-2]) > 1e4 * _LL_TOL


@pytest.mark.skipif(
    os.environ.get("AMICA_RUN_FORTRAN") != "1",
    reason="opt-in Fortran-binary integration test (set AMICA_RUN_FORTRAN=1)",
)
def test_native_binary_llt_is_the_last_estep(tmp_path):
    """The same check against a freshly-run binary, not the committed fixture.

    Opt-in (``AMICA_RUN_FORTRAN=1``) like the other binary-driven tests, so the
    default suite does not depend on a runnable ``amica15mac``. Uses the whole
    record, as the native-engine tests do: the binary NaNs on a short slice at
    its default 512 block size (issue #292), which would test nothing here.
    """
    from pamica import AMICANative

    full = load_data_file(str(_FDT), NW, FIELD, dtype=np.float32).astype(np.float64)
    eng = AMICANative(
        binary=os.environ.get("PAMICA_NATIVE_BINARY"),
        n_models=1,
        n_mix=3,
        max_iter=10,
        threads=2,
    )
    eng.fit(full)
    out = eng.output_
    assert out is not None and out.Lt is not None and out.LL is not None
    nw = out.W.shape[0]
    assert abs(_llt_invariant(out.Lt, out.Lt.size, nw) - out.LL[-1]) <= _LL_TOL
    assert abs(out.LL[-1] - out.LL[-2]) > 1e4 * _LL_TOL


# --- the same invariant in memory, per backend ------------------------------
def test_torch_stashed_llt_is_the_estep_behind_final_ll(real_data):
    """``final_ll_`` is the total of the stashed per-sample LL."""
    m = _torch_fit(real_data, n_models=2, keep_best=False)
    assert m._llt_lt is not None and m._llt_lht is not None
    assert abs(_llt_invariant(m._llt_lt, m._llt_lt.size, NW) - m.final_ll_) <= _LL_TOL
    assert abs(m.ll_history[-1] - m.ll_history[-2]) > 1e4 * _LL_TOL
    # Lt really is the model-wise log-sum-exp of Lht, not a separately drifting
    # quantity (the definitional LLt identity).
    np.testing.assert_allclose(
        m._llt_lt, np.log(np.exp(m._llt_lht).sum(axis=0)), rtol=0, atol=1e-11
    )


def test_numpy_stashed_llt_is_the_estep_behind_reported_ll(real_data, tmp_path):
    m = _numpy_fit(real_data, n_models=2, outdir=str(tmp_path / "out"))
    Lht, Lt = m._llt_arrays()
    assert Lht is not None and Lt is not None
    inv = _llt_invariant(Lt, m.num_good_samples, m.data_dim)
    assert abs(inv - m.ll[-1]) <= _LL_TOL
    assert abs(m.ll[-1] - m.ll[-2]) > 1e4 * _LL_TOL
    np.testing.assert_allclose(Lt, np.log(np.exp(Lht).sum(axis=0)), rtol=0, atol=1e-11)


def test_stashed_llt_is_one_m_step_behind_the_written_parameters(real_data):
    """The staleness is real and measurable, not a claim about nothing.

    A fit of N iterations stashes the E-step of a fit of N-1 iterations'
    parameters -- bit for bit -- and that differs materially from the E-step of
    the parameters the N-iteration fit returns. This is the reference ordering
    (amica15.f90:996 -> 1122 -> 1146) and is what makes pamica's on-disk LLt
    comparable with the binary's.
    """
    m8 = _torch_fit(real_data, n_models=2, max_iter=8, keep_best=False)
    m7 = _torch_fit(real_data, n_models=2, max_iter=7, keep_best=False)
    assert len(m8.ll_history) == 8 and len(m7.ll_history) == 7

    np.testing.assert_array_equal(m7.model_loglik(real_data), m8._llt_lht)
    assert np.abs(m8.model_loglik(real_data) - m8._llt_lht).max() > 1e-6


# --- cross-backend agreement (.rules/backend_parity.md) ---------------------
def test_torch_and_numpy_stash_the_same_llt_on_matched_state(real_data, tmp_path):
    """One state, one sphered data matrix, two E-steps: the stashes agree.

    Both backends are driven from the parameters of a single NumPy fit, so the
    only thing under test is what each one stashes. They run the same float64
    arithmetic and differ only in BLAS association order, hence ~1e-16 in
    practice; 1e-10 is the tolerance with margin.
    """
    npm = _numpy_fit(real_data, n_models=2, max_iter=4, outdir=str(tmp_path / "out"))

    ng = AMICATorchNG(
        n_channels=npm.data_dim,
        n_models=npm.num_models,
        n_mix=npm.num_mix,
        device="cpu",
        block_size=_BLOCK,
        seed=7,
        pdftype=0,
    )
    ng._initialize_parameters()
    for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c", "sphere"):
        setattr(ng, name, torch.from_numpy(np.asarray(getattr(npm, name)).copy()))
    ng.comp_list = torch.from_numpy(npm.comp_list.copy())
    # The sphere log-det enters logV directly (unlike the responsibilities,
    # which are shift-invariant), so it must be matched too or the two stashes
    # differ by a constant.
    ng.sldet = npm.sldet
    ng._update_unmixing_matrices()

    X_t = torch.from_numpy(npm.data)  # the NumPy-sphered data, shared by both
    n_samples = X_t.shape[1]
    ng._llt_logv = torch.zeros((n_samples, ng.n_models), dtype=ng.dtype)
    ng._llt_ll = torch.zeros(n_samples, dtype=ng.dtype)
    ng._accumulate_blocks(X_t, stash_llt=True)

    npm._get_updates_and_likelihood()  # refill the NumPy stash at this state

    np.testing.assert_allclose(ng._llt_logv.numpy(), npm._llt_logv, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ng._llt_ll.numpy(), npm._llt_ll, rtol=0, atol=1e-10)
    # Not vacuous: these are real log-likelihoods, not a pair of zero buffers.
    assert np.abs(npm._llt_ll).min() > 0.0


# --- do_reject: the one reference-faithful break in the invariant -----------
def _reject_kwargs(rejstart):
    """Fire exactly one rejection, on iteration ``rejstart`` (0-indexed).

    ``rejint=3`` keeps the modulo arm of the schedule from firing earlier (both
    backends clamp ``max(1, iter - rejstart)``, so a ``rejint`` of 1 would
    reject on every iteration from the start), and ``maxrej=1`` caps it at one
    pass so the iteration under test is unambiguous.
    """
    return dict(do_reject=True, rejstart=rejstart, rejint=3, maxrej=1, rejsig=3.0)


def test_llt_invariant_breaks_when_rejection_fires_on_the_last_iteration(
    real_data, tmp_path
):
    """A rejection on the fit's own last iteration leaves a bounded residual.

    ``ll`` is normalized over the good set as it stood BEFORE that iteration's
    rejection, and ``_reject_outliers`` then zeroes the dropped samples' stash
    entries, so the two sides of the invariant stop counting the same samples.
    This is Fortran's ordering, not a pamica defect: the reference computes
    ``LL(iter) = LLtmp2/dble(numgoodsum*nw)`` (amica15.f90:1770) before
    ``reject_data`` (amica15.f90:1138) shrinks ``numgoodsum``
    (amica15.f90:2252) and zeroes the rejected ``modloglik``/``loglik``
    (amica15.f90:2232-2234), and the binary shows the same residual on the same
    schedule.

    Pinned as behavior rather than silenced, on both backends, and paired with
    the control below: one more iteration after the rejection re-normalizes
    over the shrunk good set and the equality returns exactly. The residual's
    size depends on how many samples that one pass drops, so it is bounded
    rather than pinned to a value -- what is asserted is that it is far above
    the summation tolerance and far below anything resembling a blow-up.
    """
    tm = _torch_fit(real_data, n_models=1, max_iter=6, **_reject_kwargs(5))
    assert len(tm.ll_history) == 6 and tm.numrej == 1
    assert tm.good_idx is not None and int(tm.good_idx.numel()) < real_data.shape[1]
    assert tm._llt_lt is not None
    n_good = int(tm.good_idx.numel())
    # Rejected samples carry Fortran's zero sentinel, which is what makes the
    # two sides count different sets.
    assert int((tm._llt_lt == 0.0).sum()) == real_data.shape[1] - n_good
    residual = abs(_llt_invariant(tm._llt_lt, n_good, NW) - tm.final_ll_)
    assert 1e4 * _LL_TOL < residual < 1.0

    nm = _numpy_fit(
        real_data,
        n_models=1,
        max_iter=6,
        outdir=str(tmp_path / "rej"),
        **_reject_kwargs(5),
    )
    assert len(nm.ll) == 6 and nm.numrej == 1
    _, n_lt = nm._llt_arrays()
    assert n_lt is not None and nm.num_good_samples < real_data.shape[1]
    n_residual = abs(_llt_invariant(n_lt, nm.num_good_samples, nm.data_dim) - nm.ll[-1])
    assert 1e4 * _LL_TOL < n_residual < 1.0


def test_llt_invariant_returns_one_iteration_after_a_rejection(real_data, tmp_path):
    """The control: the same rejection, one iteration earlier, and it holds.

    Confines the exception above to exactly the case named -- a rejection on
    the fit's own last iteration -- rather than to ``do_reject`` in general.
    The E-step after a rejection re-normalizes over the shrunk good set, and
    the rejected samples contribute 0 to both sides, so the equality is exact
    again.
    """
    tm = _torch_fit(real_data, n_models=1, max_iter=7, **_reject_kwargs(5))
    assert len(tm.ll_history) == 7 and tm.numrej == 1
    assert tm.good_idx is not None and tm._llt_lt is not None
    n_good = int(tm.good_idx.numel())
    assert n_good < real_data.shape[1]  # a rejection really did fire
    assert abs(_llt_invariant(tm._llt_lt, n_good, NW) - tm.final_ll_) <= _LL_TOL

    nm = _numpy_fit(
        real_data,
        n_models=1,
        max_iter=7,
        outdir=str(tmp_path / "rej2"),
        **_reject_kwargs(5),
    )
    assert len(nm.ll) == 7 and nm.numrej == 1
    _, n_lt = nm._llt_arrays()
    assert n_lt is not None and nm.num_good_samples < real_data.shape[1]
    assert (
        abs(_llt_invariant(n_lt, nm.num_good_samples, nm.data_dim) - nm.ll[-1])
        <= _LL_TOL
    )


# --- keep_best (issue #51) --------------------------------------------------
def test_keep_best_restore_rolls_the_llt_stash_back(real_data, tmp_path):
    """A best-iterate restore takes the stashed LLt back with the parameters.

    ``_snapshot_params`` is taken right after the E-step that measured
    ``best_ll`` and before that iteration's M-step, so the restored parameters
    and the restored stash come from the same point in the loop: under a
    restore there is no staleness at all, and ``model_loglik`` on the returned
    model reproduces the exported LLt exactly. Without the rollback the stash
    would still hold the discarded last iterate's values, which the second
    assertion rules out.
    """
    m = _torch_fit(
        real_data,
        n_models=2,
        max_iter=60,
        seed=0,
        do_newton=True,
        newt_start=1,
        lrate=0.5,
    )
    if m.stop_reason in AMICATorchNG._DEGENERATE_STOP_REASONS:
        pytest.skip("aggressive run ended degenerate; not the case under test")
    if np.isclose(m.ll_history[-1], m.final_ll_):
        pytest.skip("run was monotone; keep_best restore did not fire")

    assert m._llt_lt is not None and m._llt_lht is not None
    inv = _llt_invariant(m._llt_lt, m._llt_lt.size, NW)
    # The stash is the restored iterate's E-step ...
    assert abs(inv - m.final_ll_) <= _LL_TOL
    # ... and demonstrably not the discarded last iterate's.
    assert abs(inv - m.ll_history[-1]) > 1e4 * _LL_TOL
    # Snapshot taken pre-M-step => restored params and restored LLt coincide.
    np.testing.assert_array_equal(m.model_loglik(real_data), m._llt_lht)

    # And that is what reaches disk.
    outdir = tmp_path / "amicaout"
    m.write_amica_output(outdir)
    out = loadmodout(outdir)
    assert out.Lt is not None
    assert out.LL[-1] == m.final_ll_
    assert abs(_llt_invariant(out.Lt, out.Lt.size, NW) - out.LL[-1]) <= _LL_TOL


# --- the point of the issue: no forward pass at write time ------------------
def test_torch_write_runs_no_forward_pass(real_data, tmp_path, monkeypatch):
    """``write_amica_output`` performs zero E-step forward passes.

    A counting passthrough around the real ``_forward`` (every line of it still
    runs; nothing is stubbed out) shows the write path consumes the stash
    instead of walking the data again.
    """
    m = _torch_fit(real_data, n_models=2)
    calls = []
    real_forward = m._forward
    monkeypatch.setattr(
        m, "_forward", lambda X: (calls.append(X.shape[1]), real_forward(X))[1]
    )
    m.write_amica_output(tmp_path / "amicaout")
    assert calls == []
    assert (tmp_path / "amicaout" / "LLt").exists()


def test_numpy_checkpoint_runs_no_forward_pass(real_data, tmp_path, monkeypatch):
    """The NumPy ``writestep`` checkpoint performs zero forward passes either.

    This was the expensive case: ``_write_results`` runs at every checkpoint
    during training, so before this change a fit paid one extra full-dataset
    pass per checkpoint.
    """
    m = _numpy_fit(real_data, n_models=2, outdir=str(tmp_path / "out"))
    calls = []
    real_forward = m._forward_block
    monkeypatch.setattr(
        m,
        "_forward_block",
        lambda X: (calls.append(X.shape[1]), real_forward(X))[1],
    )
    m._write_results()
    assert calls == []
    assert (tmp_path / "out" / "LLt").exists()


def test_llt_is_omitted_when_no_estep_ran(real_data, tmp_path):
    """A model that never completed an E-step writes no LLt, rather than an
    all-zero one that ``load_rej`` would read as "every sample rejected"."""
    m = AMICATorchNG(
        n_channels=NW, n_models=1, n_mix=3, seed=42, device="cpu", block_size=_BLOCK
    )
    m.fit(real_data, max_iter=0, verbose=False)
    assert m._llt_lht is None and m._llt_lt is None
    m.write_amica_output(tmp_path / "amicaout")
    assert not (tmp_path / "amicaout" / "LLt").exists()
    assert (tmp_path / "amicaout" / "W").exists()
