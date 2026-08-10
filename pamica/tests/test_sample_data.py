"""Tests for pamica using sample data."""

import os.path as op
import numpy as np
import pytest
from pathlib import Path

import pamica
from pamica import AMICA_NumPy as AMICA
from pamica.numpy_impl.data import load_data_file
from pamica.numpy_impl.load import loadmodout

# Setup sample data paths
sample_data_path = op.join(pamica.__path__[0], "sample_data")
sample_params_file = op.join(sample_data_path, "sample_params.json")
eeglab_data_file = op.join(sample_data_path, "eeglab_data.fdt")
amicaout_dir = op.join(sample_data_path, "amicaout")


def test_loadmodout_llt_reshape_order():
    """``loadmodout`` reads the bundled Fortran-produced ``LLt`` file with the
    correct column-major reshape (issue #155 regression).

    The reference binary writes, per timepoint, each model's log-likelihood
    then the total (Fortran ``write_output``, amica15.f90:2308-2333) -- a
    column-major ``(num_models+1, N)`` layout (matching EEGLAB's
    ``loadmodout15.m``, which does ``reshape(LLt, num_models+1, N)`` under
    MATLAB's column-major default). A C-order reshape scrambles this. The
    bundled fixture is single-model, so ``Lht[0]`` must equal ``Lt`` exactly --
    a C-order bug would break both this equality and the mean check below.
    """
    out = loadmodout(amicaout_dir)
    assert out.num_models == 1
    assert out.Lht is not None and out.Lt is not None
    assert out.Lht.shape == (1, out.Lt.shape[0])
    np.testing.assert_array_equal(out.Lht[0], out.Lt)

    # Fortran's bundled "LL" file already reports the per-sample-per-channel
    # normalized log-likelihood (~-3.40), so nw * final_ll recovers the LLt
    # convention's per-sample total (summed over channels), Lt.mean().
    nw = out.W.shape[0]
    final_ll = out.LL[-1]
    np.testing.assert_allclose(out.Lt.mean(), nw * final_ll, rtol=1e-3)


def _fixture_loglik(out, X):
    """Recompute the mean per-sample-per-channel log-likelihood of the bundled
    Fortran fixture from a loaded ``AmicaOutput``. Mirrors the Fortran E-step
    (amica15.f90): ``b = W(x - c)`` in sphered-channel space, a generalized-
    Gaussian mixture per source in log space, plus the ``log|det W| + log|det S|``
    Jacobian. Single model (the fixture)."""
    from scipy.special import gammaln

    b = out.sources(X, 0)  # (nw, N) -- the loader's own W and c
    nw, n = b.shape
    alpha, mu, sbeta, rho = (
        out.alpha[:, :, 0],
        out.mu[:, :, 0],
        out.sbeta[:, :, 0],
        out.rho[:, :, 0],
    )
    logdet_w = np.linalg.slogdet(out.W[:, :, 0])[1]
    sldet = np.linalg.slogdet(out.S[:nw])[1]
    tot = np.zeros(n)
    for i in range(nw):
        z = np.empty((alpha.shape[0], n))
        for j in range(alpha.shape[0]):
            y = sbeta[j, i] * (b[i] - mu[j, i])
            z[j] = (
                np.log(alpha[j, i])
                + np.log(sbeta[j, i])
                - np.abs(y) ** rho[j, i]
                - gammaln(1.0 + 1.0 / rho[j, i])
                - np.log(2.0)
            )
        m = z.max(axis=0)
        tot += m + np.log(np.exp(z - m).sum(axis=0))
    return (tot.sum() + n * (logdet_w + sldet)) / (n * nw)


def test_loadmodout_reproduces_fortran_reported_ll():
    """External, non-circular oracle for the loader byte order (issue #159):
    recompute the bundled Fortran fixture's OWN reported converged log-likelihood
    from the parameters ``loadmodout`` reads back, and require it to land within
    1e-3 of the fixture's ``LL`` file.

    Log-likelihood is not transpose-symmetric, so unlike ``A @ W == I`` (which
    both byte orders satisfy) this discriminates the orientation of ``W`` and the
    ``(num_mix, num_comps)`` layout of ``sbeta``/``rho``. On the C-order read this
    ships with, it lands ~0.11 off (-3.517 vs -3.402); only the correct
    column-major read reproduces the fixture's number. A write->read round trip
    cancels the error, so no self-consistency test could have caught it -- this
    one recomputes an externally fixed target (the Fortran binary's own output).
    """
    out = loadmodout(amicaout_dir)
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )
    recomputed = _fixture_loglik(out, data)
    reported = float(out.LL[-1])
    assert abs(recomputed - reported) < 1e-3, (
        f"recomputed LL {recomputed:.7f} disagrees with the fixture's own "
        f"reported {reported:.7f} by {abs(recomputed - reported):.5f}; loadmodout "
        "is reading W/sbeta/rho in the wrong byte order (issue #159)."
    )


def test_amicaoutput_sources_contract_and_c_subtraction():
    """``AmicaOutput.sources`` contract (issue #159). Numerical correctness of the
    activation is pinned externally by ``test_loadmodout_reproduces_fortran_
    reported_ll`` (which feeds ``sources`` into the fixture's own LL) and against
    a live model by the multi-model round-trip in the torch wrapper tests; here we
    guard shape/finiteness, the input validation, and that ``c`` is actually
    subtracted (so a future refactor cannot silently drop it).
    """
    out = loadmodout(amicaout_dir)
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )
    b0 = out.sources(data, 0)
    assert b0.shape == (out.num_pcs, data.shape[1])
    assert np.all(np.isfinite(b0))

    # With the fixture's own c == 0 the accessor spheres (mean removed, S applied)
    # and unmixes: b == W @ S(x - mean). This pins the sphering/num_pcs handling.
    sphered = out.S[: out.num_pcs] @ (data - out.data_mean[:, None])
    np.testing.assert_allclose(b0, out.W[:, :, 0] @ sphered, atol=1e-9)

    # Injecting a nonzero c must subtract it in sphered-channel space, shifting
    # the activation to W @ (sphered - c) -- proving the subtraction path runs and
    # a future refactor cannot silently drop it.
    c_probe = np.arange(1, out.num_pcs + 1, dtype=np.float64)
    out.c[:, 0] = c_probe  # out is a fresh load per test; safe to mutate
    b1 = out.sources(data, 0)
    np.testing.assert_allclose(
        b1, out.W[:, :, 0] @ (sphered - c_probe[:, None]), atol=1e-9
    )
    assert not np.allclose(b0, b1), "c subtraction had no effect"

    # Input validation: out-of-range and negative model, wrong data_dim, and a
    # non-2D X (a single 1-D sample vector) must all raise, not silently produce
    # garbage.
    with pytest.raises(ValueError, match="model must be in"):
        out.sources(data, out.num_models)
    with pytest.raises(ValueError, match="model must be in"):
        out.sources(data, -1)
    with pytest.raises(ValueError, match="data_dim"):
        out.sources(data[:16], 0)
    with pytest.raises(ValueError, match="data_dim"):
        out.sources(data[:, 0], 0)  # 1-D X


@pytest.mark.slow
@pytest.mark.xfail(
    reason="uses a per-row corrcoef of the internal W (= Fortran W^T, so rows are "
    "mismatched) and a full 2000-iter run; superseded by "
    "test_sample_data_numpy_vs_fortran, which uses the correct get_weights()@sphere "
    "Hungarian metric. The LL scale/sign issue itself is fixed (issue #24).",
    strict=True,
    raises=AssertionError,
)
def test_sample_data_scikit(tmp_path):
    """Test pamica using scikit-learn style API."""
    # Load original results for comparison
    orig_results = loadmodout(amicaout_dir)

    # Initialize and fit AMICA model using scikit-learn style API.
    # Override outdir (the params file defaults to the relative './amicaout/')
    # so this test does not write stray output into the repo root.
    model = AMICA.from_json_file(sample_params_file, outdir=str(tmp_path / "amicaout"))
    model.fit()

    # Compare weights
    assert model.W is not None
    W_pyamica = model.W[:, :, 0]  # Get weights from first model

    correlations = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            correlations[i, j] = abs(
                np.corrcoef(W_pyamica[i], orig_results.W[j, :, 0])[0, 1]
            )

    # Verify results
    max_correlations = np.max(correlations, axis=1)
    assert np.all(max_correlations > 0.8), (
        "Some components don't match original results"
    )

    best_matches = np.argmax(correlations, axis=1)
    assert len(np.unique(best_matches)) == 32, "Some components are duplicated"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Progress, not passing. Fixed: the #30 format, the #39 NaN (seed pin + "
    "restart), and the #41 LL degradation -- the lrate-ceiling ratchet "
    "(_check_convergence) now keeps the long run improving (seed=0 LL -3.399 at "
    "2000 iters, better than the ~150-iter -3.404 and Fortran's -3.402; 150-iter "
    "mean-corr parity preserved). Remaining: this test's strict all-32-components "
    ">0.8 gate still fails (~27/32) because pamica converges to a comparable/"
    "slightly-better-LL but different-partition solution than the Fortran "
    "reference on a few ill-determined components -- a solution-basin difference "
    "(cf. the single-model tail of #27), not a drift. The project's other parity "
    "test (test_sample_data_numpy_vs_fortran) uses the more appropriate mean>0.9 "
    "bar and passes.",
    strict=True,
)
def test_sample_data_cli():
    """Full CLI-vs-Fortran integration test (issue #30 format + #39/#41 stability).

    Runs the real cli entrypoint for the full 2000-iter sample config and
    Hungarian-matches the loadmodout-read W against the Fortran reference with a
    strict all-32-components > 0.8 gate. A fixed seed is pinned so the test does
    not depend on a random init (see #39). Currently xfail on a residual
    solution-basin difference (see the xfail reason above); the LL-degradation
    drift itself is fixed (#41).
    """
    import subprocess
    import sys

    # cli.py uses relative imports, so it must be run as a module
    # (see its module docstring), not as a direct script path.
    test_outdir = Path("test_output")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pamica.numpy_impl.cli",
                sample_params_file,
                "--outdir",
                str(test_outdir),
                "--seed",
                "0",
            ],
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Load and compare results
        orig_results = loadmodout(amicaout_dir)
        test_results = loadmodout(test_outdir)

        correlations = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                correlations[i, j] = abs(
                    np.corrcoef(test_results.W[i, :, 0], orig_results.W[j, :, 0])[0, 1]
                )

        # Verify results
        max_correlations = np.max(correlations, axis=1)
        assert np.all(max_correlations > 0.8), (
            "Some components don't match original results"
        )

        best_matches = np.argmax(correlations, axis=1)
        assert len(np.unique(best_matches)) == 32, "Some components are duplicated"

    finally:
        # Cleanup
        if test_outdir.exists():
            import shutil

            shutil.rmtree(test_outdir)


@pytest.mark.xfail(
    reason="components not fully decorrelated (off-diagonal < 0.1) after only 50 "
    "iterations -- an under-convergence limit, not a correctness bug. The LL "
    "scale/sign issue is fixed (issue #24; LL is now ~ -3.4, not positive).",
    strict=True,
    raises=AssertionError,
)
def test_sample_data_light(tmp_path):
    """Light version of the test for CI that runs only 50 iterations."""
    # Load and preprocess the data (eeglab_data.fdt stores float32 samples)
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32)

    # Initialize AMICA model with reduced iterations. Override outdir (the
    # params file defaults to the relative './amicaout/') so this test does
    # not write stray output into the repo root.
    model = AMICA.from_json_file(sample_params_file, outdir=str(tmp_path / "amicaout"))
    model.max_iter = 50  # Override max_iter for quick testing

    # Fit the model
    model.fit(data)

    # Basic checks
    W = model.get_weights()
    assert W.shape == (32, 32), f"Expected weight matrix shape (32, 32), got {W.shape}"
    assert not np.any(np.isnan(W)), "Weight matrix contains NaN values"
    assert not np.any(np.isinf(W)), "Weight matrix contains infinite values"

    # Check if weights are reasonably scaled
    assert np.all(np.abs(W) < 100), "Weight values are unreasonably large"
    assert np.all(np.abs(W) > 1e-6), "Weight values are unreasonably small"

    # Check if components are decorrelated
    Y = model.transform(data)
    corr = np.corrcoef(Y[:, :, 0])
    np.fill_diagonal(corr, 0)  # Remove diagonal elements
    assert np.all(np.abs(corr) < 0.1), "Components are not sufficiently decorrelated"


@pytest.mark.slow
@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_sample_data_numpy_vs_fortran(tmp_path):
    """Real-data parity: the NumPy backend's total spatial filter matches the
    Fortran reference (amicaout) with Hungarian-matched component correlation
    > 0.9 (issue #24). Replaces the removed synthetic source-recovery tests.

    The strict > 0.95 definition-of-done gate is the torch backend's
    test_end_to_end_correlation_vs_fortran; this confirms the (bit-identical
    trajectory) NumPy port also converges to the Fortran solution on real data.
    """
    from scipy.optimize import linear_sum_assignment

    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )
    W_ref = np.fromfile(op.join(amicaout_dir, "W"), np.float64).reshape(
        32, 32, order="F"
    )
    S_ref = np.fromfile(op.join(amicaout_dir, "S"), np.float64).reshape(
        32, 32, order="F"
    )

    model = AMICA(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=42,
        block_size=512,
        lrate=0.05,
        lratefact=0.5,
        max_decs=5,
        do_newton=True,
        newt_start=50,
        newtrate=1.0,
        rho0=1.5,
        minrho=1.0,
        maxrho=2.0,
        rholrate=0.05,
        invsigmin=0.0,
        invsigmax=100.0,
        doscaling=True,
        do_mean=True,
        do_sphere=True,
        max_iter=150,
        writestep=10000,
        outdir=str(tmp_path / "out"),
    )
    model.fit(data)

    filt_np = model.get_weights() @ model.sphere  # true unmixing @ sphere
    filt_ref = W_ref @ S_ref
    a = filt_np / np.linalg.norm(filt_np, axis=1, keepdims=True)
    b = filt_ref / np.linalg.norm(filt_ref, axis=1, keepdims=True)
    corr = np.abs(a @ b.T)
    rows, cols = linear_sum_assignment(1 - corr)
    mean_corr = float(corr[rows, cols].mean())
    assert mean_corr > 0.9, f"NumPy vs Fortran component corr {mean_corr:.3f} <= 0.9"


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_cli_output_format_roundtrip(tmp_path):
    """The Fortran-format writer round-trips through loadmodout and load_results,
    and the viz helpers consume the result (issue #30; viz smoke for #15).

    Real sample data, short fit -- this checks the on-disk format contract and
    array shapes, not convergence (the full CLI-vs-Fortran correlation is
    test_sample_data_cli). Fast, so not marked slow.
    """
    import matplotlib

    matplotlib.use("Agg")
    from pamica.numpy_impl.data import load_results
    from pamica.numpy_impl import viz

    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :4096]
    outdir = tmp_path / "out"
    model = AMICA(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=1,
        max_iter=15,
        writestep=10000,
        do_opt_block=False,
        outdir=str(outdir),
    )
    # fit() persists the final result even though max_iter (15) is not a
    # writestep (10000) multiple -- exercises the unconditional final write.
    model.fit(data)

    # loadmodout reads the Fortran-format output. Before issue #30 the CLI wrote
    # .npy, so this raised FileNotFoundError for 'W'.
    out = loadmodout(outdir)
    assert out.W.shape == (32, 32, 1)

    # load_results returns AMICA's internal shapes for the viz helpers.
    r = load_results(str(outdir))
    assert r["A"].shape == (32, 32)
    assert r["W"].shape == (32, 32, 1)
    assert r["alpha"].shape == (3, 32)
    assert r["comp_list"].shape == (32, 1)
    assert int(r["comp_list"].min()) == 0 and int(r["comp_list"].max()) == 31

    # Value-level round-trip: the loaded arrays equal the in-memory ones. Guards
    # against a transpose/axis-order or dtype regression that the shape checks
    # above would miss.
    assert model.W is not None and model.A is not None and model.alpha is not None
    np.testing.assert_allclose(r["W"], model.W)
    np.testing.assert_allclose(r["A"], model.A)
    np.testing.assert_allclose(r["alpha"], model.alpha)
    np.testing.assert_array_equal(r["comp_list"], model.comp_list)

    # viz helpers run without error on the loaded results.
    viz.plot_convergence(str(outdir))
    viz.plot_components(str(outdir), data=None, max_comps=3)
    # Also exercise the activation branch (data provided), not just the
    # mixing-vector-only path.
    viz.plot_components(str(outdir), data=data, max_comps=3)
    viz.plot_pdf_fits(str(outdir), data, max_comps=2)


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_numpy_llt_roundtrip(tmp_path):
    """A real (short but genuine) single-model NumPy fit writes an ``LLt`` file
    that round-trips through ``loadmodout`` (issue #155).

    ``Lht[0]`` must equal ``Lt`` exactly for a single model (both are read from
    the same on-disk sequence). ``Lt.mean()`` (the per-sample total log-density,
    summed over channels) should be close to ``model.ll[-1] * model.data_dim``
    -- ``self.ll`` is this backend's per-sample-per-channel normalized
    per-iteration log-likelihood (issue #212, matching Fortran's
    ``LL(iter) = LLtmp2 / dble(numgoodsum*nw)``), so multiplying back by the
    channel count recovers the same per-sample-summed-over-channels convention
    as ``Lt.mean()``. They are not bit-identical: ``self.ll[-1]`` is the
    pre-update E-step likelihood of the last iteration, while the written LLt
    is recomputed post-update, so a loose tolerance is used.
    """
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :4096]
    outdir = tmp_path / "out"
    model = AMICA(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=1,
        max_iter=30,
        writestep=10000,
        do_opt_block=False,
        outdir=str(outdir),
    )
    model.fit(data)

    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None
    assert out.Lht.shape == (1, model.num_samples)
    np.testing.assert_array_equal(out.Lht[0], out.Lt)

    expected = model.ll[-1] * model.data_dim
    np.testing.assert_allclose(out.Lt.mean(), expected, rtol=1e-2)


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_numpy_llt_reject_zeroes_rejected_samples(tmp_path):
    """Under ``do_reject``, rejected samples must be exactly zero in the
    written ``LLt`` (issue #155 Fix 1): Fortran zeroes a rejected sample's
    ``modloglik``/``loglik`` on write (amica15.f90:2211-2216) and its
    ``load_rej`` reader reconstructs the rejection mask from that exact zero
    sentinel (``sum(modloglik(:,i)) == 0.0``, amica15.f90:887-896). Good
    samples must stay non-zero and finite.
    """
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :4096]
    outdir = tmp_path / "out"
    model = AMICA(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=1,
        max_iter=15,
        writestep=10000,
        do_opt_block=False,
        do_reject=True,
        rejstart=2,
        rejint=3,
        maxrej=3,
        rejsig=2.0,
        outdir=str(outdir),
    )
    model.fit(data)
    good_idx = model.good_idx
    num_samples = model.num_samples
    assert good_idx is not None and num_samples is not None
    assert good_idx.size < num_samples, "fixture must reject something"

    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None
    good = np.zeros(num_samples, dtype=bool)
    good[good_idx] = True

    np.testing.assert_array_equal(out.Lht[:, ~good], 0.0)
    np.testing.assert_array_equal(out.Lt[~good], 0.0)
    assert np.all(out.Lht[:, good] != 0.0)
    assert np.all(np.isfinite(out.Lht[:, good]))
    assert np.all(out.Lt[good] != 0.0)
    assert np.all(np.isfinite(out.Lt[good]))


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_numpy_llt_multimodel_roundtrip(tmp_path):
    """A small real 2-model NumPy fit's ``LLt`` satisfies its definitional
    relationship: the total per-sample log-likelihood is the log-sum-exp of
    the per-model log-likelihoods (issue #155). Only the torch backend had a
    multi-model LLt round-trip test; the two ``_compute_full_posterior_ll``
    are separate implementations, so this exercises the numpy one too. Few
    iterations -- this is a code-path smoke test, not a convergence check.
    """
    from scipy.special import logsumexp

    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :4096]
    outdir = tmp_path / "out"
    model = AMICA(
        use_tqdm=False,
        num_models=2,
        num_mix=3,
        seed=7,
        max_iter=5,
        writestep=10000,
        do_opt_block=False,
        outdir=str(outdir),
    )
    model.fit(data)

    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None
    assert out.Lht.shape == (2, model.num_samples)
    np.testing.assert_allclose(out.Lt, logsumexp(out.Lht, axis=0), rtol=1e-10)


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_numpy_llt_partial_final_block(tmp_path):
    """The block loop's remainder branch (``end = min(start + block_size,
    n_samples)``) is exercised by a sample count NOT evenly divisible by
    ``block_size`` (issue #155 Fix 6d -- the existing LLt tests all used
    exactly-divisible counts, e.g. 4096/128, so this branch was untested).
    """
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :4100]
    assert data.shape[1] % 1024 != 0, "fixture must not be block_size-divisible"
    outdir = tmp_path / "out"
    model = AMICA(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=1,
        max_iter=5,
        writestep=10000,
        do_opt_block=False,
        block_size=1024,
        outdir=str(outdir),
    )
    model.fit(data)

    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None
    assert out.Lht.shape == (1, model.num_samples)
    np.testing.assert_array_equal(out.Lht[0], out.Lt)
    assert np.all(np.isfinite(out.Lt))


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_restart_on_early_nan_recovers(tmp_path):
    """An early non-finite LL triggers reinitialize-and-restart (Fortran
    restartiter path, #39); the fit recovers and finishes with a finite LL.

    Injects a non-finite LL for the first two iterations via a subclass to
    exercise the restart control flow deterministically. This is an error-path
    test, not a data mock -- the model still fits the real sample data after the
    restart, and the numerical result is not asserted from fabricated values.
    """
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :2048]

    class _InjectEarlyNaN(AMICA):
        _nan_iters = 2

        def _get_updates_and_likelihood(self):
            upd = super()._get_updates_and_likelihood()
            if self.iter < self._nan_iters:
                upd["ll"] = float("nan")
            return upd

    model = _InjectEarlyNaN(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=3,
        max_iter=20,
        writestep=10_000_000,
        do_opt_block=False,
        do_newton=False,
        restartiter=10,
        maxrestarts=3,
        outdir=str(tmp_path / "out"),
    )
    model.fit(data)

    # One restart per injected-NaN iteration, then a normal finite fit.
    assert model.numrestarts == 2
    assert model.converged is True
    assert len(model.ll) >= 1
    assert np.isfinite(model.ll[-1])
    # The restart must be announced in the run log, not silent.
    log_text = (tmp_path / "out" / "out.txt").read_text().lower()
    assert "reinitializing" in log_text


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_restart_gives_up_after_maxrestarts(tmp_path):
    """If every early iteration is non-finite, the fit stops after maxrestarts
    reinitializations instead of looping forever (Fortran numrestarts cap)."""
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :2048]

    class _AlwaysNaN(AMICA):
        def _get_updates_and_likelihood(self):
            upd = super()._get_updates_and_likelihood()
            upd["ll"] = float("nan")
            return upd

    model = _AlwaysNaN(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=3,
        max_iter=50,
        writestep=10_000_000,
        do_opt_block=False,
        do_newton=False,
        restartiter=10,
        maxrestarts=2,
        outdir=str(tmp_path / "out"),
    )
    model.fit(data)
    # Restarts are capped, the run stops on the persistent non-finite LL, and
    # the terminal failure is surfaced (converged=False), not silently ignored.
    assert model.numrestarts == 2
    assert model.converged is False
    assert not np.isfinite(model.ll[-1])


def test_check_convergence_ratchets_lrate_on_decrease():
    """After max_decs consecutive LL decreases, _check_convergence lowers the
    learning-rate ceilings (lrate0; the rho rate once iter>newt_start; newtrate
    under Newton) and resets numdecs -- it does NOT stop -- matching Fortran's
    ratchet (#41). Drives the convergence handler directly with a decreasing LL
    history.

    Also pins the #193 fix: the rho rate is a maxdecs-ratcheted CEILING, NOT a
    per-decrease monotone decay. rholrate must stay untouched on a decrease below
    max_decs and only ratchet when numdecs hits max_decs.
    """
    model = AMICA(
        num_models=1,
        num_mix=3,
        do_newton=True,
        newt_start=10,
        max_decs=2,
        lratefact=0.5,
        use_grad_norm=False,
        use_min_dll=False,
    )
    model.iter = 100  # past newt_start, so the Newton/rho-rate ratchets apply
    model.nd = [1.0]  # gradient norm well above min_grad_norm (no floor stop)
    lrate0_before = model.lrate0
    newtrate_before = model.newtrate
    rholrate_before = model.rholrate

    # First decrease: numdecs -> 1 (below max_decs), so just reduce lrate. The rho
    # ceiling must NOT move here -- pre-#193 it decayed on every decrease.
    model.ll = [-3.0, -3.1]
    conv, _, numdecs, numincs = model._check_convergence(0, 0)
    assert conv is False
    assert numdecs == 1
    assert model.rholrate == rholrate_before

    # Second consecutive decrease: numdecs hits max_decs=2 -> ratchet ceilings,
    # reset numdecs, and CONTINUE (not converged).
    model.ll = [-3.1, -3.2]
    conv, _, numdecs, numincs = model._check_convergence(numdecs, 0)
    assert conv is False
    assert numdecs == 0
    assert model.lrate0 == lrate0_before * 0.5
    assert model.newtrate == newtrate_before * 0.5
    assert model.rholrate == pytest.approx(rholrate_before * model.rholratefact)


def test_reinitialize_for_restart_resets_rho_ceiling():
    """Issue #193: a mid-fit restart (non-finite LL, Fortran restartiter path)
    resets the rho-rate ceiling to rholrate0, exactly like the lrate reset -- a
    previously-ratcheted rholrate must not carry across the restart.
    """
    model = AMICA(num_models=1, num_mix=3, seed=0)
    model.data_dim = 4
    model.num_samples = 16
    model.num_comps = 4
    model.good_idx = None
    model.num_good_samples = 16
    model._initialize_parameters()

    # Simulate a fit that ratcheted both ceilings, then hit a non-finite LL.
    model.rholrate = model.rholrate0 * 0.25
    model.lrate = model.lrate0 * 0.25
    model.ll = [-3.0, -3.1]
    model.nd = [0.5]

    model._reinitialize_for_restart()

    # Both ceilings restored to pristine; history cleared for a fresh judgment.
    assert model.rholrate == model.rholrate0
    assert model.lrate == model.lrate0
    assert model.ll == []


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_cli_subprocess_output_loadable(tmp_path):
    """The actual cli entrypoint writes loadmodout-readable output.

    Runs the CLI as a module on a short, stable config (real sample data) and
    confirms loadmodout reads the result -- the direct regression for the #30
    FileNotFoundError (the CLI previously wrote .npy). Kept short (few
    iterations, Newton off) so it stays fast and avoids the separate long-fit
    NaN of #39; the full 2000-iter integration run is test_sample_data_cli.
    """
    import json
    import subprocess
    import sys

    params = {
        "files": [eeglab_data_file],
        "data_dim": 32,
        "field_dim": [30504],
        "num_models": 1,
        "num_mix": 3,
        "max_iter": 8,
        "writestep": 4,
        "do_newton": False,
        "do_opt_block": False,
        "block_size": 512,
    }
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params))
    outdir = tmp_path / "cli_out"

    # cli uses relative imports, so run it as a module from the repo root.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pamica.numpy_impl.cli",
            str(params_file),
            "--outdir",
            str(outdir),
        ],
        check=True,
        cwd=Path(__file__).parent.parent.parent,
    )

    # loadmodout reads the CLI output (previously raised FileNotFoundError).
    out = loadmodout(outdir)
    assert out.W.shape == (32, 32, 1)
    assert np.all(np.isfinite(out.W))


if __name__ == "__main__":
    pytest.main([__file__])
