"""Component sharing (issue #60) for AMICATorchNG.

Covers the ``share_comps`` reassignment ported from the Fortran
``identify_shared_comps`` (amica15.f90:1916). There is no bit-exact oracle (the
reference's ``Spinv2`` metric is declared but never allocated, like the dead
``do_choose_pdfs`` switch, #26), so the merge *mechanism* is tested with
controlled mixing matrices and the end-to-end behavior on real sample EEG.
Sharing is multi-model only and OFF by default, so single-model parity is
unchanged.

Issue #253 replaced the merge metric's ``inv(sphere)`` with ``pinv(sphere)``, so
sharing also runs on rank-reduced and rank-deficient fits (the Maxwell-filtered
MEG case reported in #221). The rank-reduced section below covers both routes
(explicit ``pcakeep`` and automatic rank detection on subspace-projected real
EEG) and pins that full-rank merge decisions are unchanged by the swap.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pamica.amica import AMICA
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _controlled_ng(A: torch.Tensor, n_channels: int, n_models: int) -> AMICATorchNG:
    """An NG with a fixed identity sphere and default comp_list, whose A is the
    given (n_channels, n_channels*n_models) matrix -- so the sharing metric is a
    plain cosine similarity on A's columns. No fit needed."""
    ng = AMICATorchNG(
        n_channels=n_channels,
        n_models=n_models,
        n_mix=3,
        device="cpu",
        share_comps=True,
        comp_thresh=0.99,
    )
    ng.sphere = torch.eye(n_channels, dtype=torch.float64)
    ng._sphere_pinv = None
    ng.iteration = 0
    cl = np.zeros((n_channels, n_models), dtype=np.int64)
    for h in range(n_models):
        cl[:, h] = np.arange(h * n_channels, (h + 1) * n_channels)
    ng.comp_list = torch.from_numpy(cl)
    ng.A = A.to(torch.float64)
    return ng


# --- default-path parity ----------------------------------------------------


def test_single_model_byte_identical_with_share_toggled(real_data):
    """Sharing is a no-op for n_models=1, so a single-model fit must be
    byte-for-byte identical with share_comps on vs off (the gm-weighted A-update
    reduces to the plain update since gm=[1] cancels exactly)."""
    x = real_data[:, :4096]
    off = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    off.fit(x, max_iter=8, seed=42, block_size=1024, do_newton=True)
    on = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    on.fit(
        x,
        max_iter=8,
        seed=42,
        block_size=1024,
        do_newton=True,
        share_comps=True,
        share_start=7,
        share_iter=8,
    )
    np.testing.assert_array_equal(off.get_mixing_matrix(0), on.get_mixing_matrix(0))
    np.testing.assert_array_equal(off.get_unmixing_matrix(0), on.get_unmixing_matrix(0))
    assert off.final_ll_ == on.final_ll_


def test_default_multimodel_leaves_comps_unshared(real_data):
    """With share_comps off (default), a 2-model fit keeps every component
    distinct (comp_list stays the full block layout)."""
    ng = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=3, device="cpu", block_size=1024, seed=1
    )
    ng.fit(real_data[:, :4096], max_iter=10)
    assert ng.A is not None
    assert int(ng.comp_used.sum()) == ng.n_comps
    assert torch.isfinite(ng.A).all()


# --- merge mechanism (controlled) ------------------------------------------


def test_identify_merges_one_collinear_pair():
    """A single collinear cross-model column pair is merged; the others (random,
    near-orthogonal) are left alone."""
    torch.manual_seed(0)
    n = 4
    A = torch.randn(n, 2 * n, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    A[:, 5] = A[
        :, 2
    ]  # model-1 source 1 (col 5) collinear with model-0 source 2 (col 2)
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    assert ng.comp_list is not None
    assert int(ng.comp_used.sum()) == 2 * n
    ng._identify_shared_comps()
    assert int(ng.comp_used.sum()) == 2 * n - 1
    assert int(ng.comp_list[1, 1]) == int(ng.comp_list[2, 0]) == 2
    assert not bool(ng.comp_used[5])  # col 5 is now unused


def test_guard_prevents_within_model_collapse():
    """When every column is collinear, sharing must NOT collapse a model's own
    sources together: with 2 sources x 2 models it settles to 2 shared comps,
    not 1 (Fortran's 'common presence in a model' guard, amica15.f90:1936)."""
    n = 2
    v = torch.tensor([1.0, 2.0], dtype=torch.float64)
    A = torch.stack([v] * (2 * n), dim=1)
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    assert ng.comp_list is not None
    ng._identify_shared_comps()
    assert int(ng.comp_used.sum()) == 2
    assert ng.comp_list[0, 0] != ng.comp_list[1, 0]


def test_three_model_guard_and_merge():
    """3-model scan (h<hh runs 3 pairs, with sequential mutation): a source
    collinear across all three models collapses to ONE shared comp, while each
    model's other source stays distinct."""
    n = 2
    torch.manual_seed(1)
    A = torch.randn(n, 3 * n, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    # source 0 of every model shares one direction; source 1 stays random.
    shared = A[:, 0].clone()
    A[:, 2] = shared  # model1 source0 (col 2)
    A[:, 4] = shared  # model2 source0 (col 4)
    ng = _controlled_ng(A, n_channels=n, n_models=3)
    assert ng.comp_list is not None
    ng._identify_shared_comps()
    # source-0 column shared by all 3 models -> one comp; three distinct source-1s.
    assert int(ng.comp_list[0, 0]) == int(ng.comp_list[0, 1]) == int(ng.comp_list[0, 2])
    assert int(ng.comp_used.sum()) == 4  # 1 shared + 3 distinct source-1 comps
    # no model shares a comp between its own two sources
    for h in range(3):
        assert ng.comp_list[0, h] != ng.comp_list[1, h]


def test_comp_thresh_one_merges_only_exact_duplicates():
    """comp_thresh=1.0: an exact-duplicate column merges (cos==1), a merely
    similar one does not."""
    n = 3
    torch.manual_seed(2)
    A = torch.randn(n, 2 * n, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    A[:, 3] = A[:, 0]  # exact duplicate across models
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    assert ng.comp_list is not None
    ng.comp_thresh = 1.0
    ng._identify_shared_comps()
    assert int(ng.comp_list[0, 1]) == int(ng.comp_list[0, 0])  # duplicate merged
    assert int(ng.comp_used.sum()) == 2 * n - 1


# --- freeze schedule --------------------------------------------------------


def test_a_frozen_window():
    """A is frozen for the merge iteration + 5 after it, thawed for the rest of
    each cycle, and frozen again at the next cycle boundary."""
    ng = AMICATorchNG(
        n_channels=4,
        n_models=2,
        device="cpu",
        share_comps=True,
        share_start=10,
        share_iter=20,
    )

    def frozen(itf):
        ng.iteration = itf - 1  # itf is the Fortran-style 1-indexed iteration
        return ng._a_frozen()

    assert not any(frozen(i) for i in range(1, 10))  # before share_start
    assert all(frozen(i) for i in range(10, 16))  # merge + 5 (residue 0..5)
    assert not any(frozen(i) for i in range(16, 30))  # thawed rest of cycle
    assert all(frozen(i) for i in range(30, 36))  # next cycle boundary


def test_a_frozen_off_for_single_model():
    ng = AMICATorchNG(
        n_channels=4,
        n_models=1,
        device="cpu",
        share_comps=True,
        share_start=2,
        share_iter=8,
    )
    ng.iteration = 3
    assert ng._a_frozen() is False


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(share_start=0), "share_start"),
        (dict(share_iter=6), "share_iter"),
        (dict(share_iter=1), "share_iter"),
        (dict(comp_thresh=0.0), "comp_thresh"),
        (dict(comp_thresh=1.5), "comp_thresh"),
    ],
)
def test_share_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AMICATorchNG(n_channels=8, n_models=2, share_comps=True, **kwargs)


def test_pca_reduction_no_longer_rejected_at_construction():
    """``pcakeep``/``pcadb`` used to be refused up front because the merge metric
    inverted the sphere (issue #253). The pseudo-inverse back-map works at any
    rank, so the combination is now legal."""
    by_count = AMICATorchNG(
        n_channels=8, n_models=2, device="cpu", share_comps=True, pcakeep=4
    )
    by_db = AMICATorchNG(
        n_channels=8, n_models=2, device="cpu", share_comps=True, pcadb=30.0
    )
    assert by_count.share_comps is True and by_count.pcakeep == 4
    assert by_db.share_comps is True and by_db.pcadb == 30.0


def test_non_finite_sphere_still_fails_loudly(real_data):
    """A degenerate fit's sphere has no pseudo-inverse either, and that must
    raise rather than silently decline every merge (the per-pair NaN guard in
    the scan would otherwise swallow it)."""
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        bad,
        max_iter=2,
        block_size=1024,
        seed=0,
        share_comps=True,
        share_start=1,
        share_iter=8,
        comp_thresh=0.9,
    )
    ng = model.model_
    assert ng is not None and ng.sphere is not None
    assert not bool(torch.isfinite(ng.sphere).all())
    with pytest.raises(RuntimeError, match="non-finite"):
        ng._identify_shared_comps()


def test_get_sensor_mixing_matrix_non_finite_sphere_fails_loudly(real_data):
    """``get_sensor_mixing_matrix()`` is a second, independent public-API caller
    of ``_pinv_sphere()`` -- it must fail the same way as the internal
    ``_identify_shared_comps`` route above (issue #261; twin of
    ``test_numpy_share_comps.py::test_non_finite_sphere_fails_loudly``, which
    already pins this for the NumPy backend)."""
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        bad,
        max_iter=2,
        block_size=1024,
        seed=0,
        share_comps=True,
        share_start=1,
        share_iter=8,
        comp_thresh=0.9,
    )
    ng = model.model_
    assert ng is not None and ng.sphere is not None
    assert not bool(torch.isfinite(ng.sphere).all())
    with pytest.raises(RuntimeError, match="non-finite"):
        ng.get_sensor_mixing_matrix()


# --- end-to-end on real data ------------------------------------------------


def test_two_model_share_fit_survives_merge(real_data):
    """A full 2-model fit with sharing runs to completion with finite parameters
    and the merge SURVIVES to the returned model (keep_best is disabled under
    share_comps, so fit returns the merged last iterate, not a pre-merge peak)."""
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=3,
        do_newton=True,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    ng.fit(real_data[:, :4096], max_iter=40)
    assert ng.A is not None and ng.final_ll_ is not None
    assert torch.isfinite(ng.A).all()
    assert np.isfinite(ng.final_ll_)
    assert int(ng.comp_used.sum()) < ng.n_comps  # at least one merge survived


def test_sharing_reduces_unique_count_without_degrading_ll(real_data):
    """Enabling sharing on matched config strictly reduces the unique-component
    count and does not materially degrade the log-likelihood."""
    x = real_data[:, :4096]
    common: dict[str, Any] = dict(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=7,
        do_newton=True,
    )
    base = AMICATorchNG(**common)
    base.fit(x, max_iter=40)
    shared = AMICATorchNG(
        **common, share_comps=True, share_start=8, share_iter=10, comp_thresh=0.9
    )
    shared.fit(x, max_iter=40)
    assert int(base.comp_used.sum()) == base.n_comps
    assert int(shared.comp_used.sum()) < base.n_comps
    assert shared.final_ll_ is not None and base.final_ll_ is not None
    assert np.isfinite(shared.final_ll_)
    assert shared.final_ll_ > base.final_ll_ - 0.3  # no material degradation


def _assert_share_result_consistent(ng: AMICATorchNG) -> None:
    """Every merged fit must hold finite parameters and a comp_list that agrees
    with the derived comp_used mask and the shared_components() grouping."""
    assert ng.A is not None and ng.comp_list is not None
    for name in ("A", "W", "mu", "alpha", "beta", "rho", "gm", "c"):
        tensor = getattr(ng, name)
        assert tensor is not None and torch.isfinite(tensor).all(), name
    assert ng.final_ll_ is not None and np.isfinite(ng.final_ll_)

    cl = ng.comp_list.cpu().numpy()
    assert cl.shape == (ng.n_channels, ng.n_models)
    assert cl.min() >= 0 and cl.max() < ng.n_comps
    assert tuple(ng.A.shape) == (ng.n_channels, ng.n_comps)
    used = int(ng.comp_used.sum())
    assert used == len(np.unique(cl))

    groups = ng.shared_components()
    for group in groups:
        cols = {int(cl[i, h]) for h, i in group}
        assert len(cols) == 1, "a shared group must reference exactly one column"
        assert len({h for h, _ in group}) >= 2, "sharing is across models"
    if ng.n_models == 2:
        # With two models the within-model guard caps a group at one source per
        # model, so every merge folds exactly one column away into one new pair.
        assert len(groups) == ng.n_comps - used


def test_merge_on_the_final_iteration_completes(real_data):
    """A merge scheduled on the LAST iteration must still leave a usable model.

    The schedule hook runs after the iteration's likelihood has been recorded
    (Fortran runs identify_shared_comps after accum_updates_and_likelihood,
    amica15.f90:1856), so ``final_ll_`` deliberately describes the state BEFORE
    this merge -- the merged model is never scored. That is faithful to the
    reference and matches AMICAMLXNG (mirrors
    tests/mlx_tests/test_mlx_sharing.py::test_merge_on_the_final_iteration_completes),
    but it is a real trap for a caller comparing ``final_ll_`` against
    ``comp_used``; issue #269 tracks documenting it across backends. Here it is
    pinned as behavior: the fit completes, the merge survives on the returned
    model, and the reported LL is finite. keep_best is disabled under
    share_comps (see the final_ll_ comment), so this is not a keep_best
    artifact.
    """
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=3,
        share_comps=True,
        share_start=10,
        share_iter=100,  # only one merge attempt, on the final iteration
        comp_thresh=0.9,
    )
    ng.fit(real_data[:, :4096], max_iter=10, verbose=False)

    assert len(ng.ll_history) == 10  # the merge did not truncate the run
    assert ng.comp_used is not None and int(ng.comp_used.sum()) < ng.n_comps
    _assert_share_result_consistent(ng)


# --- rank-reduced sharing (issue #253, MEG report in #221) -------------------


def test_rank_reduced_share_fit_completes(real_data):
    """``share_comps`` + PCA reduction used to be refused at construction, and
    the merge metric would have raised on the non-square sphere. The
    pseudo-inverse back-map handles it: the fit runs and merges."""
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        real_data[:, :4096],
        max_iter=25,
        block_size=1024,
        seed=3,
        do_newton=True,
        pcakeep=16,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    ng = model.model_
    assert ng is not None and ng.sphere is not None
    assert ng.n_channels == 16 and ng.n_channels_in == NW
    assert tuple(ng.sphere.shape) == (16, NW)  # non-square: no inverse exists
    assert int(ng.comp_used.sum()) < ng.n_comps  # the sharing path really ran
    _assert_share_result_consistent(ng)
    assert model.shared_components() == ng.shared_components()
    assert ng.get_sensor_mixing_matrix().shape == (NW, 16)


def test_low_rank_projected_data_share_fit_completes(real_data):
    """The #221 MEG route: real EEG projected onto a rank-20 subspace (what
    Maxwell filtering does to MEG), fitted with automatic rank detection and
    sharing on. Before #253 this raised "Component sharing needs an invertible
    sphere"."""
    x = real_data[:, :4096]
    x = x - x.mean(axis=1, keepdims=True)
    rank = 20
    U_r = np.linalg.svd(x, full_matrices=False)[0][:, :rank]
    x_low = U_r @ (U_r.T @ x)

    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        x_low,
        max_iter=25,
        block_size=1024,
        seed=3,
        do_newton=True,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    ng = model.model_
    assert ng is not None and ng.sphere is not None
    assert ng.n_channels == rank and ng.n_channels_in == NW
    assert tuple(ng.sphere.shape) == (rank, NW)
    assert int(ng.comp_used.sum()) < ng.n_comps
    _assert_share_result_consistent(ng)
    assert ng.get_sensor_mixing_matrix().shape == (NW, rank)


def test_full_rank_merge_decisions_unchanged_by_pinv(real_data):
    """Regression for the inv -> pinv swap: on a full-rank sphere the two
    de-sphering metrics give the *same* merge decisions.

    Run twice from one real fitted state -- once with ``pinv(sphere)`` (current)
    and once with the exact ``inv(sphere)`` (pre-#253) injected into the same
    cache -- and require identical comp_lists, with merges actually firing so the
    comparison is not vacuous."""
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=3,
        do_newton=True,
    )
    ng.fit(real_data[:, :4096], max_iter=40)
    assert ng.sphere is not None and ng.comp_list is not None
    assert tuple(ng.sphere.shape) == (NW, NW)
    ng.comp_thresh = 0.9
    initial = ng.comp_list.clone()

    ng._sphere_pinv = None  # rebuilt as pinv(sphere)
    ng._identify_shared_comps()
    with_pinv = ng.comp_list.clone()

    ng.comp_list = initial.clone()
    ng._sphere_pinv = torch.linalg.inv(ng.sphere)  # the pre-#253 metric
    ng._identify_shared_comps()
    with_inv = ng.comp_list.clone()

    assert not torch.equal(with_pinv, initial), "no merge fired; test is vacuous"
    assert torch.equal(with_pinv, with_inv)


def test_pinv_matches_inv_on_a_fitted_full_rank_sphere(real_data):
    """Numerical pin behind the swap: on the fitted full-rank sphere the two
    de-sphered mixing matrices agree far below any comp_thresh boundary."""
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=7,
        do_newton=True,
    )
    ng.fit(real_data[:, :4096], max_iter=10)
    assert ng.sphere is not None and ng.A is not None
    delta = (
        (torch.linalg.pinv(ng.sphere) @ ng.A - torch.linalg.inv(ng.sphere) @ ng.A)
        .abs()
        .max()
        .item()
    )
    assert delta < 1e-10, f"pinv/inv de-sphering differ by {delta:.3e}"


def test_share_config_and_comp_list_roundtrip(real_data, tmp_path):
    """save/load preserves the share configuration and the merged comp_list."""
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        real_data[:, :4096],
        max_iter=25,
        block_size=1024,
        seed=3,
        do_newton=True,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    assert model.model_ is not None
    assert int(model.model_.comp_used.sum()) < model.model_.n_comps  # a merge happened
    path = str(tmp_path / "shared.pt")
    model.save(path)
    loaded = AMICA.load(path, device="cpu")
    assert loaded.model_ is not None
    assert loaded.model_.share_comps is True
    assert loaded.model_.share_start == 8 and loaded.model_.share_iter == 10
    assert loaded.model_.comp_list is not None and model.model_.comp_list is not None
    assert torch.equal(loaded.model_.comp_list, model.model_.comp_list)
