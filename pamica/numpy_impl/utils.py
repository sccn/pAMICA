"""
Utility functions for AMICA implementation.

This module provides core utility functions used throughout the AMICA package,
including mathematical operations, component analysis tools, and data processing
utilities. These functions support the main AMICA algorithm implementation.
"""

import numpy as np
from scipy import special


def gammaln(x):
    """
    Compute the natural logarithm of the gamma function.

    The gamma function is an extension of the factorial function to real and
    complex numbers. This function computes ln(Γ(x)) using SciPy's implementation.

    Parameters
    ----------
    x : float or array_like
        Input value(s)

    Returns
    -------
    float or ndarray
        Natural logarithm of gamma function evaluated at x
    """
    return special.gammaln(x)


def psifun(x):
    """
    Compute the digamma function (derivative of log gamma).

    The digamma function is the logarithmic derivative of the gamma function,
    defined as ψ(x) = d/dx ln(Γ(x)). This implementation uses SciPy's digamma
    function and matches the behavior of the original Fortran implementation.

    Parameters
    ----------
    x : float or array_like
        Input value(s)

    Returns
    -------
    float or ndarray
        Digamma function evaluated at x
    """
    return special.digamma(x)


def determine_block_size(data, min_size, max_size, step_size, num_threads=1):
    """
    Determine optimal block size for data processing through empirical testing.

    This function tests different block sizes by performing representative matrix
    operations and measuring execution time. The block size that results in the
    fastest processing time is selected as optimal.

    Parameters
    ----------
    data : ndarray
        Input data array
    min_size : int
        Minimum block size to try
    max_size : int
        Maximum block size to try
    step_size : int
        Step size between block sizes to try
    num_threads : int
        Number of threads to use

    Returns
    -------
    optimal_size : int
        Optimal block size
    """
    import time

    block_times = []
    block_sizes = range(min_size, max_size + 1, step_size)

    # Test each block size
    for block_size in block_sizes:
        start_time = time.time()

        # Process one block
        for start in range(0, data.shape[1], block_size):
            end = min(start + block_size, data.shape[1])
            X = data[:, start:end]

            # Do some representative computation
            _ = np.dot(X.T, X)

        block_times.append(time.time() - start_time)

    # Return block size with minimum processing time
    return block_sizes[np.argmin(block_times)]


def identify_shared_components(atil, comp_list, comp_thresh=0.99):
    """
    Identify components that are shared between different models, in sensor space.

    Two components (model ``h`` source ``i`` and model ``hh`` source ``ii``,
    ``h < hh``) are identified when the angle between their mixing columns,
    measured in the original (de-sphered) data space, is below the
    ``comp_thresh`` cutoff::

        t0 = |a . b| / (||a|| ||b||),   a = atil[:, ci], b = atil[:, cj]

    ``atil`` must already be the de-sphered (sensor-space) mixing columns --
    callers pass ``pinv(sphere) @ A``, the Fortran ``Spinv`` back-map
    (amica15.f90:568-578) applied to the mixing matrix, mirroring
    ``identify_shared_comps`` (amica15.f90:1916). This is a cross-backend
    agreement contract (.rules/backend_parity.md): ``AMICATorchNG._identify_shared_comps``
    (torch_impl/core.py) computes the identical ``pinv(sphere) @ A`` metric,
    so both backends make the same merge decision from the same fitted state
    (see ``tests/test_numpy_share_comps.py::test_numpy_merge_decision_matches_torch_backend``).
    Before issue #258 this function compared raw columns of the *sphered* ``A``
    directly, which could disagree with the PyTorch backend under rank
    reduction or PCA whitening, where the sphere is not orthonormal.

    On a match, ``cj`` is folded into ``ci``: every ``comp_list`` entry equal
    to ``cj`` is reassigned to ``ci``, so the two now share one mixing column
    and one density.

    Greedy and order-dependent, matching the reference's quadruple loop.
    Skips a pair already merged, or one whose two columns coexist in some
    single model (a model cannot share a component with itself).

    Parameters
    ----------
    atil : ndarray of shape (data_dim_in, num_comps)
        De-sphered (sensor-space) mixing columns, i.e. ``pinv(sphere) @ A``.
    comp_list : ndarray of shape (data_dim, num_models)
        Component assignments. Not mutated -- a copy is merged and returned.
    comp_thresh : float
        Cosine-similarity threshold for identifying shared components.

    Returns
    -------
    comp_list : ndarray
        Updated component assignments (a new array; the input is untouched).
    comp_used : ndarray
        Boolean mask of used components.
    """
    cl = comp_list.copy()
    nw, num_models = cl.shape
    norms = np.linalg.norm(atil, axis=0)
    tiny = np.finfo(atil.dtype).tiny

    for h in range(num_models):
        for hh in range(h + 1, num_models):
            for i in range(nw):
                for ii in range(nw):
                    ci, cj = int(cl[i, h]), int(cl[ii, hh])
                    if ci == cj:
                        continue

                    t0 = np.abs(atil[:, ci] @ atil[:, cj]) / (
                        norms[ci] * norms[cj] + tiny
                    )
                    # NaN t0 (e.g. a zero-norm column) must NOT merge:
                    # `NaN >= thresh` is False either way, but guard finiteness
                    # explicitly so a future rewrite of the comparison direction
                    # cannot silently start merging on NaN.
                    if not np.isfinite(t0) or t0 < comp_thresh:
                        continue

                    # A model cannot share a component with itself: skip if any
                    # single model already uses both columns.
                    if any(
                        (cl[:, k] == ci).any() and (cl[:, k] == cj).any()
                        for k in range(num_models)
                    ):
                        continue

                    cl[cl == cj] = ci  # fold cj into ci everywhere

    # Derive comp_used from the final comp_list rather than tracking it during
    # the merge loop. A fresh np.ones() per call forgot every column merged away
    # in an earlier call: once comp_list is fully merged the ci == cj guard skips
    # every pair, and the mask came back all-True while half the columns were
    # dead (issue #240). Matches AMICATorchNG.comp_used, which is a property
    # derived the same way.
    num_comps = atil.shape[1]
    comp_used = np.zeros(num_comps, dtype=bool)
    comp_used[np.unique(cl)] = True

    return cl, comp_used


def get_unmixing_matrices(A, comp_list):
    """
    Compute unmixing matrices from mixing matrix and component assignments.

    For each model, constructs the unmixing matrix by inverting the appropriate
    subset of the mixing matrix columns as specified by the component assignments.
    The unmixing matrices are used to transform the mixed signals back into their
    source components.

    Parameters
    ----------
    A : ndarray
        Mixing matrix
    comp_list : ndarray
        Component assignments

    Returns
    -------
    W : ndarray
        Unmixing matrices
    """
    data_dim = A.shape[0]
    num_models = comp_list.shape[1]

    W = np.zeros((data_dim, data_dim, num_models))

    for h in range(num_models):
        idx = comp_list[:, h]
        W[:, :, h] = np.linalg.inv(A[:, idx])

    return W


def reject_outliers(ll, rejsig):
    """
    Identify outlier samples based on log likelihood values.

    Uses a statistical approach to identify outliers by comparing log likelihood
    values to the mean. Samples with log likelihood below (mean - rejsig * std)
    are considered outliers. This helps improve model robustness by excluding
    anomalous data points from the analysis.

    Parameters
    ----------
    ll : ndarray
        Log likelihood values
    rejsig : float
        Number of standard deviations for rejection threshold

    Returns
    -------
    mask : ndarray
        Boolean mask of non-outlier samples
    """
    ll_mean = np.mean(ll)
    ll_std = np.std(ll)
    return ll >= (ll_mean - rejsig * ll_std)
