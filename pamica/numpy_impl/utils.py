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


def identify_shared_components(A, W, comp_list, comp_thresh=0.99):
    """
    Identify components that are shared between different models based on correlation.

    This function analyzes the mixing matrix columns to find components that are
    highly correlated between different models, suggesting they represent the same
    underlying source. When shared components are identified, they are merged to
    maintain a more parsimonious representation.

    Parameters
    ----------
    A : ndarray
        Mixing matrix
    W : ndarray
        Unmixing matrices
    comp_list : ndarray
        Component assignments
    comp_thresh : float
        Correlation threshold for identifying shared components

    Returns
    -------
    comp_list : ndarray
        Updated component assignments
    comp_used : ndarray
        Boolean mask of used components
    """
    num_models = W.shape[2]
    num_comps = A.shape[1]
    data_dim = A.shape[0]

    # Compare components between models
    for h1 in range(num_models):
        for h2 in range(h1 + 1, num_models):
            for i1 in range(data_dim):
                for i2 in range(data_dim):
                    k1 = comp_list[i1, h1]
                    k2 = comp_list[i2, h2]

                    # Skip if components are already identified
                    if k1 == k2:
                        continue

                    # Compute correlation
                    corr = np.abs(np.dot(A[:, k1], A[:, k2])) / (
                        np.sqrt(np.sum(A[:, k1] ** 2) * np.sum(A[:, k2] ** 2))
                    )

                    if corr >= comp_thresh:
                        # Check if components appear together in any model
                        shared = False
                        for h in range(num_models):
                            if k1 in comp_list[:, h] and k2 in comp_list[:, h]:
                                shared = True
                                break

                        if not shared:
                            # Merge components
                            comp_list[comp_list == k2] = k1

    # Derive comp_used from the final comp_list rather than tracking it during
    # the merge loop. A fresh np.ones() per call forgot every column merged away
    # in an earlier call: once comp_list is fully merged the k1 == k2 guard skips
    # every pair, and the mask came back all-True while half the columns were
    # dead (issue #240). Matches AMICATorchNG.comp_used, which is a property
    # derived the same way.
    comp_used = np.zeros(num_comps, dtype=bool)
    comp_used[np.unique(comp_list)] = True

    return comp_list, comp_used


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
