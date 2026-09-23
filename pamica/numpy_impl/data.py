"""
Data loading and preprocessing for AMICA.

This module provides functionality for loading and preprocessing data for the AMICA
algorithm. It handles:

1. Loading binary data files in Fortran format
2. Data preprocessing operations:
   - Mean removal
   - Sphering (whitening) using PCA
   - Dimensionality reduction
3. Loading saved results (raw Fortran binary format; see load_results)

The preprocessing steps are crucial for AMICA's performance:
- Mean removal centers the data, removing DC offset
- Sphering decorrelates the data and normalizes variance, which can speed up
  convergence and improve separation quality
- PCA-based dimensionality reduction can help remove noise and reduce
  computational complexity
"""

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import List, Tuple, Optional, Union


def load_data_file(
    filepath: Union[str, Path],
    data_dim: int,
    field_dim: int,
    dtype: npt.DTypeLike = np.float64,
) -> np.ndarray:
    """
    Load data from binary file in Fortran format.

    Parameters
    ----------
    filepath : str or Path
        Path to binary data file
    data_dim : int
        Number of channels/dimensions
    field_dim : int
        Number of samples per field/channel
    dtype : DTypeLike
        Data type (default: float64)

    Returns
    -------
    data : ndarray
        Loaded data array of shape (data_dim, field_dim)
    """
    # Read entire file at once
    with open(filepath, "rb") as f:
        # Read all data and reshape considering Fortran order
        data = np.fromfile(f, dtype=dtype)
        # The file stores a (data_dim, field_dim) array in column-major
        # (Fortran) order, i.e. channel-fastest within each sample. Reshape
        # directly to (data_dim, field_dim) with order='F'; no transpose is
        # needed (transposing after a (field_dim, data_dim) reshape reads
        # the wrong elements for any non-square data_dim != field_dim).
        data = data.reshape((data_dim, field_dim), order="F")

    return data


def load_multiple_files(
    filepaths: List[str],
    data_dim: int,
    field_dims: List[int],
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """
    Load and concatenate data from multiple binary files.

    Parameters
    ----------
    filepaths : list of str
        Paths to binary data files
    data_dim : int
        Number of channels/dimensions
    field_dims : list of int
        Number of samples per field for each file
    dtype : DTypeLike
        Data type (default: float64)

    Returns
    -------
    data : ndarray
        Concatenated data array of shape (data_dim, total_samples)
    """
    # Calculate total samples
    total_samples = sum(field_dims)

    # Allocate output array
    data = np.zeros((data_dim, total_samples), dtype=np.dtype(dtype))

    # Load each file
    idx = 0
    for filepath, field_dim in zip(filepaths, field_dims):
        file_data = load_data_file(filepath, data_dim, field_dim, dtype)
        data[:, idx : idx + field_dim] = file_data
        idx += field_dim

    return data


def preprocess_data(
    data: np.ndarray,
    do_mean: bool = True,
    do_sphere: bool = True,
    do_approx_sphere: bool = True,
    pcakeep: Optional[int] = None,
    pcadb: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data by removing mean and applying sphering transformation.

    This function performs two main preprocessing steps:
    1. Mean removal: Subtracts the temporal mean from each channel
    2. Sphering: Applies a linear transformation to decorrelate the signals
       and normalize their variance. This can be done either exactly or
       approximately:
       - Exact: W = Λ^(-1/2) E^T where Λ, E are eigenvalues/vectors of covariance
       - Approximate: W = Λ^(-1/2) E^T (faster but less precise)

    Optionally reduces dimensionality by keeping only the top components based on
    either a fixed number (pcakeep) or an energy threshold (pcadb).

    Parameters
    ----------
    data : ndarray
        Input data array of shape (channels, samples)
    do_mean : bool
        Whether to remove mean
    do_sphere : bool
        Whether to perform sphering
    do_approx_sphere : bool
        Whether to use approximate sphering
    pcakeep : int, optional
        Number of components to keep
    pcadb : float, optional
        dB threshold for keeping components

    Returns
    -------
    data : ndarray
        Preprocessed data
    mean : ndarray
        Data mean (or zeros if do_mean=False)
    sphere : ndarray
        Sphering matrix (or identity if do_sphere=False)
    """
    data_dim = data.shape[0]

    # Remove mean if requested
    if do_mean:
        mean = np.mean(data, axis=1, keepdims=True)
        data = data - mean
    else:
        mean = np.zeros((data_dim, 1))

    # Compute sphering matrix if requested
    if do_sphere:
        # Compute covariance
        cov = np.cov(data)

        # Eigenvalue decomposition
        evals, evecs = np.linalg.eigh(cov)

        # Sort in descending order
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Determine number of components to keep
        if pcakeep is not None:
            n_comp = min(pcakeep, len(evals))
        elif pcadb is not None:
            db = 10 * np.log10(evals / evals[0])
            n_comp = np.sum(db > -pcadb)
        else:
            n_comp = len(evals)

        # Create sphering matrix
        if do_approx_sphere:
            # Approximate sphering (faster but less accurate)
            sphere = np.dot(np.diag(1.0 / np.sqrt(evals[:n_comp])), evecs[:, :n_comp].T)
        else:
            # Exact sphering
            sphere = np.linalg.inv(
                np.dot(np.diag(np.sqrt(evals[:n_comp])), evecs[:, :n_comp].T)
            )

        # Apply sphering
        data = np.dot(sphere, data)
    else:
        sphere = np.eye(data_dim)

    return data, mean, sphere


def load_results(indir: Union[str, Path], compressed: bool = False) -> dict:
    """
    Load saved AMICA results from disk.

    Reads the raw little-endian binary files written by ``AMICA._write_results``
    -- the Fortran AMICA output layout that ``load.loadmodout`` and the
    reference binary use (issue #30) -- and returns them in AMICA's internal
    array shapes, which the ``viz`` helpers consume.

    Parameters
    ----------
    indir : str or Path
        Input directory containing saved results.
    compressed : bool, default=False
        Accepted for call-site compatibility and ignored: the on-disk format is
        uncompressed raw binary (issue #30), not ``.npz``.

    Returns
    -------
    results : dict
        Dictionary of loaded parameters (``A``, ``W``, ``c``, ``mu``, ``alpha``,
        ``beta``, ``rho``, ``gm``, ``mean``, ``sphere``, ``comp_list``, ``ll``).
    """
    del compressed  # legacy no-op; the format is raw binary, never .npz
    indir = Path(indir)

    def _read(name, dtype: type = np.float64):
        path = indir / name
        return np.fromfile(path, dtype=dtype) if path.exists() else None

    gm = _read("gm")
    if gm is None:
        raise FileNotFoundError(f"No AMICA output in {indir} (missing 'gm')")
    num_models = len(gm)

    W = _read("W")
    if W is None:
        raise FileNotFoundError(f"No 'W' in {indir}")
    nw = int(round(np.sqrt(len(W) / num_models)))
    num_comps = nw * num_models

    # On disk W is genuine-Fortran: W_fortran(nw, nw, num_models), model axis
    # slowest, column-major within each model. Unlike loadmodout (which returns
    # that EEGLAB-convention W_fortran), load_results returns the internal-backend
    # W = W_fortran.T for the NumPy viz helpers, so read F-order and transpose the
    # first two axes back. For num_models == 1 this equals the old plain C-order
    # read; for num_models > 1 the old read scrambled the model axis (issue #159).
    W_fortran = W.reshape(nw, nw, num_models, order="F")
    results = {"gm": gm, "W": np.transpose(W_fortran, (1, 0, 2))}

    A = _read("A")
    if A is not None:
        results["A"] = A.reshape(
            len(A) // num_comps, num_comps
        )  # (data_dim, num_comps)

    # Mixture params are stored (num_mix, num_comps) column-major (Fortran names
    # 'sbeta'); reshape order="F" matches the write_amicaout writer and Fortran
    # output (a C-order read would scramble the non-square layout, issue #92).
    for fname, key in (
        ("alpha", "alpha"),
        ("mu", "mu"),
        ("sbeta", "beta"),
        ("rho", "rho"),
    ):
        arr = _read(fname)
        if arr is not None:
            results[key] = arr.reshape(-1, num_comps, order="F")

    comp_list = _read("comp_list", dtype=np.int32)
    if comp_list is not None:
        # Stored 1-based (Fortran convention); restore AMICA's 0-based indices.
        results["comp_list"] = comp_list.reshape(nw, num_models, order="F") - 1

    c = _read("c")
    if c is not None:
        results["c"] = c.reshape(nw, num_models, order="F")

    mean = _read("mean")
    if mean is not None:
        results["mean"] = mean

    # The sphere is (nw, nx): square unless rank reduction kept nw < nx
    # dimensions, in which case write_amicaout pads it with zero rows to the
    # reference's (nx, nx) record and writes it column-major (a square one is
    # written C-order). Read it back the way it was written.
    S = _read("S")
    if S is not None:
        nx = int(round(np.sqrt(len(S))))
        if nx == nw:
            results["sphere"] = S.reshape(nw, nw)
        else:
            results["sphere"] = S.reshape(nx, nx, order="F")[:nw]

    ll = _read("LL")
    if ll is not None:
        results["ll"] = ll

    # Fail loudly on a partially-written / interrupted output directory rather
    # than letting a downstream KeyError surface deep in a viz helper.
    required = {"A", "W", "alpha", "mu", "beta", "rho", "comp_list", "ll"}
    missing = required - results.keys()
    if missing:
        raise FileNotFoundError(
            f"Incomplete AMICA output in {indir}: missing {sorted(missing)}. "
            "The directory may be from an interrupted or partial run."
        )

    return results
