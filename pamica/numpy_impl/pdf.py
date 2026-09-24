"""
Source density for plotting a fitted model.

This is a plotting helper, not a fit path: :func:`compute_pdf` evaluates a
fitted generalized Gaussian source density in linear space for
``viz.plot_pdf_fits``. The fit itself uses the log-space density in ``core.py``
(``AMICA._compute_log_pdf``), and this module draws the same density: its
normalizers come from :mod:`pamica.reference_constants`, which every backend
uses, including the reference's single-precision ``sqrt(pi)`` at ``rho == 2``
(issue #344).

The generalized Gaussian with shape ``rho`` is
``p(y) = exp(-|y|^rho) / (2 Gamma(1 + 1/rho))``: ``rho == 1`` is the Laplace
density, ``rho == 2`` the Gaussian, and values between them interpolate. It is
the only source density the NumPy backend fits (``pdftype=0``).
"""

import numpy as np
from scipy import special

from ..reference_constants import LOG2, LOG_SQRT_PI


def compute_pdf(y: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalized Gaussian density and its derivative at ``y``.

    Computes the probability density p(y) and its derivative dp(y)/dy for
    plotting (see the module docstring; the fit uses
    ``AMICA._compute_log_pdf``). The shape ``rho`` controls tail heaviness:
    1 is the Laplace density and 2 the Gaussian.

    Parameters
    ----------
    y : ndarray
        Activation values
    rho : float
        Shape parameter

    Returns
    -------
    pdf : ndarray
        PDF values
    dpdf : ndarray
        PDF derivatives
    """
    if rho == 1.0:
        # Laplace distribution, the fit's exp(-|y| - log 2)
        pdf = np.exp(-np.abs(y) - LOG2)
        dpdf = -np.sign(y) * pdf
    elif rho == 2.0:
        # Gaussian distribution, with the reference's single-precision
        # normalizer log(dble(1.772453851)) (amica15.f90:1313), as in the fit
        pdf = np.exp(-y * y - LOG_SQRT_PI)
        dpdf = -2 * y * pdf
    else:
        # General case: p(y) = exp(-|y|^rho) / (2 * Gamma(1 + 1/rho)).
        # gamma, NOT gammaln: the Fortran reference computes this in LOG
        # space (`- gamln(1+1/rho) - log(2)`, amica15.f90:1323-1324), where
        # log-gamma is correct; transcribing that gamln into this
        # linear-space expression divided by log(Gamma(...)) instead of
        # Gamma(...). That is negative for every rho in (1, 2) -- at the
        # default rho0=1.5 the "density" integrated to -8.82 -- so the
        # curve was wrong for every rho outside the special-cased 1 and 2.
        # The fit path was never affected: it uses core.py's own log-space
        # _compute_log_pdf, which mirrors the Fortran correctly.
        pdf = np.exp(-np.power(np.abs(y), rho)) / (2.0 * special.gamma(1.0 + 1.0 / rho))
        dpdf = -rho * np.power(np.abs(y), rho - 1) * np.sign(y) * pdf

    return pdf, dpdf
