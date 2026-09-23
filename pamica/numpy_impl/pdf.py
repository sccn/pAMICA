"""
Source densities for plotting a fitted model.

This is a plotting helper, not a fit path: :func:`compute_pdf` evaluates a
fitted source density in linear space for ``viz.plot_pdf_fits``. The fit itself
uses the log-space density in ``core.py`` (``AMICA._compute_log_pdf``), and the
generalized Gaussian here draws the same density: its normalizers come from
:mod:`pamica.reference_constants`, which every backend uses, including the
reference's single-precision ``sqrt(pi)`` at ``rho == 2`` (issue #344).

The ``pdftype`` codes below are this module's legacy numbering, not the
reference's ``pdftype``: here 1 is the generalized Gaussian (the reference's 0),
and 2, 3 and 4 are densities no backend fits. The supported PDFs are:

1. Generalized Gaussian Distribution (GGD):
   p(x) = exp(-|x|^ρ) / (2Γ(1+1/ρ))
   - ρ=1: Laplace distribution (heavy-tailed)
   - ρ=2: Gaussian distribution
   - Other ρ: Interpolates between Laplace and Gaussian

2. Logistic Distribution:
   p(x) = exp(x) / (1 + exp(x))^2
   - Symmetric, slightly heavier tails than Gaussian

3. Generalized Logistic Distribution:
   p(x) = ρ exp(x) / (1 + exp(x))^(ρ+1)
   - Asymmetric, allows modeling skewed distributions

4. Gaussian Mixture:
   p(x) = 0.5[N(0,1) + N(0,ρ)]
   - Bimodal, good for modeling multimodal data

The choice of PDF can significantly impact separation quality. The module includes
functionality to automatically select appropriate PDFs based on data statistics.
"""

import numpy as np
from scipy import special

from ..reference_constants import LOG2, LOG_SQRT_2PI, LOG_SQRT_PI


def compute_pdf(
    y: np.ndarray, rho: float, pdftype: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PDF value and its derivative for given activation values.

    For each supported PDF type, computes both the probability density p(y)
    and its derivative dp(y)/dy, for plotting (see the module docstring; the
    fit uses ``AMICA._compute_log_pdf``).

    The shape parameter ρ controls the distribution's properties:
    - For GGD: Controls tail heaviness (1=Laplace, 2=Gaussian)
    - For Gen. Logistic: Controls asymmetry
    - For Gaussian Mixture: Controls variance ratio of components

    Parameters
    ----------
    y : ndarray
        Activation values
    rho : float
        Shape parameter
    pdftype : int
        PDF type, in this module's legacy numbering (not the reference's):
        1: Generalized Gaussian
        2: Logistic
        3: Generalized Logistic
        4: Gaussian Mixture

    Returns
    -------
    pdf : ndarray
        PDF values
    dpdf : ndarray
        PDF derivatives
    """
    if pdftype == 1:
        # Generalized Gaussian
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
            pdf = np.exp(-np.power(np.abs(y), rho)) / (
                2.0 * special.gamma(1.0 + 1.0 / rho)
            )
            dpdf = -rho * np.power(np.abs(y), rho - 1) * np.sign(y) * pdf

    elif pdftype == 2:
        # Logistic distribution
        exp_y = np.exp(y)
        pdf = exp_y / np.square(1 + exp_y)
        dpdf = pdf * (1 - exp_y) / (1 + exp_y)

    elif pdftype == 3:
        # Generalized logistic
        exp_y = np.exp(y)
        pdf = rho * exp_y / np.power(1 + exp_y, rho + 1)
        dpdf = pdf * (1 - (rho + 1) * exp_y / (1 + exp_y))

    elif pdftype == 4:
        # Gaussian mixture, with the log sqrt(2*pi) the Gaussian family uses
        pdf1 = np.exp(-0.5 * y * y - LOG_SQRT_2PI)
        pdf2 = np.exp(-0.5 * y * y / rho - LOG_SQRT_2PI - 0.5 * np.log(rho))
        pdf = 0.5 * (pdf1 + pdf2)
        dpdf = -0.5 * (y * pdf1 + y * pdf2 / rho)

    else:
        raise ValueError(f"Unknown PDF type: {pdftype}")

    return pdf, dpdf


def choose_pdf_type(data: np.ndarray, rho: float = 1.5) -> int:
    """
    Choose best PDF type based on data statistics.

    Uses higher-order statistics to characterize the data distribution:
    1. Kurtosis: Measures tail heaviness
       - kurt ≈ 0: Gaussian-like
       - kurt > 0: Heavier tails
       - kurt < 0: Lighter tails
    2. Skewness: Measures asymmetry
       - skew ≈ 0: Symmetric
       - skew ≠ 0: Asymmetric

    Selection criteria:
    - Near-Gaussian (low kurt & skew): Generalized Gaussian
    - Heavy tails (high kurt): Logistic
    - High asymmetry: Generalized Logistic
    - Otherwise: Gaussian Mixture

    Parameters
    ----------
    data : ndarray
        Data samples
    rho : float
        Initial shape parameter

    Returns
    -------
    pdftype : int
        Selected PDF type
    """
    # Compute statistics
    kurt = np.mean(data**4) / np.square(np.mean(data**2)) - 3
    skew = np.mean(data**3) / np.power(np.mean(data**2), 1.5)

    if abs(kurt) < 0.5 and abs(skew) < 0.5:
        # Close to Gaussian
        return 1
    elif kurt > 2:
        # Heavy tailed
        return 2
    elif abs(skew) > 1:
        # Asymmetric
        return 3
    else:
        # Multimodal
        return 4
