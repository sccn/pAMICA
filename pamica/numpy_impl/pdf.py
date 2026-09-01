"""
Probability Density Function (PDF) implementations for AMICA.

This module implements various probability density functions used in the AMICA
algorithm for modeling source distributions. The supported PDFs are:

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


def compute_pdf(
    y: np.ndarray, rho: float, pdftype: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PDF value and its derivative for given activation values.

    For each supported PDF type, computes both the probability density p(y)
    and its derivative dp(y)/dy. These are used in the AMICA algorithm for:
    1. Computing data likelihood during model fitting
    2. Gradient calculations for parameter updates
    3. Estimating component responsibilities

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
        PDF type:
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
            # Laplace distribution
            pdf = np.exp(-np.abs(y)) / 2.0
            dpdf = -np.sign(y) * pdf
        elif rho == 2.0:
            # Gaussian distribution
            pdf = np.exp(-y * y) / np.sqrt(np.pi)
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
        # Gaussian mixture
        pdf1 = np.exp(-0.5 * y * y) / np.sqrt(2 * np.pi)
        pdf2 = np.exp(-0.5 * y * y / rho) / np.sqrt(2 * np.pi * rho)
        pdf = 0.5 * (pdf1 + pdf2)
        dpdf = -0.5 * (y * pdf1 + y * pdf2 / rho)

    else:
        raise ValueError(f"Unknown PDF type: {pdftype}")

    return pdf, dpdf


def compute_log_pdf(
    y: np.ndarray, rho: float, pdftype: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute logarithm of PDF value and its derivative.

    Working in log space provides better numerical stability, especially for:
    1. Very small probability densities
    2. Product of probabilities (becomes sum of log probabilities)
    3. Computing likelihood ratios

    The derivative of log PDF (score function) is used in natural gradient
    calculations for more efficient optimization.

    Parameters
    ----------
    y : ndarray
        Activation values
    rho : float
        Shape parameter
    pdftype : int
        PDF type (see compute_pdf for details)

    Returns
    -------
    log_pdf : ndarray
        Log PDF values
    dlog_pdf : ndarray
        Log PDF derivatives
    """
    pdf, dpdf = compute_pdf(y, rho, pdftype)
    log_pdf = np.log(pdf)
    dlog_pdf = dpdf / pdf
    return log_pdf, dlog_pdf


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
