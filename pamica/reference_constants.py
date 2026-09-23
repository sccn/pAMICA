"""The reference's hard-coded numeric constants, at the precision it uses them.

``amica15.f90`` writes several constants as default-kind Fortran literals, which
are single precision, and widens them to double with ``dble`` (for example
``log(dble(2.506628274))``): the compiler rounds the decimal to float32, and
``dble`` widens that float32 value exactly. The binary therefore uses the
float32 rounding of each decimal, not the decimal itself (issue #344). The same
holds for a default-kind literal that initializes a double-precision variable
in ``amica15_header.f90``, such as ``epsdble = 1.0e-16``.

Every backend takes these values from this module, so they cannot drift apart
(``.rules/backend_parity.md``). The PyTorch and NumPy backends compute in
float64 and use them as they are; the MLX backend computes in float32 and casts
them there.

A literal that is a dyadic rational (``dble(2.0)``, ``dble(4.0)``) is exact in
single precision, so its constant is the exact double value. Values the
reference reads from ``input.param`` (``lrate``, ``maxrho``, ``comp_thresh``
and the rest) are user inputs that its parser reads as double, so they are not
constants and are not here.
"""

import math

import numpy as np


def single_precision_literal(value: float) -> float:
    """The value of a Fortran default-kind real literal once widened to double.

    That is the float32 rounding of ``value``, returned as a float64. Rounding
    the decimal to float64 first and then to float32 could round twice; the
    pinned tests check, with exact rational arithmetic, that it does not for
    any literal used here.
    """
    return float(np.float32(value))


# --- log-normalizers of the source densities (amica15.f90:1295-1372) -----------
# Each family's log-density is ``-cost - log(norm)``, with ``norm`` written as a
# literal in the reference.

# ``log(dble(2.0))``: the Laplace (rho == 1, :1307) and general generalized
# Gaussian (:1324) branches. Exact.
LOG2 = math.log(2.0)

# ``log(dble(4.0))``: the logistic family, pdtype 3 (:1346). Exact.
LOG4 = math.log(4.0)

# ``log(dble(1.772453851))``: the generalized Gaussian at exactly rho == 2
# (:1313), whose normalizer is sqrt(pi). The float32 rounding is 1.7724539041...,
# so this is 3.0e-8 above 0.5 * log(pi).
LOG_SQRT_PI = math.log(single_precision_literal(1.772453851))

# ``log(dble(2.506628274))``: the Gaussian family, pdtype 2 (:1333), sqrt(2*pi).
# 3.7e-10 above the log of the decimal.
LOG_SQRT_2PI = math.log(single_precision_literal(2.506628274))

# ``log(dble(4.132731354))``: the sub-Gaussian cosh family, pdtype 4 (:1359).
# 2.0e-8 above the log of the decimal.
LOG_NORM_COSH_SUB = math.log(single_precision_literal(4.132731354))

# ``log(dble(1.858073988))``: the super-Gaussian cosh family, pdtype 1 (:1371).
# 2.1e-8 below the log of the decimal.
LOG_NORM_COSH_SUP = math.log(single_precision_literal(1.858073988))

# --- the rho update ---------------------------------------------------------------
# ``epsdble = 1.0e-16`` (amica15_header.f90:73, never read from input.param): the
# rho-update accumulator zeros its ``|y|^rho * log(|y|^rho)`` term where
# ``|y|^rho`` falls below this (amica15.f90:1558-1560).
EPSDBLE = single_precision_literal(1.0e-16)
