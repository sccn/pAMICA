# Background

This section explains the ideas behind pamica from the ground up, for readers
new to independent component analysis.

- **[What is ICA?](what-is-ica.md)**: the blind source separation problem, the
  linear mixing model, and why statistical independence and non-Gaussianity make
  it solvable.
- **[What is AMICA?](what-is-amica.md)**: how AMICA extends ICA with *adaptive*
  source densities (mixtures of generalized Gaussians) and *multiple* ICA models.
- **[How AMICA works](how-amica-works.md)**: the log-likelihood objective and
  the expectation-maximization algorithm that fits it, one iteration at a time
  (E-step, checks, update).

If you just want to run a decomposition, start with
[Getting Started](../getting-started.md) instead.
