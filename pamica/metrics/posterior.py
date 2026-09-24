"""Shared multi-model posterior helper (issue #141; PR #329 review of #306).

Factored out of ``AMICATorchNG.model_probability``/``AMICAMLXNG.
model_probability`` so the NaN-vs-``-inf`` diagnosis and the column-wise
softmax normalization are not duplicated: both backends compute ``Lht`` (the
per-model, per-sample log-likelihood) themselves, in whatever array library
they use, then hand the resulting numpy array to this one pure function --
the same pattern already established by ``pamica.metrics.mir``/
``pairwise_mi``, which the backends also call after doing their own
backend-specific setup.
"""

import numpy as np


def model_probability_from_loglik(Lht: np.ndarray, *, caller: str) -> np.ndarray:
    """Column-wise softmax over models of ``Lht`` (model dominance), i.e.
    ``P(model h | x_t)``; each column sums to 1. All ones for a single model.

    Parameters
    ----------
    Lht : np.ndarray of shape (n_models, n_samples)
        Per-model, per-sample log-likelihood, e.g. from
        ``AMICATorchNG.model_loglik``/``AMICAMLXNG.model_loglik``.
    caller : str
        Prefix for the error message, so the raised exception still names
        which backend/method it came from (e.g.
        ``"AMICATorchNG.model_probability()"``).

    Returns
    -------
    prob : np.ndarray of shape (n_models, n_samples)

    Raises
    ------
    ValueError
        If a log-likelihood is NaN (numerical corruption) at some sample, or
        if every model underflows to ``-inf`` at some sample (the posterior
        is undefined there). The two are diagnosed and reported separately:
        NaN and ``-inf`` are different failure modes, and reporting both
        under one "-inf" message misdiagnoses the NaN case (PR #311 review
        scope extension, issue #306).
    """
    col_max = Lht.max(axis=0, keepdims=True)
    if not np.isfinite(col_max).all():
        nan_mask = np.isnan(col_max)
        if nan_mask.any():
            raise ValueError(
                f"{caller}: {int(nan_mask.sum())} sample(s) have a NaN "
                "log-likelihood (numerical corruption), so the posterior is "
                "undefined there."
            )
        n_bad = int((~np.isfinite(col_max)).sum())
        raise ValueError(
            f"{caller}: every model has -inf log-likelihood at {n_bad} "
            "sample(s), so the posterior is undefined there (an extreme "
            "outlier under a tight source density)."
        )
    ex = np.exp(Lht - col_max)
    return ex / ex.sum(axis=0, keepdims=True)
