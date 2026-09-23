"""Top-level, backend-agnostic visualizations for AMICA output (issue #136).

Unlike ``numpy_impl.viz`` (numpy-backend-scoped, built on the legacy
``load_results`` dict, returns ``None`` and mutates pyplot global state), the
functions here consume ``AmicaOutput`` from
:func:`pamica.numpy_impl.load.loadmodout` -- so they work for any backend's
written amicaout directory -- and **return a `Figure`**, accepting an optional
``ax``/``axes`` to draw on. That is what makes them testable beyond "did not
crash". ``numpy_impl/viz.py`` is left untouched; this module is not a
replacement for it.

Both plots mirror ``postAmicaUtility``'s ``modprobplot``/``pop_modPMI`` MATLAB
behavior, observed by running the real GPL-licensed functions and reading
their rendered output and ``help`` text, never their source, per the project's
clean-room posture for GPL code. Every quantity they draw is pinned to a MATLAB
oracle; see ``.context/issue-136/matlab_viz_verification.md``.

A per-component scalp-topography view is deliberately NOT here: it was the
only planned plot with no working upstream reference (``pop_topohistplot`` is
broken on current EEGLAB), so nothing external could catch a wrong activation
space. It is cut rather than shipped unverified; the loader ``W`` convention it
also depended on was fixed in issue #159. The MNE wrapper's
``AMICAICA.plot_components`` draws scalp maps through MNE instead.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal.windows import hann
from scipy.special import logsumexp

from .metrics import block_diagonal_order
from .numpy_impl.load import AmicaOutput


def plot_pmi_heatmap(
    mi_matrix: np.ndarray,
    *,
    order: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    model: int | None = None,
    mask_diagonal: bool = True,
    ax: Axes | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Square components-x-components pairwise-MI heatmap (``pop_modPMI`` view).

    Parameters
    ----------
    mi_matrix : np.ndarray
        (n, n) symmetric mutual-information matrix, e.g. from
        `pamica.metrics.pairwise_mi`.
    order : np.ndarray, optional
        Length-n permutation to reorder both axes before plotting. Defaults to
        `pamica.metrics.block_diagonal_order(mi_matrix)`.
    labels : sequence of str, optional
        Per-original-component labels; tick label at reordered position ``k``
        is ``labels[order[k]]``. Defaults to the original (0-based) component
        index, matching MATLAB's "original component indices in reordered
        position" tick convention.
    model : int, optional
        0-based model index, used only for the title ("Model N", 1-based, to
        match MATLAB's per-model panel titles). Omit for a single, unlabeled
        heatmap.
    mask_diagonal : bool, default True
        `pairwise_mi`'s diagonal is each component's self-entropy (~2.83 on
        real data), not a mutual information -- an order of magnitude above
        the ~0.06 off-diagonal values. Left unmasked it blows out the color
        scale and hides all off-diagonal structure, so this defaults to True.
        Deliberate divergence: MATLAB instead zeroes its diagonal. Masking is
        the closer analogue here because our diagonal is not a small value, it
        is a different quantity, and zeroing it would fabricate an MI of 0
        where none was measured. Do not "align" this with MATLAB by zeroing
        the diagonal of the matrix itself: `pairwise_mi`'s return value feeds
        other consumers, and this is a display choice, not a data one.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure+axes is created if omitted.
    cmap : str, default "viridis"
        Colormap name. A colorbar is always added (a deliberate improvement;
        MATLAB's `pop_modPMI` has none).

    Returns
    -------
    Figure
        The figure the heatmap was drawn on (``ax.figure`` if `ax` was given).

    Raises
    ------
    ValueError
        If `mi_matrix` is not square.
    """
    mi_matrix = np.asarray(mi_matrix)
    if mi_matrix.ndim != 2 or mi_matrix.shape[0] != mi_matrix.shape[1]:
        raise ValueError(
            f"plot_pmi_heatmap: mi_matrix must be square, got shape {mi_matrix.shape}"
        )
    n = mi_matrix.shape[0]

    # Validate unconditionally, not only on the default-order branch. Passing an
    # explicit `order` used to skip this entirely (the check rode along inside
    # `block_diagonal_order`), so a non-finite matrix from an upstream failure
    # rendered silently: imshow draws NaN as a blank cell, which reads as "no
    # dependency measured here" rather than "the computation broke".
    if not np.all(np.isfinite(mi_matrix)):
        raise ValueError(
            "plot_pmi_heatmap: mi_matrix contains non-finite values (NaN/Inf); "
            "imshow would render these as blank cells indistinguishable from a "
            "genuine near-zero mutual information. Check the pairwise_mi input "
            "(e.g. a constant or non-finite source)."
        )

    if order is None:
        order = block_diagonal_order(mi_matrix)
    order = np.asarray(order)

    reordered = mi_matrix[np.ix_(order, order)]
    if mask_diagonal:
        reordered = np.ma.array(reordered, mask=np.eye(n, dtype=bool))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
        # ax.figure is typed Figure | SubFigure (an Axes can live in a
        # SubFigure); every caller here uses a real top-level Figure.
        assert isinstance(fig, Figure)

    im = ax.imshow(reordered, cmap=cmap)
    ticks = np.arange(n)
    if labels is not None:
        tick_labels = [str(labels[i]) for i in order]
    else:
        tick_labels = [str(int(i)) for i in order]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_title("Pairwise MI" if model is None else f"Model {model + 1}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mutual information (nats)")

    return fig


def plot_model_probability(
    out: AmicaOutput | None = None,
    *,
    lht: np.ndarray | None = None,
    srate: float | None = None,
    smooth_sec: float | None = None,
    window_sec: float | None = None,
    axes: Sequence[Axes] | None = None,
) -> Figure:
    """Two-panel per-model probability + best-model log-likelihood (``modprobplot``).

    Top panel: `softmax(Lht)` over models, one line per model ("Probability of
    Model Being Active"). Bottom panel: the per-sample log-likelihood of the
    single most probable model at each timepoint (``Lht.max(axis=0)``), not
    the total ``Lt``, matching the observed MATLAB behavior.

    Provide exactly one source of ``Lht``: a written ``out`` (an
    :class:`AmicaOutput`), or a live ``lht`` array (for example from
    :meth:`pamica.AMICA.model_loglik`, so an MNE-side fit can plot dominance
    without writing an ``amicaout`` directory, issue #141).

    Parameters
    ----------
    out : AmicaOutput, optional
        Must have `out.Lht` populated (per-model per-sample log-likelihood;
        see `pamica.numpy_impl.load.loadmodout`). Mutually exclusive with `lht`.
    lht : np.ndarray of shape (n_models, n_samples), optional
        A per-model per-sample log-likelihood array. Mutually exclusive with
        `out`.
    srate : float, optional
        Sampling rate in Hz. pamica has no built-in notion of sample rate
        (`AmicaOutput`/`loadmodout`/`load_data_file` carry none), so a seconds
        x-axis is opt-in: without it, the x-axis is in samples ("Sample").
        Use `pamica.numpy_impl.load.read_eeglab_set_metadata` to get a real
        recording's srate.
    smooth_sec : float, optional
        Hanning-smoothing window, in seconds (requires `srate`). Applied to
        `Lht` BEFORE the softmax (order matters: smoothing the already-
        normalized probabilities would not reproduce the same result).
    window_sec : float, optional
        Initial x-axis view width, in seconds (requires `srate`), analogous to
        `modprobplot`'s default 20 s scrolling window -- this static figure
        cannot scroll, so it is applied as a fixed initial ``xlim`` instead.
    axes : sequence of 2 Axes, optional
        ``(top, bottom)`` axes to draw on. A new figure+axes is created if
        omitted.

    Returns
    -------
    Figure
        The figure the two panels were drawn on.

    Raises
    ------
    ValueError
        If neither or both of `out`/`lht` are given, if `out.Lht` is `None`, if
        `lht` is not 2-D, or if `smooth_sec`/`window_sec` is given without
        `srate`.
    """
    if lht is not None and out is not None:
        raise ValueError(
            "plot_model_probability: provide exactly one of `out` (an "
            "AmicaOutput with Lht) or `lht` (a (n_models, n_samples) array)."
        )
    if lht is not None:
        Lht = np.asarray(lht, dtype=np.float64)
    elif out is not None:
        if out.Lht is None:
            raise ValueError(
                "plot_model_probability: out.Lht is None (no LLt saved with this "
                "fit); refit with a backend that writes LLt (see "
                "AMICA.write_amica_output) or pass an AmicaOutput loaded from a "
                "directory that has it."
            )
        Lht = np.asarray(out.Lht, dtype=np.float64)
    else:
        raise ValueError(
            "plot_model_probability: provide `out` (an AmicaOutput with Lht) or "
            "`lht` (a (n_models, n_samples) array)."
        )
    if Lht.ndim != 2:
        raise ValueError(
            "plot_model_probability: lht must be 2-D (n_models, n_samples), got "
            f"shape {Lht.shape}."
        )
    if smooth_sec is not None and srate is None:
        raise ValueError(
            "plot_model_probability: smooth_sec requires srate (pamica has "
            "no built-in sample rate; pass srate explicitly, e.g. from "
            "read_eeglab_set_metadata)."
        )
    if window_sec is not None and srate is None:
        raise ValueError(
            "plot_model_probability: window_sec requires srate (it is a "
            "seconds-based x-axis width; pass srate explicitly)."
        )

    n_models, n_samples = Lht.shape

    if smooth_sec is not None:
        assert srate is not None  # guaranteed by the eager check above
        window = int(round(srate * smooth_sec))
        if window < 1:
            raise ValueError(
                f"plot_model_probability: smooth_sec={smooth_sec} at "
                f"srate={srate} gives a {window}-sample window (< 1); "
                "increase smooth_sec or srate."
            )
        if window > n_samples:
            raise ValueError(
                f"plot_model_probability: smooth_sec={smooth_sec} at "
                f"srate={srate} gives a {window}-sample window, longer than "
                f"the {n_samples}-sample recording; reduce smooth_sec."
            )
        # MATLAB hanning(n): no zero endpoints (unlike np.hanning/scipy's
        # symmetric hann(n)), so build it via a length-(n+2) symmetric window
        # with the zero endpoints sliced off.
        w = hann(window + 2, sym=True)[1:-1]

        # Edge-correct via window-overlap normalization: a naive
        # convolve(..., mode="same") zero-pads beyond the data, and since Lht
        # sits around -100 (nowhere near 0), that padding drags the first/last
        # ~window/2 samples violently toward zero -- confidently WRONG model
        # probabilities at the start/end of every plot after the softmax.
        # Dividing by the same window convolved with a ones-signal renormalizes
        # each output sample by the fraction of the window actually inside the
        # data. Verified against the MATLAB smoothing oracle
        # (`smooth_amica_prob`) on a real 2-model fit. Name which quantity is
        # which, because two get compared and their numbers differ: this
        # SMOOTHED LOG-LIKELIHOOD correlates at 0.9939 (1 s window) / 0.9836
        # (5 s), while the PROBABILITIES it becomes after the softmax below
        # correlate at 0.9886 / 0.9594. Both are recorded in
        # `.context/issue-136/matlab_viz_verification.md`; an unlabeled 0.994
        # here previously read as a stale copy of the 0.9886 figure (#136).
        #
        # Known, accepted divergence: MATLAB additionally pins its first/last
        # sample to the raw unsmoothed input (a MATLAB smooth() idiom); this
        # returns a locally-averaged value there instead. NOT replicated on
        # purpose -- matching it would require reading GPL source, and pinning
        # the very first sample of a "smoothed" signal to the raw value is
        # arguably wrong. Do not tune this toward MATLAB.
        def _edge_corrected_smooth(row: np.ndarray) -> np.ndarray:
            num = np.convolve(row, w, mode="same")
            den = np.convolve(np.ones_like(row), w, mode="same")
            return num / den

        Lht_smooth = np.array([_edge_corrected_smooth(row) for row in Lht])
    else:
        Lht_smooth = Lht

    v_smooth = np.exp(Lht_smooth - logsumexp(Lht_smooth, axis=0, keepdims=True))
    ll_best = Lht_smooth.max(axis=0)

    if srate is not None:
        t = np.arange(n_samples) / srate
        xlabel = "Time (s)"
    else:
        t = np.arange(n_samples)
        xlabel = "Sample"

    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    else:
        fig = axes[0].figure
        assert isinstance(fig, Figure)
    ax_top, ax_bottom = axes

    for h in range(n_models):
        ax_top.plot(t, v_smooth[h], label=f"Model {h + 1}")
    ax_top.set_ylabel("P(model active)")
    ax_top.set_ylim(0, 1)
    ax_top.set_title("Probability of Model Being Active")
    ax_top.legend()

    ax_bottom.plot(t, ll_best)
    ax_bottom.set_ylabel("Log-likelihood")
    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_title("Log-likelihood of data under most probable model")

    if window_sec is not None:
        ax_top.set_xlim(0, window_sec)
        ax_bottom.set_xlim(0, window_sec)

    fig.tight_layout()
    return fig
