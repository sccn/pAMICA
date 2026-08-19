"""MNE-Python-facing wrapper over pamica's AMICA backend (issue #139).

:class:`AMICAICA` fits AMICA directly from an :class:`mne.io.Raw` /
:class:`mne.Epochs` and exposes the standard MNE ICA consumer surface
(``get_sources``, ``apply``, ``get_components``, ``plot_components``,
``plot_sources``). Its core value-add is :meth:`AMICAICA.to_mne_ica`, which maps
pamica's fitted mean/sphere/unmixing into a fully-populated
:class:`mne.preprocessing.ICA`; the other methods delegate to that object, so MNE
performs the export/plot/back-projection machinery unchanged.

Multi-model AMICA (``n_models > 1``) is exposed per-model (each model is its own
single-model MNE ICA), plus a per-sample model-dominance accessor
(:meth:`AMICAICA.get_model_probability`) that MNE's ``ICA`` cannot represent
(issue #141). The pamica-specific fitted metadata MNE cannot hold -- source-
density family, GG shape, component sharing -- stays inspectable via
:meth:`AMICAICA.get_pdftype` / :meth:`get_rho` / :meth:`shared_components`
(issue #142), and the separation-quality metrics :meth:`AMICAICA.mir` /
:meth:`pmi` run directly on an MNE object (issue #143). Rank-deficient input is supported: the model is sized to the
numerical rank and exported through MNE's native ``pca_components_`` (issue #225).
"""

from typing import Optional, Union

import numpy as np
import torch

# MNE is an optional extra (`pip install pamica[mne]`); this subpackage is never
# imported by ``pamica.__init__``, so ``import pamica`` does not require it. The
# base type-check/CI env has no mne, hence the scoped ignore (issue #139).
import mne  # ty: ignore[unresolved-import]
from mne.preprocessing import ICA as _MNEICA  # ty: ignore[unresolved-import]
from mne.annotations import _annotations_starts_stops

from ..amica import AMICA
from ..torch_impl import PDFTYPE_NAMES

__all__ = ["AMICAICA", "PDFTYPE_NAMES"]


def _compute_pre_whitener(data: np.ndarray, info) -> np.ndarray:
    """Per-channel-type scaling factors, matching MNE's ``ICA``.

    MNE z-scores by channel type before PCA (``ICA._compute_pre_whitener``): one
    ``std`` per type, broadcast to that type's channels, then applied as
    ``data / pre_whitener``. Reproduced here rather than imported because MNE's
    version is private and reads the array off ``self``.

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
        Scale per channel. Types contributing no variance (a flat channel type)
        get 1.0 rather than 0, so the division cannot produce inf/NaN.
    """
    pre_whitener = np.ones((len(data), 1), dtype=np.float64)
    for _, picks in mne.channel_indices_by_type(info).items():
        if not picks:
            continue
        scale = float(np.std(data[picks]))
        if scale > 0:
            pre_whitener[picks] = scale
    return pre_whitener


class AMICAICA:
    """Fit AMICA from MNE objects and interoperate with ``mne.preprocessing.ICA``.

    The wrapper fits pamica's natural-gradient AMICA backend on the data of an
    MNE :class:`~mne.io.Raw` or :class:`~mne.Epochs` and lets MNE consume the
    result: :meth:`get_sources`, :meth:`apply`, :meth:`get_components`,
    :meth:`plot_components` and :meth:`plot_sources` all delegate to a real
    :class:`mne.preprocessing.ICA` built by :meth:`to_mne_ica`.

    For a multi-model fit (``n_models > 1``) each model is exported as its own
    single-model MNE ICA (``to_mne_ica(model_idx=...)`` / the ``model_idx``
    argument on the consumer methods), and the per-sample model dominance --
    which MNE's ``ICA`` cannot represent -- is exposed directly by
    :meth:`get_model_probability` / :meth:`plot_model_probability`.

    Separation-quality metrics (issue #133) are available directly on an MNE
    object: :meth:`mir` (Mutual Information Reduction) and :meth:`pmi` (pairwise
    mutual information between sources). The pamica-specific fitted metadata MNE
    cannot hold -- source-density family, GG shape, component sharing -- is
    inspectable via :meth:`get_pdftype` / :meth:`get_rho` / :meth:`shared_components`.

    Parameters
    ----------
    n_models : int, default=1
        Number of ICA models to learn (AMICA ``n_models``).
    n_mix : int, default=3
        Number of mixture components per source (AMICA ``n_mix``).
    random_state : int or None, default=None
        Seed for the AMICA fit (passed through as the backend ``seed``) and
        stored on the exported :class:`~mne.preprocessing.ICA`.
    device : str or torch.device, optional
        Torch device for the fit (``None`` = auto; the float64 parity backend
        falls back to CPU when auto-selection lands on MPS). See :class:`AMICA`.
    verbose : bool, default=True
        Whether the underlying :class:`AMICA` prints fit progress.

    Attributes
    ----------
    amica_ : AMICA
        The fitted pamica model (holds all ``n_models`` models).
    info_ : mne.Info
        The picked measurement info the fit was run on (channel subset only).
    ch_names_ : list of str
        Names of the fitted channels, in order.
    n_components_ : int
        Number of ICA components. Equals the number of fitted channels unless the
        data are rank-deficient, in which case the model is sized to the detected
        numerical rank (issue #223).
    pre_whitener_ : np.ndarray of shape (n_channels, 1)
        Per-channel-type scaling applied before fitting, following MNE's own ICA
        convention (one ``std`` per channel type, applied as ``X / pre_whitener_``).
    converged_ : bool
        Whether the last fit ended usable (not degenerate). A degenerate fit is
        kept for inspection but refused by the consumer methods (issue #50).
    stop_reason_ : str or None
        Why the backend fit stopped (e.g. ``"max_iter"``, ``"nan_ll"``).

    Notes
    -----
    Model ``h``'s AMICA transform is ``S = W_fort @ (sphere @ (X - mean) - c_h)``,
    where ``c_h`` is that model's data-space center (identically zero for a
    single model, since the ``c`` update is gated to ``n_models > 1``). MNE
    computes sources as
    ``S = unmixing_matrix_ @ pca_components_ @ (X / pre_whitener_ - pca_mean_)``.
    ``X`` is scaled by channel type before fitting, exactly as MNE's own ICA does,
    so the two pipelines agree; AMICA's sphering absorbs a global rescale, so this
    changes nothing for single-channel-type data. Writing the symmetric-ZCA ``sphere`` as
    ``V @ diag(1/sqrt(e)) @ V.T`` with ``V`` orthonormal, the exported ICA for
    model ``h`` uses ``pca_components_ = V.T``,
    ``unmixing_matrix_ = W_fort @ sphere @ V`` and
    ``pca_mean_ = mean + inv(sphere) @ c_h`` (which reduces to ``mean`` when
    ``c_h`` is zero). Keeping ``pca_components_`` orthonormal is what makes MNE's
    ``get_components`` (scalp maps ``inv(sphere) @ inv(W_fort)``) come out right,
    since MNE assumes orthonormal PCA rows. The mapping is pinned by a round-trip
    test (``to_mne_ica(model_idx=h).get_sources(raw)`` equals
    ``amica_.transform(X, model_idx=h)``).

    Rank-deficient input -- Maxwell-filtered MEG, average referencing, channel
    interpolation, or an explicit ``pcakeep``/``pcadb`` -- is supported. The sphere
    is then ``(n_kept, n_channels)`` and has no eigendecomposition, so the export
    takes its right singular vectors instead; MNE represents the result natively,
    since ``pca_components_`` is ``(n_components, n_channels)``. ``n_components_``
    reports the retained rank.
    """

    def __init__(
        self,
        n_models: int = 1,
        n_mix: int = 3,
        random_state: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ):
        self.n_models = n_models
        self.n_mix = n_mix
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.amica_: Optional[AMICA] = None
        self.info_ = None
        self.ch_names_: Optional[list] = None
        self.n_components_: Optional[int] = None
        self.pre_whitener_: Optional[np.ndarray] = None
        self.reject_by_annotation_: bool = False
        self.good_sample_mask_: Optional[np.ndarray] = None
        self.converged_: bool = False
        self.stop_reason_: Optional[str] = None
        self._n_samples: Optional[int] = None
        self._fit_kind: Optional[str] = None
        # Per-model export cache (one mne.preprocessing.ICA per model_idx).
        self._mne_ica_cache: dict = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        inst,
        picks=None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        reject_by_annotation: bool = True,
        **fit_kwargs,
    ) -> "AMICAICA":
        """Fit AMICA to the data of an MNE ``Raw`` or ``Epochs``.

        Parameters
        ----------
        inst : mne.io.BaseRaw | mne.BaseEpochs
            The data to decompose. ``Epochs`` are concatenated along time
            (``np.hstack``), matching how MNE's own ICA fits epoched data.
        picks : str | list | slice | None, default=None
            Channels to fit, in any form MNE accepts (e.g. ``"eeg"``,
            ``"data"``, a name list). ``None`` selects all good data channels
            (bads excluded), matching MNE's ICA default.
        start, stop : int | None
            Sample range for ``Raw`` input, passed to ``get_data``; ``None`` uses
            the full recording. Not supported for ``Epochs`` (raises).
        reject_by_annotation : bool, default=True
            If ``True``, omit time segments whose annotation description starts
            with ``"bad"`` when fitting from :class:`~mne.io.Raw`. This matches
            MNE's ``ICA.fit`` behavior. Has no effect for :class:`~mne.BaseEpochs`.    
        **fit_kwargs
            Forwarded to :meth:`AMICA.fit` (e.g. ``max_iter``, ``lrate``,
            ``do_newton``) and the backend constructor (e.g. ``block_size``).
            ``pcakeep``/``pcadb`` (PCA reduction) are supported: the export
            builds ``pca_components_`` from the reduced sphere (issue #225).

        Returns
        -------
        self : AMICAICA

        Raises
        ------
        TypeError
            If ``inst`` is not an MNE ``Raw``/``Epochs``.
        ValueError
            If ``start``/``stop`` are given for ``Epochs``, ``stop`` exceeds the
            recording length, or the selected data is non-finite.
        """
        if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError(
                "AMICAICA.fit expects an mne.io.Raw or mne.Epochs, got "
                f"{type(inst).__name__}."
            )
        is_raw = isinstance(inst, mne.io.BaseRaw)
        if not is_raw and (start is not None or stop is not None):
            raise ValueError("start/stop are only supported for Raw input, not Epochs.")

        picked = inst.copy().pick("data" if picks is None else picks, exclude="bads")
        ch_names = list(picked.ch_names)

        if is_raw:
            if stop is not None and stop > inst.n_times:
                raise ValueError(
                    f"stop={stop} exceeds the recording length ({inst.n_times} "
                    "samples)."
                )
            range_kw = {}
            if start is not None:
                range_kw["start"] = start
            if stop is not None:
                range_kw["stop"] = stop
            fit_start = 0 if start is None else start # used in good_sample_mask
            fit_stop = inst.n_times if stop is None else stop # used in good_sample_mask
            # Match MNE ICA: omit samples covered by annotations whose
            # description starts with "bad" before whitening and fitting.
            if reject_by_annotation:
                X = picked.get_data(reject_by_annotation="omit",**range_kw,)
                        # ----------------------------------------------------
                        # Build a mask on the original Raw sample axis.
                        #
                        # True  -> sample is used for AMICA fitting
                        # False -> sample is rejected by a "bad" annotation
                        #
                        # This uses MNE's own annotation-to-sample conversion
                        # and therefore does not confuse real NaN values in the
                        # input data with rejected samples.
                        # ----------------------------------------------------
                annotation_onsets, annotation_ends = (_annotations_starts_stops(inst,"bad",))
                good_sample_mask = np.ones(inst.n_times,dtype=bool,)
                for annotation_start, annotation_stop in zip(annotation_onsets,annotation_ends,):
                    annotation_start = max(annotation_start,fit_start,)
                    annotation_stop = min(annotation_stop,fit_stop,)
                    if annotation_start < annotation_stop:
                        good_sample_mask[annotation_start:annotation_stop] = False
            else:
                X = picked.get_data(reject_by_annotation=None,**range_kw,)
                # No samples are rejected.
                good_sample_mask = np.ones(inst.n_times,dtype=bool,)
            fit_kind = "raw"
        else:
            # Concatenate epochs along time, as MNE's ICA does for fitting.
            X = np.hstack(picked.get_data(reject_by_annotation=None,))
            fit_kind = "epochs"

        X = np.ascontiguousarray(X, dtype=np.float64)
        if not np.isfinite(X).all():
            bad = [ch_names[i] for i in np.flatnonzero(~np.isfinite(X).all(axis=1))]
            raise ValueError(
                "AMICAICA.fit: input contains non-finite (NaN/Inf) samples in "
                f"channel(s) {bad}; clean bad segments/interpolation before "
                "fitting."
            )

        # Channel-type scaling, matching MNE's own ICA._compute_pre_whitener:
        # one std per channel type, applied as data / pre_whitener. Required for
        # mixed magnetometer/gradiometer data, whose physical units differ by
        # orders of magnitude (issue #225). AMICA's sphering absorbs a *global*
        # rescale exactly, so for single-channel-type data (ordinary EEG) this
        # leaves the sources unchanged; it bites only across types, where it
        # decides which directions survive rank reduction.
        pre_whitener = _compute_pre_whitener(X, picked.info)
        X = X / pre_whitener

        if isinstance(self.random_state, (int, np.integer)):
            fit_kwargs.setdefault("seed", int(self.random_state))

        amica = AMICA(
            n_models=self.n_models,
            n_mix=self.n_mix,
            device=self.device,
            verbose=self.verbose,
        )
        amica.fit(X, **fit_kwargs)

        # Publish to self only after every fallible step above succeeds, so a
        # failed (re)fit leaves the previously fitted state intact rather than a
        # mix of the old model and the new attempt's metadata (mirrors the
        # local-first pattern in AMICA.fit).
        self.info_ = picked.info
        self.ch_names_ = ch_names
        self.pre_whitener_ = pre_whitener
        if is_raw:
            self.good_sample_mask_ = good_sample_mask
            self.reject_by_annotation_ = reject_by_annotation
        else:
            self.good_sample_mask_ = None
            self.reject_by_annotation_ = False
        # The fitted model dimension, which is the input channel count unless
        # rank reduction shrank it (issue #223). MNE represents this natively:
        # pca_components_ is (n_components, n_channels).
        self.n_components_ = (
            amica.model_.n_channels if amica.model_ is not None else X.shape[0]
        )
        self._n_samples = X.shape[1]
        self._fit_kind = fit_kind
        self.amica_ = amica
        self.converged_ = amica.converged_
        self.stop_reason_ = amica.stop_reason_
        self._mne_ica_cache = {}  # invalidate any cached exports
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_mne_ica(self, model_idx: int = 0) -> _MNEICA:
        """Build (and cache) a fully-populated :class:`mne.preprocessing.ICA`.

        The returned object is a genuine MNE ICA: ``get_sources``, ``apply``,
        ``get_components``, ``save`` and the ``mne.viz`` plotters operate on it
        natively. See the class :class:`Notes <AMICAICA>` for the
        mean/sphere/unmixing to ``pca_mean_``/``pca_components_``/
        ``unmixing_matrix_`` mapping.

        For a multi-model fit each model is exported as its own single-model
        MNE ICA (MNE has no multi-model concept); pass ``model_idx`` to pick
        one. The per-model exports are cached and returned by reference:
        mutating one (for example setting ``.exclude``) persists across
        subsequent :meth:`apply`/:meth:`get_sources` calls for that model until
        the next :meth:`fit`.

        Parameters
        ----------
        model_idx : int, default=0
            Which AMICA model to export (``0..n_models-1``).

        Returns
        -------
        ica : mne.preprocessing.ICA
        """
        self._check_fitted("build the MNE ICA")
        model_idx = self._check_model_idx(model_idx)
        if model_idx in self._mne_ica_cache:
            return self._mne_ica_cache[model_idx]

        amica = self.amica_
        if amica is None or amica.model_ is None or self.ch_names_ is None:
            raise RuntimeError(
                "AMICAICA: internal state is inconsistent; refit before to_mne_ica()."
            )
        backend = amica.model_
        if backend.mean is None or backend.sphere is None or backend.c is None:
            raise RuntimeError(
                "AMICAICA: the fitted backend is missing mean/sphere/c; refit "
                "before to_mne_ica()."
            )

        mean = backend.mean.cpu().numpy().ravel()
        sphere = backend.sphere.cpu().numpy()
        w_fort = amica.get_unmixing_matrix(model_idx=model_idx)
        c = backend.c.cpu().numpy()[:, model_idx]  # per-model center (sphered space)
        n_ch, n_in = sphere.shape

        if n_ch == n_in:
            # Orthonormal eigenbasis of the symmetric-ZCA sphere
            # (sphere = V diag(1/sqrt(cov_eval)) V.T). eigh gives ascending
            # sphere-eigenvalues (= 1/sqrt(cov_eval)); reorder to descending
            # explained variance so pca_components_ matches MNE's PCA convention.
            sphere_evals, evecs = np.linalg.eigh(sphere)
            cov_evals = 1.0 / sphere_evals**2
            order = np.argsort(cov_evals)[::-1]
            v = evecs[:, order]
            cov_evals = cov_evals[order]
        else:
            # Rank-reduced fit (issue #223): the sphere is (n_kept, n_in), so it
            # has no eigendecomposition. Its right singular vectors give the same
            # thing eigh gives in the square case -- an orthonormal basis of the
            # retained subspace -- and MNE models this natively, since
            # pca_components_ is (n_components, n_channels).
            _, svals, vt = np.linalg.svd(sphere, full_matrices=False)
            # Singular values of the sphere are 1/sqrt(cov eigenvalue); descending
            # explained variance is therefore ascending singular value.
            order = np.argsort(svals)
            v = vt[order].T
            cov_evals = 1.0 / svals[order] ** 2

        pca_components = v.T
        unmixing = w_fort @ sphere @ v
        # Fold the per-model center c (in sphered space) into pca_mean via the
        # data-space offset pinv(sphere) @ c, so MNE's (X - pca_mean) reproduces
        # AMICA's W(sphere(X - mean) - c). c is identically zero for a single
        # model, leaving pca_mean == mean bit-for-bit. pinv rather than solve: the
        # sphere is non-square under rank reduction, and for a square sphere the
        # two agree to round-off.
        pca_mean = mean + np.linalg.pinv(sphere) @ c if np.any(c) else mean

        ica = _MNEICA(
            n_components=n_ch,
            method="infomax",
            max_iter="auto",
            random_state=self.random_state,
        )
        # `method` is inert here: the fitted attributes are hand-populated and
        # ICA.fit (the only place `method` branches) is never called, but
        # ICA.__init__ still requires a valid method name.
        ica.info = self.info_
        ica.ch_names = list(self.ch_names_)
        ica.n_components_ = n_ch
        ica.pca_mean_ = pca_mean
        ica.pca_components_ = pca_components
        ica.pca_explained_variance_ = cov_evals
        ica.unmixing_matrix_ = unmixing
        ica.pre_whitener_ = self.pre_whitener_
        ica.n_iter_ = max(int(getattr(backend, "iteration", 0)), 1)
        # MNE's own fit sets these; read_ica_eeglab (the precedent for building
        # an ICA from an external decomposition) sets reject_=None. Without them
        # ICA.save()/plot_properties raise AttributeError.
        ica.reject_ = None
        ica.n_samples_ = int(self._n_samples) if self._n_samples is not None else 0
        ica._update_mixing_matrix()
        ica._update_ica_names()
        ica.current_fit = self._fit_kind

        self._mne_ica_cache[model_idx] = ica
        return ica

    # ------------------------------------------------------------------
    # MNE consumer surface (delegates to the exported ICA)
    # ------------------------------------------------------------------
    def get_sources(self, inst, *args, model_idx: int = 0, **kwargs):
        """Sources for ``inst`` from model ``model_idx`` (see ``ICA.get_sources``).

        ``model_idx`` is keyword-only so positional arguments pass straight
        through to MNE's ``ICA.get_sources`` (e.g. ``add_channels``,
        ``start``/``stop``).
        """
        return self.to_mne_ica(model_idx).get_sources(inst, *args, **kwargs)

    def apply(self, inst, *args, model_idx: int = 0, **kwargs):
        """Remove selected components of model ``model_idx`` and back-project.

        ``model_idx`` is keyword-only so positional arguments pass straight
        through to MNE's ``ICA.apply``. Pass ``exclude=[...]`` (or set it on the
        exported ICA) to drop components; with no exclusions this reconstructs
        the input.
        """
        return self.to_mne_ica(model_idx).apply(inst, *args, **kwargs)

    def get_components(self, *, model_idx: int = 0) -> np.ndarray:
        """Scalp maps of model ``model_idx``, ``(n_channels, n_components)``."""
        return self.to_mne_ica(model_idx).get_components()

    def plot_components(self, *args, model_idx: int = 0, **kwargs):
        """Plot model ``model_idx`` component topographies (see ``ICA.plot_components``)."""
        return self.to_mne_ica(model_idx).plot_components(*args, **kwargs)

    def plot_sources(self, inst, *args, model_idx: int = 0, **kwargs):
        """Plot model ``model_idx`` component time courses (see ``ICA.plot_sources``)."""
        return self.to_mne_ica(model_idx).plot_sources(inst, *args, **kwargs)

    # ------------------------------------------------------------------
    # Multi-model dominance (issue #141)
    # ------------------------------------------------------------------
    def get_model_probability(self, inst) -> np.ndarray:
        """Per-sample posterior probability of each model on ``inst`` (dominance).

        Returns ``P(model | sample)`` as ``(n_models, n_samples)`` via
        :meth:`AMICA.model_probability` on ``inst``'s data (``Epochs`` are
        concatenated along time). Each column sums to 1; all ones for a single
        model. MNE's own ``ICA`` has no multi-model concept, so this is exposed
        here rather than through the exported per-model ICA objects.
        """
        self._check_fitted("compute the model probability")
        if self.amica_ is None:
            raise RuntimeError(
                "AMICAICA: internal state is inconsistent; refit before scoring."
            )
        return self.amica_.model_probability(self._data_for(inst))

    def plot_model_probability(self, inst, *, srate: Optional[float] = None, **kwargs):
        """Plot per-model probability + best-model log-likelihood over ``inst``.

        Delegates to :func:`pamica.viz.plot_model_probability` with the live
        per-model log-likelihood (:meth:`AMICA.model_loglik`) on ``inst``'s
        data. ``srate`` defaults to the fitted recording's sampling rate, so the
        x-axis is in seconds; extra keywords (``smooth_sec``, ``window_sec``,
        ``axes``) pass through.
        """
        from ..viz import plot_model_probability as _plot_model_probability

        self._check_fitted("plot the model probability")
        if self.amica_ is None or self.info_ is None:
            raise RuntimeError(
                "AMICAICA: internal state is inconsistent; refit before plotting."
            )
        lht = self.amica_.model_loglik(self._data_for(inst))
        if srate is None:
            srate = float(self.info_["sfreq"])
        return _plot_model_probability(lht=lht, srate=srate, **kwargs)

    # ------------------------------------------------------------------
    # Separation-quality metrics (issue #143, on top of #133)
    # ------------------------------------------------------------------
    def mir(self, inst, *, model_idx: int = 0, nbins: Optional[int] = None) -> tuple:
        """Mutual Information Reduction of model ``model_idx`` on ``inst``.

        How much mutual information the fitted unmixing removes from the data,
        in nats (issue #133). Delegates to :meth:`AMICA.mir` on ``inst``'s
        fitted-channel data; MIR is shift-invariant, so mean/``c`` centering is
        irrelevant.

        Parameters
        ----------
        inst : mne.io.BaseRaw | mne.BaseEpochs
            Data to score (``Epochs`` concatenated along time).
        model_idx : int, default=0
            Which model's unmixing to use.
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.mir`.

        Returns
        -------
        mir_nats : float
        variance : float
        """
        self._check_fitted("compute MIR")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.mir(self._data_for(inst), model_idx=model_idx, nbins=nbins)

    def pmi(
        self, inst, *, model_idx: int = 0, nbins: Optional[int] = None
    ) -> np.ndarray:
        """Pairwise Mutual Information between model ``model_idx``'s sources on ``inst``.

        The residual pairwise dependence between fitted sources, in nats
        (issue #133). Delegates to :meth:`AMICA.pmi` on ``inst``'s fitted-channel
        data.

        Parameters
        ----------
        inst : mne.io.BaseRaw | mne.BaseEpochs
            Data to score (``Epochs`` concatenated along time).
        model_idx : int, default=0
            Which model's sources to use.
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.pairwise_mi`.

        Returns
        -------
        mi_matrix : np.ndarray of shape (n_components, n_components)
            Symmetric; the diagonal is each source's own entropy.
        """
        self._check_fitted("compute PMI")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.pmi(self._data_for(inst), model_idx=model_idx, nbins=nbins)

    # ------------------------------------------------------------------
    # pamica-specific metadata (issue #142)
    #
    # MNE's ``ICA`` carries no source-density family, GG shape, or
    # component-sharing state, so these are exposed here rather than silently
    # dropped by the ``mne.preprocessing.ICA`` export.
    # ------------------------------------------------------------------
    def get_pdftype(self, *, model_idx: int = 0) -> np.ndarray:
        """Per-component source-density family code for model ``model_idx``.

        One integer per ICA component (0-4); map to names with
        :data:`pamica.mne_compat.PDFTYPE_NAMES`. All components share one family
        unless the adaptive switcher (``pdftype=1``) moved them (issue #26).
        """
        self._check_fitted("get the density family")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.get_pdftype(model_idx=model_idx)

    def get_rho(self, *, model_idx: int = 0) -> np.ndarray:
        """Generalized-Gaussian shape ``rho`` for model ``model_idx``.

        Shape ``(n_mix, n_components)``; ``rho == 2`` is Gaussian-shaped,
        ``rho == 1`` Laplacian, ``rho < 1`` heavier-tailed. Meaningful only for
        the generalized-Gaussian family (``pdftype=0``).
        """
        self._check_fitted("get rho")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.get_rho(model_idx=model_idx)

    def shared_components(self) -> list:
        """Components shared across models by ``share_comps`` (issue #60).

        One group of ``(model_idx, component_idx)`` pairs per shared column;
        empty when nothing is shared (always so for a single model or a default
        multi-model fit with ``share_comps`` off).
        """
        self._check_fitted("get the shared components")
        assert self.amica_ is not None
        return self.amica_.shared_components()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _data_for(self, inst) -> np.ndarray:
        """The fitted-channel data of ``inst`` as ``(n_channels, n_samples)``.

        Selects the exact channels the fit used (by name) so the array aligns
        with the stored sphere/unmixing; ``Epochs`` are concatenated along time.
        The channel-type pre-whitener is applied, because the backend was fitted
        on scaled data and would otherwise be handed a differently-scaled array
        (issue #225).
        """
        if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError(
                f"expected an mne.io.Raw or mne.Epochs, got {type(inst).__name__}."
            )
        picked = inst.copy().pick(self.ch_names_)
        if isinstance(inst, mne.io.BaseRaw):
            # Always start from the complete Raw data.
            X = picked.get_data(reject_by_annotation=None)
            # Reproduce the exact sample selection used during fitting.
            if self.reject_by_annotation_:
                if self.good_sample_mask_ is None:
                    raise RuntimeError(
                        "AMICAICA: good_sample_mask_ is not available. ")
                if len(self.good_sample_mask_) != picked.n_times:
                    raise ValueError(
                        "AMICAICA: good_sample_mask_ does not match the "
                        "number of samples in the provided Raw instance.")
                X = X[:, self.good_sample_mask_]
        else:
            X = np.hstack(picked.get_data(reject_by_annotation=None))
        X = np.ascontiguousarray(X, dtype=np.float64)
        if self.pre_whitener_ is not None:
            X = X / self.pre_whitener_
        return X

    def _check_model_idx(self, model_idx: int) -> int:
        if not isinstance(model_idx, (int, np.integer)):
            raise TypeError(
                f"model_idx must be an int, got {type(model_idx).__name__}."
            )
        # Bound against the fitted backend's model count (the source of truth),
        # not the mutable constructor hyperparameter self.n_models.
        n = (
            self.amica_.model_.n_models
            if self.amica_ is not None and self.amica_.model_ is not None
            else self.n_models
        )
        if not (0 <= model_idx < n):
            raise ValueError(
                f"model_idx={model_idx} out of range for a {n}-model fit "
                f"(valid: 0..{n - 1})."
            )
        return int(model_idx)

    def _check_fitted(self, action: str = "this call") -> None:
        """Raise if no usable model is available.

        ``ValueError`` when never fitted (matching :class:`AMICA`'s convention);
        ``RuntimeError`` when the fit ended degenerate (non-finite parameters,
        issue #50), so the failure surfaces here rather than as opaque NaNs
        downstream.
        """
        if self.amica_ is None:
            raise ValueError(f"AMICAICA must be fitted before {action}; run fit().")
        if not self.converged_:
            raise RuntimeError(
                f"Refusing to {action}: the AMICA fit ended degenerate "
                f"(stop_reason={self.stop_reason_!r}), so it holds non-finite "
                "parameters and would produce NaN output. Lower lrate, disable "
                "Newton, or check data conditioning, then refit."
            )

    def __repr__(self) -> str:
        if self.amica_ is None:
            return (
                f"<AMICAICA (unfitted, n_models={self.n_models}, n_mix={self.n_mix})>"
            )
        if not self.converged_:
            return (
                f"<AMICAICA (degenerate fit, stop_reason={self.stop_reason_!r}, "
                f"n_models={self.n_models}, n_mix={self.n_mix}, {self._fit_kind})>"
            )
        return (
            f"<AMICAICA (fitted: {self.n_components_} components, "
            f"n_models={self.n_models}, n_mix={self.n_mix}, {self._fit_kind})>"
        )
