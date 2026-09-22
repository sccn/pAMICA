"""NumPy backend params-file surface (issue #304).

``AMICA_NumPy`` used to accept only JSON parameter files -- ``params_file=...``
routed straight to ``json.load``, so handing it the literal Fortran
``input.param`` text produced a raw ``json.JSONDecodeError`` rather than
either support or a named error. It now reads through the same
``pamica.fortran_params.read_params_file`` every backend uses (both the
bundled default ``params.json`` and a user-supplied file), maps the result to
this backend's own attribute spellings, and warns once by name about any
setting it does not consume. A ``pdftype`` other than 0 -- the only
source-density family this backend implements -- raises
``NotImplementedError`` at construction instead of silently doing nothing
(the legacy behavior the bundled ``params.json`` shipped with: its default
``pdftype`` was 1, even though the fit path never read it).
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from pamica import AMICA_NumPy as AMICA
from pamica.fortran_params import read_params_file
from pamica.numpy_impl.core import _CANONICAL_TO_NUMPY_KEY
from pamica.numpy_impl.data import load_data_file

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
PARAM_FILE = SAMPLE_DIR / "input.param"
JSON_FILE = SAMPLE_DIR / "sample_params.json"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"

pytestmark = pytest.mark.skipif(
    not PARAM_FILE.exists() or not JSON_FILE.exists() or not DATA_FILE.exists(),
    reason="bundled sample_data/input.param, sample_params.json or eeglab_data.fdt missing",
)


def _shared_numpy_hyperparam_attrs() -> list:
    """Canonical keys ``read_params_file`` returns for *both* bundled files,
    minus data-location metadata (not a hyperparameter, and not stored under
    a plain same-named attribute), mapped to this backend's own attribute
    spellings. Derived from the parsed files themselves rather than
    hand-listed, so it cannot drift from the anti-drift guarantee
    ``test_fortran_params.py`` already proves at the params-file layer."""
    from_json = read_params_file(JSON_FILE)
    from_param = read_params_file(PARAM_FILE)
    # Data-location metadata, plus outdir: both bundled files happen to set
    # the same "./amicaout/" value, but the constructions below pass an
    # explicit outdir= override (test hygiene -- never write into the repo),
    # which would make this comparison spuriously fail on that key alone.
    excluded = {"files", "data_dim", "field_dim", "num_samples", "outdir"}
    shared = (set(from_json) & set(from_param)) - excluded
    return sorted(_CANONICAL_TO_NUMPY_KEY.get(k, k) for k in shared)


def _real_data(n_samples: int) -> np.ndarray:
    data = load_data_file(str(DATA_FILE), 32, 30504, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


class TestNumpyFromBothParamsFormats:
    """Gate 3: NumPy from ``input.param`` equals NumPy from
    ``sample_params.json`` on every shared hyperparameter attribute."""

    def test_shared_hyperparameters_agree(self, tmp_path):
        from_json = AMICA(
            params_file=str(JSON_FILE),
            outdir=str(tmp_path / "json_out"),
            use_tqdm=False,
        )
        from_param = AMICA(
            params_file=str(PARAM_FILE),
            outdir=str(tmp_path / "param_out"),
            use_tqdm=False,
        )
        shared_attrs = _shared_numpy_hyperparam_attrs()
        assert len(shared_attrs) >= 40  # sanity: most settings really do overlap
        mismatched = {
            attr: (getattr(from_param, attr), getattr(from_json, attr))
            for attr in shared_attrs
            if getattr(from_param, attr) != getattr(from_json, attr)
        }
        assert mismatched == {}

    def test_fortran_text_fit_runs_with_no_explicit_data(self, tmp_path, monkeypatch):
        """input.param's own ``files``/``data_dim``/``field_dim`` resolve
        ``./eeglab_data.fdt`` relative to the working directory, exactly as
        the Fortran binary itself resolves it -- so this chdirs into
        sample_data/ rather than passing an absolute path."""
        monkeypatch.chdir(SAMPLE_DIR)
        model = AMICA(
            params_file="input.param",
            max_iter=3,
            seed=0,
            use_tqdm=False,
            outdir=str(tmp_path / "amicaout"),
        )
        model.fit()
        assert model.converged is True
        assert len(model.ll) == 3
        assert np.isfinite(model.ll[-1])


class TestNumpyKwargsOverrideFile:
    """Explicit kwargs win over params-file defaults (mirrors the PyTorch
    wrapper's ``TestFitAppliesFileDefaults.test_explicit_kwargs_win_over_
    file_defaults`` in ``test_fortran_params.py``): ``params.update(kwargs)``
    in ``__init__`` applies the file's dict first and then overwrites it with
    whatever the caller passed explicitly."""

    def test_explicit_kwargs_win_over_file_defaults(self, tmp_path):
        model = AMICA(
            params_file=str(PARAM_FILE),
            min_grad_norm=5e-5,
            max_decs=7,
            share_int=42,
            use_tqdm=False,
            outdir=str(tmp_path / "out"),
        )
        assert model.min_grad_norm == 5e-5  # kwarg wins over input.param's 1e-7
        assert model.max_decs == 7  # kwarg wins over input.param's 3
        assert model.share_int == 42  # kwarg wins over input.param's 100


class TestNumpyPdftypeGuard:
    """Gate 4: pdftype 1-4 raise NotImplementedError; pdftype 0 constructs."""

    @pytest.mark.parametrize("pdftype", [1, 2, 3, 4])
    def test_non_gg_pdftype_raises(self, pdftype, tmp_path):
        with pytest.raises(NotImplementedError, match="generalized-Gaussian"):
            AMICA(pdftype=pdftype, use_tqdm=False, outdir=str(tmp_path / "out"))

    def test_gg_pdftype_constructs(self, tmp_path):
        model = AMICA(pdftype=0, use_tqdm=False, outdir=str(tmp_path / "out"))
        assert model.pdftype == 0

    def test_default_pdftype_is_gg(self, tmp_path):
        """The bundled params.json default changed 1 -> 0 (issue #304): the
        constructor's own fallback was already 0, but the file default
        silently overrode it on every params_file=None construction."""
        model = AMICA(use_tqdm=False, outdir=str(tmp_path / "out"))
        assert model.pdftype == 0


class TestNumpyUnconsumedKeyWarning:
    """Gate 6: NumPy's unconsumed-key warning names the keys; writestep/
    do_history/histstep from Fortran text reach NumPy's attributes."""

    def test_input_param_unconsumed_keys_are_named(self, tmp_path, caplog):
        """input.param carries kurt_start/num_kurt/kurt_int (adaptive pdf,
        not on this backend) and num_samples (no NumPy-backend consumer)."""
        with caplog.at_level(logging.WARNING, logger="pamica.numpy_impl.core"):
            AMICA(
                params_file=str(PARAM_FILE),
                use_tqdm=False,
                outdir=str(tmp_path / "out"),
            )
        warnings = "\n".join(r.message for r in caplog.records)
        assert "kurt_start" in warnings
        assert "num_kurt" in warnings
        assert "kurt_int" in warnings
        assert "num_samples" in warnings

    def test_sample_params_json_has_no_unconsumed_keys(self, tmp_path, caplog):
        """sample_params.json carries nothing this backend cannot honor."""
        with caplog.at_level(logging.WARNING, logger="pamica.numpy_impl.core"):
            AMICA(
                params_file=str(JSON_FILE),
                use_tqdm=False,
                outdir=str(tmp_path / "out"),
            )
        assert caplog.records == []

    def test_bundled_default_has_no_unconsumed_keys(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="pamica.numpy_impl.core"):
            AMICA(use_tqdm=False, outdir=str(tmp_path / "out"))
        assert caplog.records == []

    def test_checkpoint_keys_reach_numpy_attributes(self, tmp_path):
        """writestep/do_history/histstep moved from unsupported to
        translated (issue #304): the legacy NumPy backend is the one backend
        that actually implements periodic on-disk checkpointing under these
        names, so a Fortran input.param's settings must reach it."""
        model = AMICA(
            params_file=str(PARAM_FILE), use_tqdm=False, outdir=str(tmp_path / "out")
        )
        assert model.writestep == 20  # input.param's own writestep
        assert model.do_history is False  # input.param's own do_history (0)
        assert model.histstep == 10  # input.param's own histstep
