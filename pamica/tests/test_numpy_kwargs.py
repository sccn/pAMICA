"""AMICA_NumPy(**kwargs) rejects unknown/unsupported keywords (issue #346).

Before this fix ``AMICA_NumPy(**kwargs)`` forwarded every keyword into a
params dict read with ``params.get(...)``, so a typo (``max_iters=50``) or an
option the legacy backend does not implement (``keep_best=True``) constructed
silently and had no effect -- unlike the PyTorch/MLX constructors (explicit
keyword parameters, ``TypeError`` on an unknown name) and the params-file
reader (``pamica.fortran_params.read_params_file``, issue #304), which was
already loud about a setting it could not apply.

``pamica.numpy_impl.core.__init__`` now validates ``**kwargs`` before any
other processing (``_reject_unknown_kwargs``). Every offending name is
categorized -- an alternate spelling of a setting this backend does support
(``n_models``/``n_mix``), ``n_channels`` (inferred from data on every
backend, not a constructor keyword on any of them), genuinely torch-only
(``keep_best``, ``device``, ``dtype``, the kurtosis-switch schedule), or
unrecognized with a ``difflib``-based "did you mean" suggestion -- and every
category present is named in ONE ``TypeError`` (PR #347 review item 3: a
first version of this raised on the first category it found, so
``AMICA_NumPy(n_models=2, totally_bogus=1)`` named only ``n_models`` and
dropped ``totally_bogus``).

The three settings this backend spells differently from ``AMICATorchNG``
(``min_nd``/``maxdecs``/``share_iter``, the canonical spelling a params file
already resolves either way) are also accepted as keyword arguments
directly, translated to this backend's own attribute name; passing both
spellings of the same setting at once is rejected as ambiguous, mirroring
``fortran_params._apply_json_aliases``'s identical guard for a params file.
``files``/``data_dim``/``field_dim`` (data-location metadata, normally read
from ``params_file``) are also accepted directly, with the same meaning (PR
#347 review item 1); setting both a params file and a keyword argument to
different values is rejected the same way.

No mocks: every construction below is a real ``AMICA_NumPy(...)`` call. Most
need no data (no ``.fit()``, exercising only ``__init__``'s validation); the
``TestDataLocationKwargs`` class fits the real bundled sample EEG
(``pamica/sample_data/eeglab_data.fdt``, 32 channels x 30504 frames), which
``test_numpy_params_file.py``/``test_numpy_cli.py`` also use.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from pamica import AMICA_NumPy
from pamica.numpy_impl import core as numpy_core
from pamica.numpy_impl.core import (
    _ACCEPTED_KWARGS,
    _ALTERNATE_SPELLING,
    _CANONICAL_ALIAS_KWARGS,
    _CANONICAL_TO_NUMPY_KEY,
    _CONSTRUCTOR_ONLY_KWARGS,
    _CONSUMED_KEYS,
    _N_CHANNELS_OPTION,
    _torch_only_options,
)
from pamica.torch_impl.core import AMICATorchNG

_CORE_SOURCE = Path(numpy_core.__file__).resolve()
_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
_DATA_FILE = _SAMPLE_DIR / "eeglab_data.fdt"
_NW = 32
_FIELD = 30504


def _construct(**kwargs: Any) -> AMICA_NumPy:
    """Build ``AMICA_NumPy`` from a keyword dict whose keys are only known at
    runtime (a parametrized name, or a loop variable). Typed ``**kwargs: Any``
    -- the same pattern ``test_schedule_gates.py``'s/``test_doscaling_rows.py``'s
    own ``_construct`` already uses -- so a static checker does not
    speculatively match a computed key against one of this constructor's own
    explicitly named parameters (``params_file``/``use_tqdm``/``verbose``)
    just because it cannot prove the key is not one of those names."""
    return AMICA_NumPy(**kwargs)


# --- Gate: the accepted-kwargs set cannot drift from the read sites --------


class TestKeySetMatchesReadSites:
    """``_CONSUMED_KEYS`` is exactly the set of keys ``__init__`` reads
    (``params.get(...)``/``raw_params.get(...)``), a source scan rather than a
    hand-maintained duplicate list, so the two cannot silently drift apart."""

    def _init_body(self) -> str:
        src = _CORE_SOURCE.read_text()
        start = src.index("    def __init__(")
        end = src.index("    def _setup_logging")
        return src[start:end]

    def test_every_consumed_key_is_read_and_vice_versa(self):
        """Depends entirely on every hyperparameter read in ``__init__``
        being spelled as a literal-string ``params.get("key")`` /
        ``raw_params.get("key")`` call -- the only shape the regexes below
        can extract a key name from. ``files``/``data_dim``/``field_dim``
        are the one exception, read off ``kwargs``/``raw_params`` directly
        (not through ``params.get``, see PR #347 review item 1) and
        exempted explicitly below, since they are already asserted present
        in ``_CONSUMED_KEYS`` by construction (the frozenset literal)."""
        body = self._init_body()
        # raw_params.get(...) first, so its matches can be excluded from the
        # plain params.get(...) scan below (the latter's regex would otherwise
        # also match inside "raw_params.get(...)").
        raw_keys = set(
            re.findall(r"raw_params\.get\(\s*['\"]([a-zA-Z0-9_]+)['\"]", body)
        )
        plain_body = body.replace("raw_params.get(", "")
        plain_keys = set(
            re.findall(r"params\.get\(\s*['\"]([a-zA-Z0-9_]+)['\"]", plain_body)
        )
        read_keys = raw_keys | plain_keys

        assert read_keys, "source scan found no params.get(...) calls at all"
        exempt = {"files", "data_dim", "field_dim"}
        assert read_keys - _CONSUMED_KEYS == set(), (
            f"read but not in _CONSUMED_KEYS: {sorted(read_keys - _CONSUMED_KEYS)}"
        )
        assert _CONSUMED_KEYS - read_keys - exempt == set(), (
            "in _CONSUMED_KEYS but never read via params.get(...)/"
            f"raw_params.get(...): {sorted(_CONSUMED_KEYS - read_keys - exempt)}"
        )

    def test_no_bracket_or_non_literal_reads_are_hiding_from_the_scan(self):
        """PR #347 review item 6: the scan above is blind to a hyperparameter
        read as ``params["key"]``/``raw_params["key"]`` (bracket access) or
        as ``params.get(some_variable)`` (a non-literal key) -- either shape
        would silently understate what ``_CONSUMED_KEYS`` actually covers,
        with the scan above passing regardless. This fails loudly, naming
        the offending snippet, if ``__init__`` ever reads a hyperparameter
        that way instead of a literal ``params.get("key")``."""
        body = self._init_body()

        bracket_reads = re.findall(r"\b(?:raw_)?params\[[^\]]*\]", body)
        assert not bracket_reads, (
            "__init__ reads params[...]/raw_params[...] by bracket access, "
            f"invisible to the source-scan test above: {bracket_reads}. "
            "Convert to params.get('key') or extend the scan."
        )

        # Every params.get(/raw_params.get( call site, regardless of its
        # argument's shape, so a non-literal (computed/variable) key is
        # caught even though the literal-key regexes above cannot extract
        # its name at all.
        for match in re.finditer(r"\b(raw_)?params\.get\(\s*(['\"]?)", body):
            quote = match.group(2)
            if quote not in ("'", '"'):
                snippet = body[match.start() : match.start() + 60]
                pytest.fail(
                    "__init__ has a params.get(...)/raw_params.get(...) call "
                    "whose key is not a literal string, invisible to the "
                    f"source-scan test above: {snippet!r}"
                )

    def test_accepted_kwargs_is_the_union_of_its_declared_sources(self):
        """``_ACCEPTED_KWARGS`` cannot drift from its three declared sources
        (issue #346): params-file keys, the two canonical aliases, and this
        constructor's own named parameters."""
        assert _ACCEPTED_KWARGS == (
            _CONSUMED_KEYS | _CANONICAL_ALIAS_KWARGS | _CONSTRUCTOR_ONLY_KWARGS
        )
        assert _CANONICAL_ALIAS_KWARGS == {"min_nd", "maxdecs", "share_iter"}
        assert _CONSTRUCTOR_ONLY_KWARGS == {"params_file", "use_tqdm", "verbose"}

    def test_torch_only_options_derived_from_torch_signature(self):
        """``_torch_only_options()`` is mechanically ``AMICATorchNG``'s own
        signature minus what this backend accepts (directly, or under an
        alternate spelling/``n_channels``'s own message), so it stays in
        sync as ``AMICATorchNG`` gains options, rather than a hand-picked
        list that can go stale."""
        expected = (
            frozenset(inspect.signature(AMICATorchNG).parameters)
            - _ACCEPTED_KWARGS
            - frozenset(_ALTERNATE_SPELLING)
            - {_N_CHANNELS_OPTION}
        )
        assert _torch_only_options() == expected
        # Sanity: the options issue #346 names explicitly are really in it.
        for name in ("keep_best", "device", "dtype"):
            assert name in _torch_only_options()
        # And nothing this backend actually does accept, under any name,
        # leaked in.
        assert _torch_only_options().isdisjoint(_ACCEPTED_KWARGS)
        assert _torch_only_options().isdisjoint(_ALTERNATE_SPELLING)
        assert _N_CHANNELS_OPTION not in _torch_only_options()

    def test_numpy_impl_core_has_no_module_level_torch_import(self):
        """PR #347 review item 5: ``_torch_only_options()`` imports
        ``AMICATorchNG`` lazily, inside its own function body, called only
        on the error path -- so ``numpy_impl/core.py``'s own source carries
        no module-level (top-of-file) dependency on ``torch_impl``.

        Not a runtime "torch never gets imported" check: ``import pamica``
        already imports ``torch_impl`` unconditionally, via ``amica.py``
        (the default-backend wrapper), regardless of anything in
        ``numpy_impl`` -- so that claim would be false to test. What issue
        #346/PR #347 actually changed is this module's OWN import graph:
        before, ``from ..torch_impl.core import AMICATorchNG`` sat at the
        top of this file; now the only mention of ``torch_impl`` anywhere
        in the module is the deferred import inside
        ``_torch_only_options``'s body, which this scans for directly."""
        src = _CORE_SOURCE.read_text()
        fn_start = src.index("def _torch_only_options")
        # Everything before _torch_only_options's own definition: the
        # file's real header import block plus every module-level constant
        # declared above it -- no "from ..torch_impl..." belongs there.
        header = src[:fn_start]
        top_level_torch_imports = [
            line
            for line in header.splitlines()
            if "torch_impl" in line and not line.strip().startswith("#")
        ]
        assert not top_level_torch_imports, (
            "numpy_impl/core.py imports torch_impl before _torch_only_options "
            f"is even defined (i.e. at module level): {top_level_torch_imports}"
        )
        # The one place torch_impl is actually imported: inside
        # _torch_only_options's own body, not at module level.
        fn_end = src.index("\ndef _reject_unknown_kwargs")
        assert "from ..torch_impl.core import AMICATorchNG" in src[fn_start:fn_end]
        # And nothing else in the whole file (a second, still-top-level
        # occurrence elsewhere) imports it either.
        rest = src[fn_end:]
        assert "import" not in "\n".join(
            line for line in rest.splitlines() if "torch_impl" in line
        )

    # Reason NumPy has no equivalent, one entry per name _torch_only_options()
    # currently derives (PR #347 review item 2): a hardcoded, reasoned
    # mapping checked against the mechanical derivation, so a name added to
    # AMICATorchNG that this backend still lacks is caught with an
    # explanation to update here, rather than silently absorbed into
    # "torch-only" by the set difference alone.
    _TORCH_ONLY_REASONS = {
        "device": "this backend always runs on CPU; there is no device to select",
        "dtype": "this backend always computes in float64; there is no dtype to choose",
        "keep_best": (
            "the best-iterate restore (issue #51) is not ported to this "
            "backend's fit loop"
        ),
        "kurt_start": (
            "the kurtosis-switch schedule (extended-Infomax adaptive pdf, "
            "pdftype=1) has nothing to switch: this backend implements only "
            "the generalized-Gaussian density, pdftype=0"
        ),
        "num_kurt": "the same kurt_start schedule, and reason, as above",
        "kurt_int": "the same kurt_start schedule, and reason, as above",
    }

    def test_every_torch_only_option_has_a_reason_and_no_numpy_equivalent(self):
        assert set(self._TORCH_ONLY_REASONS) == _torch_only_options()
        for name in self._TORCH_ONLY_REASONS:
            assert name not in _CONSUMED_KEYS, (
                f"{name!r} has a NumPy-side consumer (params.get); it is not "
                "torch-only, and belongs off this list"
            )
            assert name not in _ALTERNATE_SPELLING
            assert name != _N_CHANNELS_OPTION


# --- Gate: a typo raises TypeError with a suggestion ------------------------


class TestTypoRaises:
    def test_typo_raises_type_error_with_suggestion(self):
        with pytest.raises(TypeError) as exc:
            AMICA_NumPy(max_iters=50, use_tqdm=False)
        message = str(exc.value)
        assert "AMICA_NumPy got invalid keyword argument(s)" in message
        assert "unexpected keyword argument(s)" in message
        assert "max_iters" in message
        assert "max_iter" in message  # the suggested close match

    @pytest.mark.parametrize(
        ("typo", "correct"),
        [
            ("verbos", "verbose"),
            ("param_file", "params_file"),
            ("use_tqdmm", "use_tqdm"),
        ],
    )
    def test_typo_of_a_named_constructor_parameter_is_suggested(self, typo, correct):
        """``verbose``/``use_tqdm``/``params_file`` never legitimately reach
        **kwargs (Python binds a correctly spelled one directly), but a typo
        of one of them does, and should be named rather than silently
        vanishing into the ignored params dict (PR #347 review item 8 adds
        params_file/use_tqdm alongside the original verbose coverage)."""
        with pytest.raises(TypeError, match=correct):
            _construct(**{typo: True})

    def test_unknown_keyword_with_no_close_match_still_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            AMICA_NumPy(totally_unrelated_nonsense_xyz=1, use_tqdm=False)

    def test_multiple_unknown_keywords_all_named(self):
        with pytest.raises(TypeError) as exc:
            AMICA_NumPy(max_iters=50, lrates=0.1, use_tqdm=False)
        message = str(exc.value)
        assert "max_iters" in message
        assert "lrates" in message


# --- Gate: an unsupported cross-backend option raises the specific message --


class TestUnsupportedCrossBackendOption:
    @pytest.mark.parametrize("name", sorted(_torch_only_options()))
    def test_torch_only_option_raises_specific_message(self, name):
        with pytest.raises(TypeError) as exc:
            _construct(use_tqdm=False, **{name: object()})
        message = str(exc.value)
        assert "implemented on the PyTorch backend (AMICATorchNG)" in message
        assert "but not the legacy NumPy backend" in message
        assert name in message
        assert "AMICA(backend='torch')" in message

    def test_keep_best_specifically(self):
        """The option issue #346 names by example."""
        with pytest.raises(TypeError, match=r"keep_best.*AMICA\(backend='torch'\)"):
            AMICA_NumPy(keep_best=True, use_tqdm=False)

    def test_device_and_dtype_specifically(self):
        with pytest.raises(TypeError, match="device"):
            AMICA_NumPy(device="cpu", use_tqdm=False)
        with pytest.raises(TypeError, match="dtype"):
            AMICA_NumPy(dtype="float64", use_tqdm=False)


# --- Gate: alternate spellings and n_channels get their own messages -------
# (PR #347 review item 2: n_models/n_mix used to get the generic torch-only
# "use AMICA(backend='torch')" message, which is actively wrong -- this
# backend *does* support the setting, under its own name.)


class TestAlternateSpellingAndNChannels:
    @pytest.mark.parametrize(
        ("torch_name", "numpy_name"),
        [("n_models", "num_models"), ("n_mix", "num_mix")],
    )
    def test_alternate_spelling_names_the_correct_key(self, torch_name, numpy_name):
        with pytest.raises(TypeError) as exc:
            _construct(use_tqdm=False, **{torch_name: 2})
        message = str(exc.value)
        assert torch_name in message
        assert numpy_name in message
        # Not the generic torch-only message: this setting IS supported.
        assert "AMICA(backend='torch')" not in message

    def test_n_models_and_n_mix_are_not_torch_only(self):
        assert "n_models" not in _torch_only_options()
        assert "n_mix" not in _torch_only_options()
        assert _ALTERNATE_SPELLING == {"n_models": "num_models", "n_mix": "num_mix"}

    def test_n_channels_gets_its_own_message(self):
        with pytest.raises(TypeError) as exc:
            AMICA_NumPy(n_channels=32, use_tqdm=False)
        message = str(exc.value)
        assert "n_channels" in message
        assert "inferred from the data" in message
        # Not the torch-only message either: pointing to
        # AMICA(backend='torch') would be wrong here (that wrapper's own
        # fit(**kwargs) excludes n_channels too).
        assert "implemented on the PyTorch backend" not in message

    def test_n_channels_is_not_torch_only_or_alternate_spelling(self):
        assert _N_CHANNELS_OPTION == "n_channels"
        assert "n_channels" not in _torch_only_options()
        assert "n_channels" not in _ALTERNATE_SPELLING


# --- Gate: every offending category in one call is named in ONE error ------
# (PR #347 review item 3.)


class TestAllCategoriesNamedTogether:
    @pytest.mark.parametrize(
        ("kwargs", "must_contain"),
        [
            (
                {"n_models": 2, "totally_bogus": 1},
                ["n_models", "num_models", "totally_bogus"],
            ),
            (
                {"n_models": 2, "keep_best": True},
                ["n_models", "num_models", "keep_best"],
            ),
            (
                {"n_models": 2, "n_channels": 8},
                ["n_models", "num_models", "n_channels", "inferred from the data"],
            ),
            (
                {"totally_bogus": 1, "keep_best": True},
                ["totally_bogus", "keep_best"],
            ),
            (
                {"totally_bogus": 1, "n_channels": 8},
                ["totally_bogus", "n_channels", "inferred from the data"],
            ),
            (
                {"keep_best": True, "n_channels": 8},
                ["keep_best", "n_channels", "inferred from the data"],
            ),
            (
                {
                    "n_models": 2,
                    "n_channels": 8,
                    "keep_best": True,
                    "totally_bogus": 1,
                },
                ["n_models", "num_models", "n_channels", "keep_best", "totally_bogus"],
            ),
        ],
    )
    def test_every_offending_key_is_named(self, kwargs, must_contain):
        with pytest.raises(TypeError) as exc:
            _construct(use_tqdm=False, **kwargs)
        message = str(exc.value)
        for needle in must_contain:
            assert needle in message, f"{needle!r} missing from: {message}"

    def test_single_category_messages_are_unchanged_by_the_refactor(self):
        """The combined-message machinery must not change what a single
        category alone reports (regression guard for the item 3 fix)."""
        with pytest.raises(TypeError) as exc:
            AMICA_NumPy(keep_best=True, use_tqdm=False)
        assert "n_models" not in str(exc.value)
        assert "n_channels" not in str(exc.value)


# --- Gate: canonical-spelling kwargs are accepted, both-spellings rejected --


class TestCanonicalAliasKwargs:
    @pytest.mark.parametrize(
        ("canonical", "numpy_key", "value"),
        [
            ("min_nd", "min_grad_norm", 5e-6),
            ("maxdecs", "max_decs", 7),
            ("share_iter", "share_int", 42),
        ],
    )
    def test_canonical_spelling_takes_effect(self, canonical, numpy_key, value):
        model = _construct(use_tqdm=False, **{canonical: value})
        assert getattr(model, numpy_key) == value

    @pytest.mark.parametrize(
        ("canonical", "numpy_key"),
        [
            ("min_nd", "min_grad_norm"),
            ("maxdecs", "max_decs"),
            ("share_iter", "share_int"),
        ],
    )
    def test_both_spellings_at_once_raises(self, canonical, numpy_key):
        with pytest.raises(
            TypeError, match=f"{canonical}.*{numpy_key}|{numpy_key}.*{canonical}"
        ):
            _construct(use_tqdm=False, **{canonical: 1, numpy_key: 2})

    def test_canonical_to_numpy_key_matches_alias_set(self):
        assert set(_CANONICAL_TO_NUMPY_KEY) == {"min_nd", "maxdecs", "share_iter"}
        assert set(_CANONICAL_TO_NUMPY_KEY.values()) == {
            "min_grad_norm",
            "max_decs",
            "share_int",
        }


# --- Gate: files/data_dim/field_dim work as keywords, with real data -------
# (PR #347 review item 1.)


@pytest.mark.skipif(
    not _DATA_FILE.exists(), reason="bundled sample_data/eeglab_data.fdt missing"
)
class TestDataLocationKwargs:
    """Before this fix, ``files``/``data_dim``/``field_dim`` passed the
    unknown-kwarg check (already in ``_CONSUMED_KEYS``) but were silently
    inert -- only a params file's own setting reached
    ``_config_files``/``_config_data_dim``/``_config_field_dim`` -- so
    ``fit()`` with no data still said "no 'files' configured" even though
    the caller had just set one via a keyword argument. Real bundled sample
    EEG only (32 channels x 30504 frames), no synthetic data."""

    def _minimal_files_params_file(self, tmp_path: Path) -> Path:
        payload = {"files": [str(_DATA_FILE)], "data_dim": _NW, "field_dim": [_FIELD]}
        path = tmp_path / "files_only.json"
        path.write_text(json.dumps(payload))
        return path

    def test_files_kwarg_loads_real_data_and_fits(self, tmp_path):
        model = AMICA_NumPy(
            files=[str(_DATA_FILE)],
            data_dim=_NW,
            field_dim=[_FIELD],
            max_iter=3,
            seed=0,
            use_tqdm=False,
            outdir=str(tmp_path / "kw_out"),
        )
        model.fit()  # no data argument: must load via the kwargs above
        assert model.converged is True
        assert len(model.ll) == 3
        assert np.isfinite(model.ll).all()

    def test_files_kwarg_matches_params_file_bit_for_bit(self, tmp_path):
        """Same real data, same hyperparameters, same seed, loaded through
        two different surfaces for files/data_dim/field_dim -- everything
        else identical, so the two trajectories must be bit-for-bit equal."""
        params_file = self._minimal_files_params_file(tmp_path)
        from_file = AMICA_NumPy(
            params_file=str(params_file),
            max_iter=3,
            seed=0,
            use_tqdm=False,
            outdir=str(tmp_path / "file_out"),
        )
        from_file.fit()

        from_kwargs = AMICA_NumPy(
            files=[str(_DATA_FILE)],
            data_dim=_NW,
            field_dim=[_FIELD],
            max_iter=3,
            seed=0,
            use_tqdm=False,
            outdir=str(tmp_path / "kw_out"),
        )
        from_kwargs.fit()

        assert from_file.converged is True and from_kwargs.converged is True
        np.testing.assert_array_equal(from_file.ll, from_kwargs.ll)
        assert from_file.A is not None and from_kwargs.A is not None
        np.testing.assert_array_equal(from_file.A, from_kwargs.A)
        np.testing.assert_array_equal(from_file.mu, from_kwargs.mu)
        np.testing.assert_array_equal(from_file.sphere, from_kwargs.sphere)

    @pytest.mark.parametrize(
        ("name", "kwarg_value"),
        [("files", ["different.fdt"]), ("data_dim", _NW + 1), ("field_dim", [1])],
    )
    def test_conflicting_value_with_params_file_raises(
        self, tmp_path, name, kwarg_value
    ):
        params_file = self._minimal_files_params_file(tmp_path)
        with pytest.raises(TypeError, match=name):
            AMICA_NumPy(
                params_file=str(params_file),
                use_tqdm=False,
                outdir=str(tmp_path / "out"),
                **{name: kwarg_value},
            )

    def test_same_value_from_both_sources_is_not_a_conflict(self, tmp_path):
        params_file = self._minimal_files_params_file(tmp_path)
        model = AMICA_NumPy(
            params_file=str(params_file),
            data_dim=_NW,  # same value the file already sets
            use_tqdm=False,
            outdir=str(tmp_path / "out"),
        )
        assert model._config_data_dim == _NW
        assert model._config_files == [str(_DATA_FILE)]
        assert model._config_field_dim == [_FIELD]

    def test_no_files_error_message_mentions_keyword_argument(self, tmp_path):
        model = AMICA_NumPy(use_tqdm=False, outdir=str(tmp_path / "out"))
        with pytest.raises(ValueError, match="as a keyword argument"):
            model.fit()

    def test_no_data_dim_error_message_mentions_keyword_argument(self, tmp_path):
        model = AMICA_NumPy(
            files=[str(_DATA_FILE)], use_tqdm=False, outdir=str(tmp_path / "out")
        )
        with pytest.raises(ValueError, match="as a keyword argument"):
            model.fit()


# --- Gate: every documented NumPy option constructs -------------------------

# One representative, valid value per accepted keyword-only option (issue
# #346): every key in _CONSUMED_KEYS/_CANONICAL_ALIAS_KWARGS must appear here
# exactly once, enforced by test_values_cover_every_consumed_key below.
# pdftype must be exactly 0 (this backend implements only the
# generalized-Gaussian density); outdir/files/data_dim/field_dim are handled
# specially (outdir below, files/data_dim/field_dim in
# TestDataLocationKwargs above -- real data, not a placeholder value).
_REPRESENTATIVE_VALUES: Dict[str, Any] = {
    "num_models": 2,
    "num_mix": 2,
    "max_iter": 5,
    "do_newton": True,
    "newt_start": 3,
    "newt_ramp": 5,
    "newtrate": 0.3,
    "do_reject": True,
    "rejsig": 2.5,
    "rejstart": 3,
    "rejint": 2,
    "maxrej": 2,
    "num_comps": 4,
    "lrate": 0.05,
    "minlrate": 1e-10,
    "lratefact": 0.5,
    "rho0": 1.3,
    "minrho": 1.0,
    "maxrho": 2.5,
    "rholrate": 0.03,
    "rholratefact": 0.2,
    "invsigmax": 500.0,
    "invsigmin": 1e-3,
    "do_history": True,
    "histstep": 5,
    "do_opt_block": True,
    "block_size": 4096,
    "blk_min": 2048,
    "blk_max": 16384,
    "blk_step": 2048,
    "share_comps": True,
    "comp_thresh": 0.95,
    "share_start": 50,
    "share_int": 50,
    "doscaling": True,
    "scalestep": 2,
    "do_sphere": True,
    "do_mean": True,
    "do_approx_sphere": True,
    "pcakeep": 5,
    "pcadb": 1e-3,
    "mineig": 1e-8,
    "mineig_rel": 1e-10,
    "writestep": 50,
    "max_decs": 4,
    "maxincs": 3,
    "restartiter": 5,
    "maxrestarts": 2,
    "min_dll": 1e-8,
    "min_grad_norm": 1e-6,
    "use_min_dll": True,
    "use_grad_norm": True,
    "pdftype": 0,
    "seed": 123,
    "n_restarts": 1,
    "restart_seeds": [7],
    # Canonical-spelling aliases (issue #346): also documented, also
    # constructs, checked against their own (translated) attribute name.
    "min_nd": 1e-6,
    "maxdecs": 4,
    "share_iter": 50,
}

# outdir is exercised through the tmp_path-scoped test below instead of a
# hardcoded path (constructing with it creates the directory and an out.txt).
# files/data_dim/field_dim are exercised with real data in
# TestDataLocationKwargs above, not with a placeholder value here.
_SPECIAL_CASED_KWARGS = {"files", "data_dim", "field_dim", "outdir"}

# The attribute this backend stores a canonical-spelling kwarg under.
_ATTR_FOR_KEY = {**_CANONICAL_TO_NUMPY_KEY}


def _attr_name(key: str) -> str:
    return _ATTR_FOR_KEY.get(key, key)


class TestEveryDocumentedOptionConstructs:
    def test_values_cover_every_consumed_key(self):
        """The representative-value table itself does not drift from the
        accepted-kwargs set: every non-special-cased key has an entry, and
        every entry names a real accepted key."""
        documented = (_CONSUMED_KEYS | _CANONICAL_ALIAS_KWARGS) - _SPECIAL_CASED_KWARGS
        assert set(_REPRESENTATIVE_VALUES) == documented

    @pytest.mark.parametrize("key", sorted(_REPRESENTATIVE_VALUES))
    def test_option_constructs_and_takes_effect(self, key, tmp_path):
        value = _REPRESENTATIVE_VALUES[key]
        model = _construct(
            use_tqdm=False, outdir=str(tmp_path / f"out_{key}"), **{key: value}
        )
        assert getattr(model, _attr_name(key)) == value

    def test_outdir_constructs_and_takes_effect(self, tmp_path):
        target = tmp_path / "explicit_outdir"
        model = AMICA_NumPy(use_tqdm=False, outdir=str(target))
        assert model.outdir == target
        assert target.exists()

    def test_all_options_together_construct(self, tmp_path):
        """Every non-special-cased option at once, not just one at a time --
        the combination the one-at-a-time loop above cannot catch (e.g. two
        options whose validations only interact when both are set)."""
        kwargs = dict(_REPRESENTATIVE_VALUES)
        kwargs.pop("min_nd")  # would conflict with min_grad_norm, also present
        kwargs.pop("maxdecs")  # would conflict with max_decs, also present
        kwargs.pop("share_iter")  # would conflict with share_int, also present
        model = AMICA_NumPy(use_tqdm=False, outdir=str(tmp_path / "out_all"), **kwargs)
        for key, value in kwargs.items():
            assert getattr(model, _attr_name(key)) == value
