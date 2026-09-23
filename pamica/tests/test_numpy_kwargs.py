"""AMICA_NumPy(**kwargs) rejects unknown/unsupported keywords (issue #346).

Before this fix ``AMICA_NumPy(**kwargs)`` forwarded every keyword into a
params dict read with ``params.get(...)``, so a typo (``max_iters=50``) or an
option the legacy backend does not implement (``keep_best=True``) constructed
silently and had no effect -- unlike the PyTorch/MLX constructors (explicit
keyword parameters, ``TypeError`` on an unknown name) and the params-file
reader (``pamica.fortran_params.read_params_file``, issue #304), which was
already loud about a setting it could not apply.

``pamica.numpy_impl.core.__init__`` now validates ``**kwargs`` before any
other processing: an unrecognized name raises ``TypeError`` (with a
``difflib``-based "did you mean" suggestion when a close match exists among
the names this constructor does accept), and a name implemented on the
PyTorch backend (``AMICATorchNG``) but not this one gets a specific message
naming it and pointing to ``AMICA(backend='torch')``. The three settings this
backend spells differently from ``AMICATorchNG`` (``min_nd``/``maxdecs``/
``share_iter``, the canonical spelling a params file already resolves either
way) are now also accepted as keyword arguments directly, translated to this
backend's own attribute name exactly as the params-file route already
translates them; passing both spellings of the same setting at once is
rejected as ambiguous, mirroring ``fortran_params._apply_json_aliases``'s
identical guard for a params file.

No mocks: every construction below is a real ``AMICA_NumPy(...)`` call (no
``.fit()``, so no data is needed beyond a few tests that reuse the bundled
sample EEG already exercised by ``test_numpy_params_file.py``/
``test_numpy_cli.py``, which this file does not duplicate).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from pamica import AMICA_NumPy
from pamica.numpy_impl import core as numpy_core
from pamica.numpy_impl.core import (
    _ACCEPTED_KWARGS,
    _CANONICAL_ALIAS_KWARGS,
    _CANONICAL_TO_NUMPY_KEY,
    _CONSTRUCTOR_ONLY_KWARGS,
    _CONSUMED_KEYS,
    _TORCH_ONLY_OPTIONS,
)
from pamica.torch_impl.core import AMICATorchNG

_CORE_SOURCE = Path(numpy_core.__file__).resolve()


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
        assert read_keys - _CONSUMED_KEYS == set(), (
            f"read but not in _CONSUMED_KEYS: {sorted(read_keys - _CONSUMED_KEYS)}"
        )
        assert _CONSUMED_KEYS - read_keys == set(), (
            f"in _CONSUMED_KEYS but never read: {sorted(_CONSUMED_KEYS - read_keys)}"
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
        """``_TORCH_ONLY_OPTIONS`` is mechanically ``AMICATorchNG``'s own
        signature minus what this backend accepts, so it stays in sync as
        ``AMICATorchNG`` gains options, rather than a hand-picked list that
        can go stale."""
        expected = (
            frozenset(inspect.signature(AMICATorchNG).parameters) - _ACCEPTED_KWARGS
        )
        assert _TORCH_ONLY_OPTIONS == expected
        # Sanity: the options issue #346 names explicitly are really in it.
        for name in ("keep_best", "device", "dtype"):
            assert name in _TORCH_ONLY_OPTIONS
        # And nothing this backend actually does accept leaked in.
        assert _TORCH_ONLY_OPTIONS.isdisjoint(_ACCEPTED_KWARGS)


# --- Gate: a typo raises TypeError with a suggestion ------------------------


class TestTypoRaises:
    def test_typo_raises_type_error_with_suggestion(self):
        with pytest.raises(TypeError) as exc:
            AMICA_NumPy(max_iters=50, use_tqdm=False)
        message = str(exc.value)
        assert "AMICA_NumPy got unexpected keyword argument(s)" in message
        assert "max_iters" in message
        assert "max_iter" in message  # the suggested close match

    def test_typo_of_a_named_constructor_parameter_is_suggested(self):
        """``verbose``/``use_tqdm``/``params_file`` never legitimately reach
        **kwargs (Python binds a correctly spelled one directly), but a typo
        of one of them does, and should be named rather than silently
        vanishing into the ignored params dict."""
        with pytest.raises(TypeError, match="verbose"):
            AMICA_NumPy(verbos=True)

    def test_unknown_keyword_with_no_close_match_still_raises(self):
        with pytest.raises(TypeError, match="AMICA_NumPy got unexpected keyword"):
            AMICA_NumPy(totally_unrelated_nonsense_xyz=1, use_tqdm=False)

    def test_multiple_unknown_keywords_all_named(self):
        with pytest.raises(TypeError) as exc:
            AMICA_NumPy(max_iters=50, lrates=0.1, use_tqdm=False)
        message = str(exc.value)
        assert "max_iters" in message
        assert "lrates" in message


# --- Gate: an unsupported cross-backend option raises the specific message --


class TestUnsupportedCrossBackendOption:
    @pytest.mark.parametrize("name", sorted(_TORCH_ONLY_OPTIONS))
    def test_torch_only_option_raises_specific_message(self, name, tmp_path):
        with pytest.raises(TypeError) as exc:
            _construct(use_tqdm=False, **{name: object()})
        message = str(exc.value)
        assert "AMICA_NumPy does not support" in message
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


# --- Gate: every documented NumPy option constructs -------------------------

# One representative, valid value per accepted keyword-only option (issue
# #346): every key in _CONSUMED_KEYS/_CANONICAL_ALIAS_KWARGS must appear here
# exactly once, enforced by test_values_cover_every_consumed_key below.
# pdftype must be exactly 0 (this backend implements only the
# generalized-Gaussian density); outdir/files/data_dim/field_dim are handled
# specially in the test itself.
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

# Data-location metadata: read only from a parsed params file
# (params_file=...), never from a bare kwarg (see AMICA_NumPy.__init__'s
# _config_files/_config_data_dim/_config_field_dim, gated on params_file is
# not None), so a kwarg of these three names has no attribute to check --
# accepted (not a TypeError), just inert. Covered separately below rather
# than through the value dict.
_DATA_LOCATION_ONLY_KWARGS = {"files", "data_dim", "field_dim"}

# outdir is exercised through the tmp_path-scoped test below instead of a
# hardcoded path (constructing with it creates the directory and an out.txt).
_SPECIAL_CASED_KWARGS = _DATA_LOCATION_ONLY_KWARGS | {"outdir"}

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

    @pytest.mark.parametrize("key", sorted(_DATA_LOCATION_ONLY_KWARGS))
    def test_data_location_kwarg_is_accepted_but_inert(self, key, tmp_path):
        """Accepted (no TypeError): this is deliberate pre-existing behavior
        (data-location metadata is read from a parsed params file, never a
        bare kwarg), not something issue #346 changes -- only that an
        *unrecognized* name now raises. Passing one of these three names
        alone must not raise."""
        value = ["irrelevant"] if key == "files" else 1
        model = _construct(
            use_tqdm=False, outdir=str(tmp_path / f"out_{key}"), **{key: value}
        )
        assert model is not None

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
