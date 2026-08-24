"""Shared best-of-N restart policy (issue #198).

Seed derivation, the selection rule and the state-copy bookkeeping are one
decision used by three backends, so the decision lives in ``pamica.restarts``
and is tested once here. Two anti-drift guards required by
``.rules/backend_parity.md`` live at the bottom:

* the **surface** guard -- the three backends must expose the same restart
  parameters and the same record attributes; and
* the **state-completeness** guard -- each backend's ``_RESTART_STATE_ATTRS``
  plus ``_RESTART_INVARIANT_ATTRS`` plus the shared record names must account
  for *every* attribute its fit path assigns, checked by parsing the source. A
  field added to any fit-path method later fails this test until it is
  classified as copied-per-restart or data-invariant, which is what stops it
  from being silently dropped from a restart snapshot.

The structural guard is a source-level check on purpose: no fit runs here, so it
costs nothing and cannot be fooled by a code path a short fit does not reach.
"""

import ast
import inspect
from pathlib import Path
from typing import Any, Set

import numpy as np
import pytest

from pamica import restarts
from pamica.numpy_impl.core import AMICA as AMICA_NumPy
from pamica.torch_impl.core import AMICATorchNG

# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def test_single_restart_passes_the_base_seed_through_untouched():
    """The parity-preserving default: one restart, the constructor's own seed,
    including ``None`` (which is what lets an unseeded fit stay unseeded)."""
    assert restarts.resolve_seeds(1, None, 42) == [42]
    assert restarts.resolve_seeds(1, None, None) == [None]


def test_seeds_derive_consecutively_from_the_base_seed():
    assert restarts.resolve_seeds(4, None, 7) == [7, 8, 9, 10]


def test_explicit_seeds_win_over_derivation():
    assert restarts.resolve_seeds(3, [5, 9, 1], 42) == [5, 9, 1]


def test_explicit_seeds_must_match_n_restarts():
    with pytest.raises(ValueError, match="exactly one seed per restart"):
        restarts.resolve_seeds(3, [5, 9], 42)


def test_n_restarts_must_be_at_least_one():
    with pytest.raises(ValueError, match="n_restarts must be >= 1"):
        restarts.resolve_seeds(0, None, 42)


def test_n_restarts_must_be_an_int():
    # Deliberately the wrong type: a caller reading n_restarts out of a config
    # file gets a clear TypeError instead of a float sneaking into range().
    with pytest.raises(TypeError, match="n_restarts must be an int"):
        restarts.resolve_seeds(2.5, None, 42)  # ty: ignore[invalid-argument-type]


def test_seeds_must_be_ints():
    with pytest.raises(TypeError, match="restart_seeds entries must be ints"):
        restarts.resolve_seeds(2, [1, "two"], 42)  # ty: ignore[invalid-argument-type]


def test_multi_restart_without_a_seed_is_refused():
    """No clock, no OS entropy: a winner that cannot be reproduced is not a
    result, so the configuration is rejected instead of silently guessing."""
    with pytest.raises(ValueError, match="seed=<int>"):
        restarts.resolve_seeds(3, None, None)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_highest_likelihood_wins():
    assert restarts.select_best([-3.5, -3.2, -3.9], [False] * 3) == 1


def test_ties_keep_the_earlier_restart():
    assert restarts.select_best([-3.2, -3.2], [False, False]) == 0


def test_degenerate_restarts_are_excluded_even_when_they_score_higher():
    """A degenerate fit's likelihood is not comparable (and is NaN on the array
    backends); it must never be selected, however it scores."""
    assert restarts.select_best([-1.0, -3.2], [True, False]) == 1
    assert restarts.select_best([float("nan"), -3.2], [True, False]) == 1


def test_non_finite_likelihood_is_excluded_even_if_not_flagged():
    assert restarts.select_best([float("nan"), -3.2], [False, False]) == 1
    assert restarts.select_best([float("-inf"), -3.2], [False, False]) == 1


def test_all_degenerate_selects_nothing():
    assert restarts.select_best([float("nan")] * 3, [True] * 3) is None


def test_records_must_stay_aligned():
    with pytest.raises(ValueError, match="aligned"):
        restarts.select_best([-3.2, -3.1], [False])


# ---------------------------------------------------------------------------
# State copying
# ---------------------------------------------------------------------------


def test_copies_are_independent_of_the_live_value():
    array = np.arange(4.0)
    copied = restarts.copy_state_value(array)
    array[0] = 99.0
    assert copied[0] == 0.0

    history = [1.0, 2.0]
    copied_history = restarts.copy_state_value(history)
    history.append(3.0)
    assert copied_history == [1.0, 2.0]

    rng = np.random.RandomState(0)
    copied_rng = restarts.copy_state_value(rng)
    rng.rand(10)
    assert np.array_equal(copied_rng.rand(5), np.random.RandomState(0).rand(5))


def test_unknown_types_raise_rather_than_alias():
    with pytest.raises(TypeError, match="copy_state_value cannot copy"):
        restarts.copy_state_value(object())


# ---------------------------------------------------------------------------
# Cross-backend guards
# ---------------------------------------------------------------------------

_BACKENDS = ["torch", "numpy"]
_MLX_CLASS: Any = None
try:  # MLX is Apple-Silicon only and optional; guard the import, not the tests.
    from pamica.mlx_impl.core import AMICAMLXNG

    _MLX_CLASS = AMICAMLXNG
    _BACKENDS.append("mlx")
except ImportError:  # pragma: no cover - exercised on non-Apple hosts
    pass


def _backend_class(name: str):
    return {
        "torch": AMICATorchNG,
        "numpy": AMICA_NumPy,
        "mlx": _MLX_CLASS,
    }[name]


def _ctor_kwargs(name: str) -> dict:
    """The minimum each backend needs to construct. ``device="cpu"`` because the
    float64 default cannot live on MPS, and this file never fits anything."""
    return {
        "torch": {"n_channels": 4, "device": "cpu"},
        "numpy": {"use_tqdm": False},
        "mlx": {"n_channels": 4},
    }[name]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_every_backend_exposes_the_same_restart_surface(backend: str):
    """The three backends must be configured the same way (``.rules/
    backend_parity.md``): ``n_restarts``/``restart_seeds`` alongside ``seed``,
    with the same parity-preserving default."""
    cls = _backend_class(backend)
    if backend != "numpy":
        # The NumPy backend takes its parameters through a params dict /
        # **kwargs rather than named arguments, so only the resolved attributes
        # below are checkable there.
        parameters = inspect.signature(cls).parameters
        assert "n_restarts" in parameters
        assert "restart_seeds" in parameters
        assert parameters["n_restarts"].default == restarts.DEFAULT_N_RESTARTS
        assert parameters["restart_seeds"].default is None
    model = cls(**_ctor_kwargs(backend))
    assert model.n_restarts == restarts.DEFAULT_N_RESTARTS
    assert model.restart_seeds is None
    assert model._restart_seeds == [None]
    for name in restarts.RECORD_ATTRS:
        assert getattr(model, name) == []


@pytest.mark.parametrize("backend", _BACKENDS)
def test_every_backend_validates_the_configuration_at_construction(backend: str):
    """A bad restart configuration must fail before any data is touched."""
    cls = _backend_class(backend)
    kwargs = _ctor_kwargs(backend)
    with pytest.raises(ValueError, match="n_restarts must be >= 1"):
        cls(n_restarts=0, seed=1, **kwargs)
    with pytest.raises(ValueError, match="exactly one seed per restart"):
        cls(n_restarts=3, restart_seeds=[1, 2], seed=1, **kwargs)
    with pytest.raises(ValueError, match="seed=<int>"):
        cls(n_restarts=2, **kwargs)


# Methods excluded from the fit-path scan below, with the reason each is not
# part of a fit: the constructor assigns the configuration (which a restart must
# NOT reset), the logging setup assigns file handles, and the PyTorch backend's
# _load_params is the deserialization path, not a fit.
_EXCLUDED_METHODS = {"__init__", "_setup_logging", "_load_params"}


def _fit_path_assignments(path: Path, class_name: str) -> Set[str]:
    """Every ``self.<attr> = ...`` (or ``+=``) target in the class's methods,
    excluding :data:`_EXCLUDED_METHODS`. ``setattr`` is deliberately invisible
    here: it is how the snapshot restore writes back, and counting it would make
    the guard circular."""
    tree = ast.parse(path.read_text())
    found: Set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name in _EXCLUDED_METHODS:
                continue
            for sub in ast.walk(method):
                if isinstance(sub, ast.Assign):
                    targets = sub.targets
                elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
                    targets = [sub.target]
                else:
                    continue
                for target in targets:
                    for name in ast.walk(target):
                        if (
                            isinstance(name, ast.Attribute)
                            and isinstance(name.value, ast.Name)
                            and name.value.id == "self"
                        ):
                            found.add(name.attr)
    return found


@pytest.mark.parametrize("backend", _BACKENDS)
def test_restart_state_lists_account_for_everything_a_fit_writes(backend: str):
    """State-completeness guard (issue #198 policy 5).

    Every attribute the fit path assigns must be classified as either copied
    per restart (``_RESTART_STATE_ATTRS``), constant across the restarts of one
    fit because it is a function of the data alone
    (``_RESTART_INVARIANT_ATTRS``), or a record of the search itself
    (``restarts.RECORD_ATTRS``). Exact set equality in both directions, so
    adding a field to a fit-path method -- or leaving a stale name in a list --
    fails here rather than producing a subtly incomplete winning state.
    """
    cls = _backend_class(backend)
    source = inspect.getsourcefile(cls)
    assert source is not None, f"no source file for {cls.__name__}"
    assigned = _fit_path_assignments(Path(source), cls.__name__)
    classified = (
        set(cls._RESTART_STATE_ATTRS)
        | set(cls._RESTART_INVARIANT_ATTRS)
        | set(restarts.RECORD_ATTRS)
    )
    unclassified = assigned - classified
    stale = classified - assigned - set(restarts.RECORD_ATTRS)
    assert not unclassified, (
        f"{cls.__name__}: the fit path writes {sorted(unclassified)}, which is "
        f"in neither _RESTART_STATE_ATTRS nor _RESTART_INVARIANT_ATTRS. Classify "
        f"it: would a restart snapshot need to copy it?"
    )
    assert not stale, (
        f"{cls.__name__}: {sorted(stale)} is listed as restart state but nothing "
        f"in the fit path assigns it any more."
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_state_and_invariant_lists_do_not_overlap(backend: str):
    cls = _backend_class(backend)
    overlap = set(cls._RESTART_STATE_ATTRS) & set(cls._RESTART_INVARIANT_ATTRS)
    assert not overlap, f"{cls.__name__}: {sorted(overlap)} is in both lists"


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_fitted_parameters_are_all_restart_state(backend: str):
    """The parameters a caller reads off a fitted model are the minimum a
    snapshot has to carry; this pins them independently of the source scan."""
    cls = _backend_class(backend)
    for name in ("A", "W", "c", "mu", "alpha", "beta", "rho", "gm", "comp_list"):
        assert name in cls._RESTART_STATE_ATTRS, f"{cls.__name__} misses {name}"


@pytest.mark.parametrize("backend", _BACKENDS)
def test_derived_seeds_are_identical_across_backends(backend: str):
    """Same base seed, same restart seeds -- so a cross-backend comparison of a
    best-of-N fit compares the same searches."""
    cls = _backend_class(backend)
    model = cls(seed=100, n_restarts=3, **_ctor_kwargs(backend))
    assert model._restart_seeds == [100, 101, 102]
