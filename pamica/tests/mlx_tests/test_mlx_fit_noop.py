"""Fit-path no-op check for epic #278 Phase 3/#289.

A default fit (``do_reject=False``, ``mir_step=0`` -- both the phase's new
knobs left at their off/inert defaults) must be bit-identical to what
``AMICAMLXNG`` produced at the epic tip BEFORE this phase (commit
``2e04006``, ``feature/issue-278-epic-mlx-parity``): the stash write, the
do_reject bookkeeping and the mir_step gate all have to be provably inert
when unused, not just "probably fine".

This is verified directly, not by assumption: the pre-phase-3
``pamica/mlx_impl/core.py`` is read from that git revision with ``git show``
and executed as a SIBLING module of the live ``pamica.mlx_impl`` package (so
its ``from .. import blocktune`` etc. relative imports resolve against the
real, unmodified helper modules), giving two independent ``AMICAMLXNG``
classes in the SAME process. Both then fit the same real data from the same
seed, and every fitted array is compared bit for bit -- a same-process
comparison, per the Phase 1/2 lesson against recording cross-machine
literals (``test_mlx_transform.py``'s ``_NOOP_PIN_*`` module comment): MLX
float32 is bit-reproducible run-to-run on ONE machine, so this needs no
recorded constant at all, just the two classes agreeing with each other,
here, now.

Both fits run with ``doscaling=False``. Issue #333 (epic #324 Phase 7)
deliberately changed the rescale from stored columns to component rows, so a
``doscaling=True`` fit of the historical class no longer matches by design;
with the rescale off, every other part of the fit path is still compared bit
for bit against the pre-Phase-3 tip. The rescale itself is pinned by
``pamica/tests/test_doscaling_rows.py``.

Issue #334 (epic #324 Phase 8) stores ``A`` with one component per row, so the
historical ``A`` (components as stored columns) is mapped onto rows with the
same lossless conversion a pre-#334 save goes through
(:func:`pamica.component_layout.rows_from_legacy_columns`) before the bit-for-bit
comparison; every per-model block, and so every other array, is unchanged.
"""

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.component_layout import rows_from_legacy_columns  # noqa: E402
from pamica.mlx_impl.core import AMICAMLXNG as CurrentAMICAMLXNG  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024

# The epic tip this phase branched from (see the phase plan): the last
# commit before issue #289's do_reject/LLt/export/MIR work landed.
_EPIC_TIP = "2e04006"
_HISTORICAL_MODULE_NAME = "pamica.mlx_impl._historical_phase3_noop_check"

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )[:, :4096]


@pytest.fixture(scope="module")
def historical_amicamlxng():
    """The pre-Phase-3 ``AMICAMLXNG`` class, loaded as a sibling module of
    ``pamica.mlx_impl`` so its relative imports (``from .. import
    blocktune`` etc.) resolve against the live, unmodified helper modules
    -- exactly as they would have at that commit, since none of
    blocktune/restarts/rank/numpy_impl.utils changed in this phase.

    Registered under a private name in ``sys.modules`` for the duration of
    this test module and removed afterward, so it cannot leak into any
    other test's import cache.

    In a shallow (depth-1) clone ``_EPIC_TIP`` is not a reachable object
    and a bare ``git show`` fails with "bad object" -- reproduced
    empirically on the macOS job before the CI jobs checked out full
    history (``fetch-depth: 0``). ``git fetch --depth 1`` that one commit
    first (deepening the clone by exactly the object needed, tolerating
    failure -- e.g. no network, or a remote that has since been pruned) and
    only then read it; if the object is still unreachable, fail under CI
    (the ``CI`` environment variable) and otherwise skip loudly naming the
    shallow-clone cause rather than erroring the whole module.
    """
    repo_root = Path(__file__).resolve().parents[3]
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{_EPIC_TIP}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
    )
    if present.returncode != 0:
        # Only a clone lacking the object fetches it: `git fetch --depth 1`
        # records the commit as a shallow boundary, which would hide the
        # history behind it in a complete clone.
        subprocess.run(
            ["git", "fetch", "origin", _EPIC_TIP, "--depth", "1"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )  # best-effort: a shallow CI checkout may not have `origin`; either
        # way the git show below is the real check, so a fetch failure here
        # is not itself fatal.
    result = subprocess.run(
        ["git", "show", f"{_EPIC_TIP}:pamica/mlx_impl/core.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        reason = (
            f"git object {_EPIC_TIP} is not reachable in this checkout "
            f"(likely a shallow/depth-1 clone that the fetch above could "
            f"not deepen -- e.g. no network or the remote history was "
            f"pruned); git show stderr: {result.stderr.strip()!r}"
        )
        # As in test_doscaling_rows.py: the CI jobs check out full history
        # (fetch-depth: 0), so there an unreachable pin fails instead of
        # silently dropping this regression guard.
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)
    module = types.ModuleType(_HISTORICAL_MODULE_NAME)
    module.__package__ = "pamica.mlx_impl"
    module.__name__ = _HISTORICAL_MODULE_NAME
    module.__file__ = f"<git {_EPIC_TIP} pamica/mlx_impl/core.py>"
    sys.modules[_HISTORICAL_MODULE_NAME] = module
    try:
        code = compile(result.stdout, module.__file__, "exec")
        exec(code, module.__dict__)
        yield module.AMICAMLXNG
    finally:
        del sys.modules[_HISTORICAL_MODULE_NAME]


_PARAM_NAMES = (
    "A",
    "W",
    "c",
    "mu",
    "alpha",
    "beta",
    "rho",
    "gm",
    "comp_list",
    "mean",
    "sphere",
    "pdtype",
)  # every array in AMICAMLXNG._PARAM_ARRAYS (PR #318 review item 10) -- the
# docstring above claims "every fitted array is compared bit for bit"; this
# list is what makes that literally true rather than a subset.


def _historical(model: Any, name: str) -> np.ndarray:
    """A fitted array of the historical model, ``A`` mapped onto the
    component rows the live backend stores (issue #334)."""
    value = np.array(getattr(model, name))
    if name == "A":
        return rows_from_legacy_columns(
            value, np.array(model.comp_list), owner="AMICAMLXNG"
        )
    return value


@pytest.mark.parametrize("n_models", [1, 2])
def test_unscaled_fit_is_bit_identical_to_the_pre_phase3_epic_tip(
    real_data, historical_amicamlxng, n_models
):
    kwargs: dict[str, Any] = dict(
        n_channels=NW,
        n_models=n_models,
        n_mix=NMIX,
        seed=42,
        block_size=BLOCK,
        doscaling=False,  # the rescale changed deliberately (#333; docstring)
    )

    old = historical_amicamlxng(**kwargs)
    old.fit(real_data, max_iter=10, verbose=False)

    new = CurrentAMICAMLXNG(**kwargs)
    # do_reject defaults False, mir_step defaults 0 -- exactly the phase's
    # two new "inert unless requested" knobs, left at their inert values.
    new.fit(real_data, max_iter=10, verbose=False)

    assert old.stop_reason == new.stop_reason
    assert old.final_ll_ == new.final_ll_
    assert old.ll_history == new.ll_history
    for name in _PARAM_NAMES:
        a = _historical(old, name)
        b = np.array(getattr(new, name))
        assert np.array_equal(a, b), f"{name}: diverged from the pre-phase-3 fit"


def test_unscaled_fit_with_keep_best_off_is_also_bit_identical(
    real_data, historical_amicamlxng
):
    """Same check with keep_best explicitly off, so the comparison does not
    depend on whichever safeguard branch a given seed happens to take."""
    kwargs: dict[str, Any] = dict(
        n_channels=NW,
        n_mix=NMIX,
        seed=7,
        block_size=BLOCK,
        keep_best=False,
        doscaling=False,  # the rescale changed deliberately (#333; docstring)
    )
    old = historical_amicamlxng(**kwargs)
    old.fit(real_data, max_iter=8, verbose=False)
    new = CurrentAMICAMLXNG(**kwargs)
    new.fit(real_data, max_iter=8, verbose=False)

    assert old.ll_history == new.ll_history
    for name in _PARAM_NAMES:
        np.testing.assert_array_equal(
            _historical(old, name), np.array(getattr(new, name)), err_msg=name
        )
