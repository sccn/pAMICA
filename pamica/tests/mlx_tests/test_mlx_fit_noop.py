"""Fit-path no-op check for epic #278 Phase 3/#289.

A default fit (``do_reject=False``, ``mir_step=0`` -- both the phase's new
knobs left at their off/inert defaults) must be bit-identical to what
``AMICAMLXNG`` produced at the epic tip BEFORE this phase (commit
``2e04006``, ``feature/issue-278-epic-mlx-parity``): the stash write, the
do_reject bookkeeping and the mir_step gate all have to be provably inert
when unused, not just "probably fine".

This is verified directly, not by assumption: the whole pamica package at
that revision is read from git and imported under a private name through the
shared loader ``pamica/tests/pre_change.py`` (which never fetches, issue
#343), giving two independent ``AMICAMLXNG`` classes in the SAME process, each
with its own helper modules. Both then fit the same real data from the same
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

Issue #344 (epic #324 Phase 13) gave every backend the reference's
single-precision density constants, and these fits reach ``rho == 2``, where
the generalized Gaussian's normalizer is one of them, so the live class runs
with the constants it had before that change
(:func:`pamica.tests.pre_change.use_pre_344_constants`); the new values are
pinned against the reference by ``pamica/tests/test_reference_constants.py``.
"""

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.component_layout import rows_from_legacy_columns  # noqa: E402
from pamica.mlx_impl.core import AMICAMLXNG as CurrentAMICAMLXNG  # noqa: E402
from pamica.tests.pre_change import (  # noqa: E402
    load_pre_change_package,
    use_pre_344_constants,
)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
BLOCK = 1024

# The epic tip this phase branched from (see the phase plan): the last
# commit before issue #289's do_reject/LLt/export/MIR work landed.
_EPIC_TIP = "2e04006c85b96f3beb62dc12405479a05f2d5bc6"

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
def historical_amicamlxng(tmp_path_factory):
    """The pre-Phase-3 ``AMICAMLXNG`` class: the package at ``_EPIC_TIP``,
    imported beside the live one. A clone without that commit fails under
    ``CI`` and otherwise skips with the command that fetches it."""
    old = load_pre_change_package(
        _EPIC_TIP, "pamica_pre289", tmp_path_factory.mktemp("pre289")
    )
    return importlib.import_module(f"{old.__name__}.mlx_impl.core").AMICAMLXNG


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
    real_data, historical_amicamlxng, n_models, monkeypatch
):
    use_pre_344_constants(monkeypatch, "mlx")
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
    real_data, historical_amicamlxng, monkeypatch
):
    """Same check with keep_best explicitly off, so the comparison does not
    depend on whichever safeguard branch a given seed happens to take."""
    use_pre_344_constants(monkeypatch, "mlx")
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
