"""``state_dict``/``from_state_dict`` and ``.npz`` ``save``/``load`` on the MLX
backend -- issue #287, epic #278 Phase 1.

Apple-Silicon only, real sample EEG (no synthetic/mock), same module guards as
``test_mlx_backend.py``. Covers the round trip (every param bit-identical,
every config/extra field equal, ``transform`` bit-identical pre/post), the
component-sharing and adaptive-switcher state round trip, and the refusal
guards (unfitted, degenerate, wrong version, missing sections, shape drift).
"""

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42
BLOCK = 1024

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]


def _real_data(n_samples: int | None = None) -> np.ndarray:
    from pamica.torch_impl.utils import load_eeglab_data

    data = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return data if n_samples is None else data[:, :n_samples]


def _fitted_model(n_models: int = 1, max_iter: int = 5, **kwargs) -> AMICAMLXNG:
    m = AMICAMLXNG(
        n_channels=NW, n_models=n_models, n_mix=NMIX, seed=SEED, block_size=BLOCK,
        **kwargs,
    )  # fmt: skip
    m.fit(_real_data(4096), max_iter=max_iter, verbose=False)
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    return m


def _force_merged_column(model):
    """Fold model 1's first component into model 0's, as a fit-time merge
    does. Same construction as ``test_mlx_sharing.py::_force_merged_column``
    and ``test_mlx_newton_cross_backend.py::_force_merged_column``.
    Deterministic on purpose: keying the round-trip test on whether a merge
    happened to occur naturally would let it skip under the very bug it
    guards."""
    cl = np.array(model.comp_list)
    kept, dead = int(cl[0, 0]), int(cl[0, 1])
    a_np = np.array(model.A)
    a_np[:, dead] = a_np[:, kept]
    model.A = mx.array(a_np)
    cl[cl == dead] = kept
    model.comp_list = mx.array(cl)
    used = np.zeros(model.n_comps, dtype=bool)
    used[np.unique(cl)] = True
    model._comp_used_arr = mx.array(used)
    model._update_unmixing_matrices()
    mx.eval(model.A, model.W)
    return kept, dead


# --- round trip: state_dict / from_state_dict --------------------------------


def test_state_dict_round_trip_params_are_bit_identical():
    m = _fitted_model(n_models=2, max_iter=5)
    state = m.state_dict()
    m2 = AMICAMLXNG.from_state_dict(state)

    for name in AMICAMLXNG._PARAM_ARRAYS:
        a = np.array(getattr(m, name))
        b = np.array(getattr(m2, name))
        assert np.array_equal(a, b), f"{name} is not bit-identical after round trip"
        assert a.dtype == b.dtype, f"{name} dtype drifted: {a.dtype} vs {b.dtype}"


def test_state_dict_round_trip_config_and_extra_are_equal():
    m = _fitted_model(n_models=2, max_iter=5, do_newton=True, newt_start=0)
    state = m.state_dict()
    m2 = AMICAMLXNG.from_state_dict(state)
    state2 = m2.state_dict()

    assert state["config"] == state2["config"]
    assert state["extra"] == state2["extra"]


def test_state_dict_round_trip_transform_is_bit_identical():
    m = _fitted_model(n_models=2, max_iter=5)
    data = _real_data(64)
    S_before = [m.transform(data, model_idx=h) for h in range(2)]

    m2 = AMICAMLXNG.from_state_dict(m.state_dict())
    S_after = [m2.transform(data, model_idx=h) for h in range(2)]

    for h in range(2):
        assert np.array_equal(S_before[h], S_after[h]), (
            f"model {h}: transform output is not bit-identical after round trip"
        )


def test_state_dict_round_trip_accessors_are_bit_identical():
    """get_mixing_matrix/get_unmixing_matrix/get_rho depend only on directly
    persisted params (A/comp_list, W, rho), so they round-trip bit-exactly
    too -- unlike get_sensor_mixing_matrix, which depends on _sphere_np, a
    derived cache reconstructed from the persisted float32 sphere (see
    _load_params); that one is checked separately, with a tolerance, below.
    """
    m = _fitted_model(n_models=2, max_iter=5)
    m2 = AMICAMLXNG.from_state_dict(m.state_dict())
    for h in range(2):
        assert np.array_equal(m.get_mixing_matrix(h), m2.get_mixing_matrix(h))
        assert np.array_equal(m.get_unmixing_matrix(h), m2.get_unmixing_matrix(h))
        assert np.array_equal(m.get_rho(h), m2.get_rho(h))


def test_state_dict_round_trip_sensor_mixing_matrix_is_close():
    """get_sensor_mixing_matrix reads _pinv_sphere(), which is built from
    _sphere_np -- the true float64 sphere during a live fit, but the float32
    persisted sphere upcast to float64 after a reload (documented in
    _load_params). So this one is NOT bit-identical, only close; the gap is
    the float32 rounding of the sphere itself.
    """
    m = _fitted_model(n_models=1, max_iter=5)
    before = m.get_sensor_mixing_matrix()
    m2 = AMICAMLXNG.from_state_dict(m.state_dict())
    after = m2.get_sensor_mixing_matrix()
    assert np.allclose(before, after, rtol=1e-5, atol=1e-6)


# --- round trip: save / load (.npz) ------------------------------------------


def test_save_load_round_trip_bit_identical(tmp_path):
    m = _fitted_model(n_models=2, max_iter=5)
    data = _real_data(64)
    S_before = m.transform(data, model_idx=0)

    filepath = tmp_path / "model.npz"
    m.save(str(filepath))
    assert filepath.exists()

    m2 = AMICAMLXNG.load(str(filepath))
    for name in AMICAMLXNG._PARAM_ARRAYS:
        assert np.array_equal(np.array(getattr(m, name)), np.array(getattr(m2, name)))
    assert m.state_dict()["config"] == m2.state_dict()["config"]
    assert m.state_dict()["extra"] == m2.state_dict()["extra"]
    S_after = m2.transform(data, model_idx=0)
    assert np.array_equal(S_before, S_after)


def test_save_load_npz_layout(tmp_path):
    """The file is exactly one ``.npz`` with the documented top-level keys
    (format_version/config/extra as JSON-encoded scalars, plus the 12 native
    param arrays) -- not a pickle, not multiple files."""
    m = _fitted_model(n_models=1, max_iter=3)
    filepath = tmp_path / "model.npz"
    m.save(str(filepath))

    with np.load(str(filepath), allow_pickle=False) as data:
        files = set(data.files)
        assert files == {"format_version", "config", "extra"} | set(
            AMICAMLXNG._PARAM_ARRAYS
        )
        assert int(data["format_version"]) == 1
        config = json.loads(data["config"].item())
        assert config["n_channels"] == m.n_channels
        extra = json.loads(data["extra"].item())
        assert extra["stop_reason"] == m.stop_reason


# --- sharing + adaptive-switcher state round trip (plan item C5) ------------


def test_round_trip_preserves_forced_merge_state():
    """A forced-merge (``share_comps``-style) state round-trips
    ``comp_list``/``comp_used``/``shared_components()`` exactly."""
    m = _fitted_model(n_models=2, max_iter=3, share_comps=True)
    kept, dead = _force_merged_column(m)
    assert dead not in np.unique(np.array(m.comp_list))
    groups_before = m.shared_components()
    assert groups_before, "the forced merge did not produce a shared group"

    m2 = AMICAMLXNG.from_state_dict(m.state_dict())
    assert np.array_equal(np.array(m.comp_list), np.array(m2.comp_list))
    assert np.array_equal(np.array(m.comp_used), np.array(m2.comp_used))
    assert m2.shared_components() == groups_before


def test_round_trip_preserves_adaptive_switcher_state():
    """A ``pdftype=1`` (adaptive extended-Infomax switcher) fit round-trips
    ``pdtype``/``n_kurt_done`` exactly. seed=9 is one of a handful (swept 0-11
    on the bundled sample) where the switcher actually flips at least one
    source to code 4 -- most seeds stay uniformly at the code-1 init on this
    data (real-data-dependent behavior, no bit-exact oracle; see the module
    docstring in ``pamica/mlx_impl/core.py``) -- so this is a genuinely mixed
    array to round-trip, not a vacuously uniform one."""
    m = AMICAMLXNG(
        n_channels=NW, n_mix=1, pdftype=1, seed=9,
        kurt_start=3, num_kurt=5, kurt_int=1,
    )  # fmt: skip
    m.fit(_real_data(), max_iter=20, verbose=False)
    assert m.n_kurt_done == 5
    codes_before = m.get_pdftype()
    assert set(np.unique(codes_before).tolist()).issubset({1, 4})
    # Actually exercise the switch, not just the schedule counter, or the
    # dtype round-trip check below would be trivially true for a uniform
    # array.
    assert len(set(codes_before.tolist())) > 1, "the switcher never flipped a source"

    m2 = AMICAMLXNG.from_state_dict(m.state_dict())
    assert m2.n_kurt_done == 5
    assert np.array_equal(m.get_pdftype(), m2.get_pdftype())
    assert np.array_equal(np.array(m.pdtype), np.array(m2.pdtype))
    assert np.array(m2.pdtype).dtype == np.array(m.pdtype).dtype


# --- refusals (plan item C6) -------------------------------------------------


def test_state_dict_refuses_unfitted_model():
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX)
    with pytest.raises(RuntimeError, match="requires a fitted model"):
        m.state_dict()


def test_save_refuses_unfitted_model(tmp_path):
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX)
    with pytest.raises(RuntimeError, match="requires a fitted model"):
        m.save(str(tmp_path / "model.npz"))


def test_state_dict_refuses_degenerate_fit():
    data = _real_data(4096).copy()
    data[0, 0] = np.nan
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, do_sphere=False, do_mean=False)
    m.fit(data, max_iter=5, verbose=False)
    assert m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS
    with pytest.raises(RuntimeError, match="degenerate"):
        m.state_dict()


def test_save_refuses_degenerate_fit(tmp_path):
    data = _real_data(4096).copy()
    data[0, 0] = np.nan
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, do_sphere=False, do_mean=False)
    m.fit(data, max_iter=5, verbose=False)
    assert m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS
    with pytest.raises(RuntimeError, match="degenerate"):
        m.save(str(tmp_path / "model.npz"))


def test_from_state_dict_rejects_wrong_format_version():
    m = _fitted_model(n_models=1, max_iter=3)
    state = m.state_dict()
    state["format_version"] = 999
    with pytest.raises(ValueError, match="format_version"):
        AMICAMLXNG.from_state_dict(state)


@pytest.mark.parametrize("section", ["config", "params", "extra"])
def test_from_state_dict_rejects_missing_section(section):
    m = _fitted_model(n_models=1, max_iter=3)
    state = m.state_dict()
    del state[section]
    with pytest.raises(ValueError, match=section):
        AMICAMLXNG.from_state_dict(state)


def test_from_state_dict_rejects_missing_param():
    m = _fitted_model(n_models=1, max_iter=3)
    state = m.state_dict()
    del state["params"]["A"]
    with pytest.raises(ValueError, match="A"):
        AMICAMLXNG.from_state_dict(state)


def test_from_state_dict_rejects_shape_drift():
    """A restored ``A`` that does not match the config-derived dimensions
    raises rather than failing later with a confusing matmul error."""
    m = _fitted_model(n_models=1, max_iter=3)
    state = m.state_dict()
    state["params"]["A"] = state["params"]["A"][:, :-1]  # drop one column
    with pytest.raises(ValueError, match="shape"):
        AMICAMLXNG.from_state_dict(state)


def test_from_state_dict_rejects_comp_list_shape_drift():
    m = _fitted_model(n_models=2, max_iter=3)
    state = m.state_dict()
    state["params"]["comp_list"] = state["params"]["comp_list"][:, :1]
    with pytest.raises(ValueError, match="shape"):
        AMICAMLXNG.from_state_dict(state)


def test_load_rejects_wrong_format_version(tmp_path):
    m = _fitted_model(n_models=1, max_iter=3)
    filepath = tmp_path / "model.npz"
    m.save(str(filepath))

    with np.load(str(filepath), allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    payload["format_version"] = np.array(999)
    bad_path = tmp_path / "bad.npz"
    np.savez_compressed(str(bad_path), **payload)

    with pytest.raises(ValueError, match="format_version"):
        AMICAMLXNG.load(str(bad_path))


def test_load_rejects_truncated_file(tmp_path):
    """A payload missing one of the required top-level sections (here,
    ``extra``, simulating truncation) raises a named error instead of a
    silent partial load."""
    m = _fitted_model(n_models=1, max_iter=3)
    filepath = tmp_path / "model.npz"
    m.save(str(filepath))

    with np.load(str(filepath), allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files if name != "extra"}
    truncated_path = tmp_path / "truncated.npz"
    np.savez_compressed(str(truncated_path), **payload)

    with pytest.raises(ValueError, match="extra"):
        AMICAMLXNG.load(str(truncated_path))


def test_load_rejects_missing_param_array(tmp_path):
    m = _fitted_model(n_models=1, max_iter=3)
    filepath = tmp_path / "model.npz"
    m.save(str(filepath))

    with np.load(str(filepath), allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files if name != "rho"}
    truncated_path = tmp_path / "truncated.npz"
    np.savez_compressed(str(truncated_path), **payload)

    with pytest.raises(ValueError, match="rho"):
        AMICAMLXNG.load(str(truncated_path))
