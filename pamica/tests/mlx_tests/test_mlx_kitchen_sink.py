"""One end-to-end lifecycle test exercising every non-fitting-parity feature
of epic #278 together on a single MLX model -- PR #318 review item 8.

Every other test file in this suite isolates ONE feature (sharing, rejection,
restarts, the adaptive-PDF switcher, LLt/export/scoring/MIR) against a
default-off baseline, per ``.rules/backend_parity.md``'s no-drift-per-feature
discipline. That leaves a real gap: nothing pins that the features still
compose when several are on at once on the SAME fitted model, through the
FULL public surface (every accessor, both persistence paths, and the EEGLAB
export), in one pass. This file is that one composition pin, not a substitute
for the isolated tests.

Combination fitted: ``n_models=2``, ``share_comps=True`` (aggressive enough
to force a genuine merge, the same recipe as
``test_mlx_sharing.py::test_two_model_share_fit_completes_and_merges``),
``do_reject=True`` (a genuine rejection), ``n_restarts=2`` (a genuine
restart-state round trip), ``pdftype=1`` (the adaptive kurtosis switcher,
which requires ``n_mix=1``), and ``keep_best=True`` passed explicitly even
though ``share_comps``/``do_reject`` each independently disable it (the
"inactive by exclusion" case named in the review) -- so the test also pins
that passing it does not raise or otherwise misbehave under the exclusion.

Real sample EEG only (no synthetic/mock).
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pamica.mlx_impl import AMICAMLXNG  # noqa: E402  (after the MLX importorskip)
from pamica.numpy_impl.load import loadmodout  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
BLOCK = 1024

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


def test_kitchen_sink_lifecycle(real_data, caplog):
    """Fit with every non-fitting feature on at once, then walk the full
    public surface: every accessor, ``state_dict``/``from_state_dict``,
    ``.npz`` ``save``/``load``, and ``write_amica_output``/``loadmodout``,
    asserting internal consistency at each stage."""
    m = AMICAMLXNG(
        n_channels=NW,
        n_models=2,
        n_mix=1,
        seed=7,
        block_size=BLOCK,
        share_comps=True,
        share_start=4,
        share_iter=8,
        comp_thresh=0.9,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
        n_restarts=2,
        pdftype=1,
        keep_best=True,
    )
    with caplog.at_level(logging.WARNING, logger="pamica.mlx_impl.core"):
        m.fit(real_data, max_iter=25, verbose=False)

    # --- fit-level sanity: non-degenerate, and every feature actually fired,
    # not just accepted -- a no-op combination would make the rest of this
    # test vacuous.
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert m.final_ll_ is not None and np.isfinite(m.final_ll_)
    assert m.restart_seeds_ == [7, 8], "n_restarts=2 must reseed from seed=7"
    assert m.good_idx is not None
    n_rejected = real_data.shape[1] - int(m.good_idx.size)
    assert n_rejected > 0, "do_reject rejected nothing; test setup is too gentle"
    groups = m.shared_components()
    assert groups, "share_comps merged nothing; test setup is too gentle"
    pdt = np.array(m.pdtype)
    assert set(np.unique(pdt).tolist()) == {1, 4}, (
        "pdftype=1 never switched a source; test setup is too gentle"
    )
    # keep_best=True was passed but both exclusions apply: confirm the model
    # logged the exclusion rather than silently ignoring the request.
    assert any("keep_best is inactive under" in r.getMessage() for r in caplog.records)

    # --- every accessor ---
    assert int(np.array(m.comp_used).sum()) < m.n_comps
    assert m.n_channels_in == NW
    for h in range(m.n_models):
        tr = m.transform(real_data, model_idx=h)
        assert tr.shape == (NW, real_data.shape[1])
        assert np.all(np.isfinite(tr))
        assert m.get_pdftype(model_idx=h).shape == (NW,)
        mix = m.get_mixing_matrix(model_idx=h)
        assert mix.shape == (NW, NW) and np.all(np.isfinite(mix))
        sens = m.get_sensor_mixing_matrix(model_idx=h)
        assert sens.shape == (NW, NW) and np.all(np.isfinite(sens))
        unmix = m.get_unmixing_matrix(model_idx=h)
        assert unmix.shape == (NW, NW) and np.all(np.isfinite(unmix))
        assert m.get_rho(model_idx=h).shape == (1, NW)
        order, svar = m.variance_order(model_idx=h, return_svar=True)
        assert order.shape == (NW,) and svar.shape == (NW,)
        assert set(order.tolist()) == set(range(NW))

    mir_nats, mir_var = m.mir(real_data, model_idx=0)
    assert np.isfinite(mir_nats) and np.isfinite(mir_var)
    pmi = m.pmi(real_data, model_idx=0)
    assert pmi.shape == (NW, NW)

    lht = m.model_loglik(real_data)
    assert lht.shape == (m.n_models, real_data.shape[1])
    assert np.all(np.isfinite(lht))
    prob = m.model_probability(real_data)
    assert prob.shape == (m.n_models, real_data.shape[1])
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-4)

    # --- state_dict / from_state_dict ---
    state = m.state_dict()
    restored = AMICAMLXNG.from_state_dict(state)
    np.testing.assert_array_equal(np.array(restored.A), np.array(m.A))
    np.testing.assert_array_equal(np.array(restored.comp_list), np.array(m.comp_list))
    assert restored.shared_components() == groups
    assert restored.stop_reason == m.stop_reason
    assert restored.final_ll_ == m.final_ll_

    # --- .npz save/load ---
    with tempfile.TemporaryDirectory() as d:
        savepath = Path(d) / "kitchen_sink.npz"
        m.save(str(savepath))
        loaded = AMICAMLXNG.load(str(savepath))
        np.testing.assert_array_equal(np.array(loaded.A), np.array(m.A))
        assert loaded.shared_components() == groups
        assert loaded.good_idx is not None
        assert int(loaded.good_idx.size) == int(m.good_idx.size)

        # --- write_amica_output / loadmodout ---
        outdir = Path(d) / "kitchen_sink_amicaout"
        m.write_amica_output(outdir)
        out = loadmodout(outdir)
        assert out.W.shape == (NW, NW, m.n_models)
        assert out.LL is not None
        assert out.Lt is not None
        assert int((out.Lt == 0.0).sum()) == n_rejected
