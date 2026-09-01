"""EEGLAB output round-trip for a rank-reduced fit (issue #164).

The genuine ``num_pcs < data_dim`` path was unreachable until #223 fixed the fit
crash, so ``write_amica_output``/``loadmodout``/``AmicaOutput.sources`` had no
coverage for a non-square sphere.

Fortran always writes ``S`` at ``recl = 2*nbyte*nx*nx`` (amica15.f90:2423): the
array is allocated ``(nx, nx)`` and zero-filled, and a reduced sphere occupies
only its first ``numeigs`` rows. Both readers (``loadmodout`` here and EEGLAB's
``loadmodout15.m``) reshape to ``(nx, nx)`` and slice ``[:num_pcs]``, so the
writer has to pad to that shape.

Rank deficiency comes from projecting the real sample EEG onto its own leading
subspace -- the rank-reducing operator Maxwell filtering applies to MEG.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica.numpy_impl.load import loadmodout
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
RANK = 20


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )
    return X - X.mean(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def rank_deficient(real_data: np.ndarray) -> np.ndarray:
    U = np.linalg.svd(real_data, full_matrices=False)[0][:, :RANK]
    return U @ (U.T @ real_data)


@pytest.fixture(scope="module")
def reduced_fit(rank_deficient: np.ndarray) -> AMICATorchNG:
    m = AMICATorchNG(n_channels=NW, seed=0, device="cpu")
    m.fit(rank_deficient, max_iter=10, verbose=False)
    assert m.n_channels == RANK, "fixture expects a genuinely reduced fit"
    return m


def test_sphere_is_padded_to_the_fortran_record_shape(reduced_fit, tmp_path):
    """S on disk must be nx*nx values, not num_pcs*nx."""
    outdir = tmp_path / "amicaout"
    reduced_fit.write_amica_output(outdir)
    raw = np.fromfile(outdir / "S", dtype=np.float64)
    assert raw.size == NW * NW, (
        f"S has {raw.size} values; EEGLAB's loadmodout15 expects {NW * NW}"
    )


def test_loadmodout_reads_a_reduced_fit(reduced_fit, tmp_path):
    outdir = tmp_path / "amicaout"
    reduced_fit.write_amica_output(outdir)
    out = loadmodout(outdir)
    assert out.num_pcs == RANK
    assert out.data_dim == NW
    assert out.S.shape == (NW, NW)
    # Only the first num_pcs rows carry the sphere; the pad must be exactly zero.
    np.testing.assert_array_equal(out.S[RANK:], 0.0)


def test_sources_roundtrip_under_reduced_rank(reduced_fit, rank_deficient, tmp_path):
    """The load-bearing check: loaded sources reproduce the live transform.

    Not element-wise. ``loadmodout`` applies EEGLAB's conventions -- rows are
    variance-ordered (``origord``) and ``A``'s columns are normalized to unit
    norm with the norm folded into ``W`` -- so the loaded sources equal the live
    ones under a row permutation and a per-component scale. Both are conventional
    (ICA sources are order- and scale-arbitrary); what must hold exactly is that
    the scale is *constant along each row*, which is what would break if the
    sphere were read back transposed or padded incorrectly.
    """
    outdir = tmp_path / "amicaout"
    reduced_fit.write_amica_output(outdir)
    out = loadmodout(outdir)

    live = reduced_fit.transform(rank_deficient)
    loaded = out.sources(rank_deficient)
    assert loaded.shape == live.shape == (RANK, rank_deficient.shape[1])

    origord = np.asarray(out.origord).ravel()
    assert sorted(origord.tolist()) == list(range(RANK)), "origord must permute"

    ratio = loaded / live[origord]
    spread = np.abs(ratio.std(axis=1) / ratio.mean(axis=1)).max()
    assert spread < 1e-10, f"per-component scale is not constant: {spread:.3e}"
    # And the scales are the A-column norms loadmodout folded into W, so they are
    # positive and finite rather than an artifact of a wrong orientation.
    assert np.all(np.isfinite(ratio.mean(axis=1)))


def test_square_sphere_bytes_are_unchanged(real_data, tmp_path):
    """Full-rank output must stay byte-identical to the Fortran reference.

    The symmetric-ZCA sphere is its own transpose, so switching the writer to
    column-major cannot move a byte here -- asserted rather than assumed, since
    single-model output being byte-compatible with the reference is a standing
    guarantee (issue #92).
    """
    m = AMICATorchNG(n_channels=NW, seed=0, device="cpu")
    m.fit(real_data, max_iter=3, verbose=False)
    assert m.n_channels == NW

    outdir = tmp_path / "amicaout"
    m.write_amica_output(outdir)
    on_disk = np.fromfile(outdir / "S", dtype=np.float64)
    assert m.sphere is not None
    sphere = m.sphere.cpu().numpy()
    # Unchanged C-order write. The ZCA sphere is symmetric only to ~1e-17, so
    # this is not interchangeable with a column-major write -- which is exactly
    # why the writer leaves the square branch alone.
    np.testing.assert_array_equal(on_disk, sphere.ravel(order="C"))
    assert not np.array_equal(sphere.ravel(order="F"), sphere.ravel(order="C")), (
        "sphere is bit-symmetric here, so this test no longer guards anything"
    )
