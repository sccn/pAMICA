"""Parity against the bundled ``amicaout`` fixture (200 reference iterations of
the bundled macOS binary, ``input.param`` settings, unseeded), for every
backend. The protocol of the epic #324 Phase 7 and Phase 11 before/after
figures (PR 340 and PR 348), kept here as the issue #351 run record:
each backend fits the sample through ``input.param`` for K iterations, seed 42,
and is compared with the fixture's W @ S and its LL at iteration K.

    uv run python .context/issue-351/fixture_parity.py ROOT BACKEND [K]

ROOT is a pamica source tree (the worktree, or an older checkout)."""

import sys
import tempfile
import time

root = sys.argv[1]
sys.path.insert(0, root)
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import pamica  # noqa: E402

assert pamica.__file__.startswith(root), pamica.__file__
from pamica.torch_impl.utils import load_eeglab_data  # noqa: E402

backend = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 200
SD = root + "/pamica/sample_data/"
vspec = importlib.util.spec_from_file_location(
    "vi", root + "/validate_implementations.py"
)
vi = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(vi)
X = load_eeglab_data(SD + "eeglab_data.fdt", data_dim=32, field_dim=30504).astype(
    np.float64
)
AO = SD + "amicaout/"
W_ref = np.fromfile(AO + "W", np.float64).reshape(32, 32, order="F")
S_ref = np.fromfile(AO + "S", np.float64).reshape(32, 32, order="F")
LL_ref = np.fromfile(AO + "LL", np.float64)
LL_ref = LL_ref[LL_ref != 0]

t0 = time.time()
if backend == "numpy":
    from pamica.numpy_impl.core import AMICA as AMICA_NumPy

    with tempfile.TemporaryDirectory() as td:
        m = AMICA_NumPy(
            params_file=SD + "input.param",
            max_iter=K,
            seed=42,
            use_tqdm=False,
            outdir=td,
            writestep=10**6,
        )
        m.fit(X)
    Wp = m.get_weights() @ m.sphere
    ll = float(m.ll[-1])
    n_it = len(m.ll)
    stop = m.stop_reason
else:
    from pamica import AMICA

    a = AMICA.from_params_file(SD + "input.param", backend=backend)
    a.fit(X, max_iter=K, seed=42)
    mm = a.model_
    Wp = np.asarray(mm.get_unmixing_matrix(0)) @ np.asarray(mm.get_sphere())
    ll = float(mm.final_ll_)
    n_it = len(mm.ll_history)
    stop = mm.stop_reason
Wr = W_ref @ S_ref
an = Wp / np.linalg.norm(Wp, axis=1, keepdims=True)
bn = Wr / np.linalg.norm(Wr, axis=1, keepdims=True)
c = np.abs(an @ bn.T)
ri, ci = linear_sum_assignment(1 - c)
print(
    f"{backend} K={K}: LL {ll:.6f} fixture {LL_ref[K - 1]:.6f} |dLL|={abs(ll - LL_ref[K - 1]):.2e} "
    f"corr={c[ri, ci].mean():.6f} min={c[ri, ci].min():.4f} amari={vi.amari_distance(Wp, Wr):.3e} "
    f"stop={stop} iters={n_it} ({time.time() - t0:.0f}s)"
)
