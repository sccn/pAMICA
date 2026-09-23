"""Refit the pamica side of the bundled tier's multi-model ensemble with the
pamica tree at ROOT (issue #351 item 3), for a comparison against the SAME
reference fits: the reference half is copied from ``--fortran-npz`` (the
bundled tier's ``bundled_multimodel_ensemble.npz``), the pamica half is fitted
here with ``reproduce_table1.run_multimodel_ensemble``'s keyword set (seeds
1-20, 100 iterations). With ROOT at the code before epic #324 this separates the
epic's effect on the ensemble from the draw of the reference runs.

    uv run python .context/issue-351/multimodel_pamica_fits.py ROOT --fortran-npz PATH OUT.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--fortran-npz", type=Path, required=True)
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(root))
    import pamica
    from pamica import AMICA
    from pamica.torch_impl.utils import load_eeglab_data

    assert pamica.__file__.startswith(str(root)), pamica.__file__
    data = load_eeglab_data(
        str(root / "pamica/sample_data/eeglab_data.fdt"),
        data_dim=32,
        field_dim=30504,
        dtype=np.float32,
    ).astype(np.float64)
    Gs, G_ll = [], []
    for i in range(a.n):
        model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
        model.fit(
            data, max_iter=100, lrate=0.05, do_newton=True, seed=1 + i,
            block_size=512, minlrate=1e-8, lratefact=0.5, maxdecs=3, newt_start=50,
            newt_ramp=10, newtrate=1.0, rho0=1.5, minrho=1.0, maxrho=2.0,
            rholrate=0.05, rholratefact=0.5, invsigmin=1e-8, invsigmax=100.0,
            doscaling=True, scalestep=1,
        )  # fmt: skip
        assert model.converged_, model.stop_reason_
        Gs.append(
            np.vstack([model.get_unmixing_matrix(0), model.get_unmixing_matrix(1)])
        )
        G_ll.append(float(model.final_ll_))
        print(f"pamica {i + 1}/{a.n}: LL={G_ll[-1]:.4f}", flush=True)
    f = np.load(a.fortran_npz)
    np.savez(a.out, Fs=f["Fs"], F_ll=f["F_ll"], Gs=np.array(Gs), G_ll=np.array(G_ll))


if __name__ == "__main__":
    main()
