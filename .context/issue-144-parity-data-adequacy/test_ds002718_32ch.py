"""Single-model Fortran-vs-NG parity on a data-adequate real recording (k>=60
per issue #90's documented threshold), using the bundled amica15mac binary
(also used for Table 1's bundled-sample Amari row; Table 1's headline
correlation instead uses the external tier of ../../benchmarks/reproduce_table1.py,
issue #144) and AMICATorchNG, instead of the under-determined bundled
32ch/30504-frame EEGLAB tutorial recording (k=29.8).

Data: OpenNeuro ds002718 sub-002 (Wakeman-Henson faces), first 32 of the first
70 EEG channels, first N frames -- real data, not committed (not bundled; see
benchmarks/README_dimsweep.md for the download recipe this mirrors).

    uv run python .context/issue-144-parity-data-adequacy/test_ds002718_32ch.py <npy> [n_seeds] [max_iter]
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pamica.torch_impl import AMICATorchNG  # noqa: E402

BIN = REPO / "pamica/sample_data/amica15mac"
INPUT_PARAM = REPO / "pamica/sample_data/input.param"


def xcorr(Wa, Wb):
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(na @ nb.T)
    r, c = linear_sum_assignment(1 - corr)
    return corr[r, c]


def main():
    npy_path = Path(sys.argv[1])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    data = np.load(npy_path).astype(np.float64)
    nw, field = data.shape
    k = field / nw**2
    print(f"data: {nw} channels x {field} frames, k={k:.1f}, max_iter={max_iter}")

    work = Path(tempfile.mkdtemp(prefix="ds002718_parity_"))
    fdt_path = work / "data.fdt"
    data.astype(np.float32).T.tofile(fdt_path)

    results = []
    for seed in range(n_seeds):
        d = work / f"run_{seed}"
        (d / "fortran_output").mkdir(parents=True, exist_ok=True)
        shutil.copy(fdt_path, d / "data.fdt")
        lines = []
        for ln in INPUT_PARAM.read_text().splitlines():
            if ln.startswith("files"):
                lines.append("files ./data.fdt")
            elif ln.startswith("outdir"):
                lines.append("outdir ./fortran_output/")
            elif ln.startswith("data_dim"):
                lines.append(f"data_dim {nw}")
            elif ln.startswith("field_dim"):
                lines.append(f"field_dim {field}")
            elif ln.startswith("max_iter"):
                lines.append(f"max_iter {max_iter}")
            elif ln.startswith("pcakeep"):
                # AMICATorchNG has no PCA source reduction (n_sources ==
                # n_channels always), so pcakeep must track nw or Fortran's
                # W comes back reduced-rank (the template's literal 32 is
                # only correct by coincidence at nw=32).
                lines.append(f"pcakeep {nw}")
            elif ln.startswith("use_min_dll"):
                # Force the full max_iter budget instead of Fortran's own
                # early-stopping: otherwise Fortran can converge and stop
                # well short of max_iter while AMICATorchNG (no equivalent)
                # keeps optimizing past that point, drifting weakly-
                # determined components to a different, still-valid optimum.
                lines.append("use_min_dll 0")
            elif ln.startswith("use_grad_norm"):
                lines.append("use_grad_norm 0")
            else:
                lines.append(ln)
        (d / "input.param").write_text("\n".join(lines) + "\n")

        orig = os.getcwd()
        os.chdir(d)
        try:
            r = subprocess.run(
                [str(BIN), "input.param"], capture_output=True, text=True, timeout=1800
            )
        finally:
            os.chdir(orig)
        if r.returncode != 0:
            print(f"seed {seed}: Fortran failed: {r.stderr[-400:]}")
            continue
        W_fortran = np.fromfile(d / "fortran_output/W", dtype=np.float64).reshape(
            nw, nw, order="F"
        )
        fort_ll = next(
            (
                float(ln.split("LL =")[1].split()[0])
                for ln in reversed(r.stdout.splitlines())
                if "LL =" in ln
            ),
            None,
        )
        if fort_ll is None:
            print(f"seed {seed}: WARNING could not parse LL from Fortran stdout")
            fort_ll = float("nan")

        m = AMICATorchNG(
            n_channels=nw, n_models=1, n_mix=3, block_size=512, lrate=0.05,
            minlrate=1e-8, lratefact=0.5, maxdecs=3, do_newton=True,
            newt_start=50, newt_ramp=10, newtrate=1.0, rho0=1.5, minrho=1.0,
            maxrho=2.0, rholrate=0.05, rholratefact=0.5, invsigmin=0.0,
            invsigmax=100.0, doscaling=True, scalestep=1, seed=seed, device="cpu",
        )  # fmt: skip
        m.fit(data, max_iter=max_iter, verbose=False)
        if m.stop_reason in AMICATorchNG._DEGENERATE_STOP_REASONS:
            print(
                f"seed {seed}: NG fit ended degenerate (stop_reason={m.stop_reason!r}), skipping"
            )
            continue
        W_ng = m.get_unmixing_matrix(0)

        corrs = xcorr(W_fortran, W_ng)
        print(
            f"seed {seed}: mean_corr={corrs.mean():.4f} min={corrs.min():.4f} "
            f"fortran_LL={fort_ll:.4f} ng_LL={m.final_ll_:.4f}"
        )
        results.append(corrs.mean())

    if results:
        print(
            f"\n{nw}ch/{field}fr (k={k:.1f}), n={len(results)}: "
            f"mean={np.mean(results):.4f} range={np.min(results):.4f}-{np.max(results):.4f}"
        )

    if len(results) < n_seeds:
        print(f"\n{n_seeds - len(results)}/{n_seeds} seed(s) failed or were degenerate")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
