"""Newton-enabled single-model runs on the full ds002718 sub-002 recording
(issue #351 item 4a, re-running the issue #145 protocol under epic #324).

The paper's claim: with Newton enabled (``do_newton=1``) and independent seeds,
one pamica seed of three (seed 42) put ten of seventy components in a different
basin, the other two matched the reference at ~0.99, and a matched
initialization (A = I, mu = [-1, 0, 1], sbeta = 1, rho = rho0, alpha = 1/3,
c = 0 on both sides) restored ~0.997. The issue #145 records are
``.context/issue-145/full_data_reproduction.md`` and ``same_init_result.md``;
this script repeats that protocol with the code at the epic head and the
pinned v0.3.3 native binary.

Protocol, as in #145:

* data: the full 70-channel x 747,750-frame recording (k ~ 153), rounded to
  float32 as the #145 runs read it (``data.fdt``) and as the binary reads it;
* pamica: ``AMICATorchNG`` on CUDA float64, 2000 iterations, the #145 keyword
  set (``fit_torch_full.py``: ``lrate=0.05``, ``do_newton=True``,
  ``newt_start=50``, ``newt_ramp=10``, ``newtrate=1.0``, ``lratefact=0.5``,
  ``rholrate=0.05``, ``rholratefact=0.5``, ``maxdecs=3``, ``block_size=512``;
  everything else at the constructor default), seeds 42, 7 and 13; the early
  stops are off here so both sides run the full budget, as they did in #145
  (pamica had no ``min_dll`` stop then, and every #145 fit ran 2000 iterations);
* reference: two independent fits of the native binary with the ``input.param``
  settings, ``do_newton=1``, 2000 iterations, early stops off (as the #145
  load-path run had them), here seeded (1 and 2) so the run is reproducible,
  where #145 used clock-seeded runs;
* same-init: both sides from the deterministic start above, the binary through
  its ``load_*`` path.

Subcommands (each writes one ``.npz`` under ``--out-dir``)::

    python newton_seeds.py pamica  --seed 42 --data DATA.npy --out-dir OUT
    python newton_seeds.py pamica  --fixinit --data DATA.npy --out-dir OUT
    python newton_seeds.py fortran --seed 1 --threads 10 --data DATA.npy --out-dir OUT
    python newton_seeds.py fortran --fixinit --threads 10 --data DATA.npy --out-dir OUT
    python newton_seeds.py compare --out-dir OUT
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_ITER = 2000
NMIX = 3
PAMICA_KW = dict(
    n_mix=NMIX, lrate=0.05, do_mean=True, do_sphere=True, do_approx_sphere=True,
    do_newton=True, newt_start=50, newt_ramp=10, newtrate=1.0, lratefact=0.5,
    rholrate=0.05, rholratefact=0.5, maxdecs=3, block_size=512,
    use_min_dll=False, use_grad_norm=False,
)  # fmt: skip
FORTRAN_KW = dict(
    num_models=1, num_mix_comps=NMIX, max_iter=MAX_ITER, do_newton=1,
    use_min_dll=0, use_grad_norm=0, block_size=512,
)  # fmt: skip


def load_data(path: Path) -> np.ndarray:
    """The recording rounded to float32 (what the binary reads), in float64."""
    return np.load(path).astype(np.float32).astype(np.float64)


def fixinit_state(nw: int, rho0: float = 1.5) -> dict[str, np.ndarray]:
    """The deterministic start of the #145 same-init test (reference layout:
    ``mu``/``sbeta``/``rho``/``alpha`` are ``(num_mix, num_comps)``)."""
    return {
        "A": np.eye(nw),
        "mu": np.tile(np.linspace(-1.0, 1.0, NMIX)[:, None], (1, nw)),
        "sbeta": np.ones((NMIX, nw)),
        "rho": np.full((NMIX, nw), rho0),
        "alpha": np.full((NMIX, nw), 1.0 / NMIX),
        "gm": np.ones(1),
        "c": np.zeros((nw, 1)),
    }


def run_pamica(
    data: np.ndarray, seed: int | None, fixinit: bool, device: str, max_iter: int
) -> dict:
    import torch

    from pamica.torch_impl.core import AMICATorchNG

    nw = data.shape[0]

    class FixInitNG(AMICATorchNG):
        """The #145 deterministic start in place of the drawn one."""

        def _initialize_parameters(self):
            super()._initialize_parameters()
            st = fixinit_state(self.n_channels, self.rho0)
            dev, dt = self.device, self.dtype
            self.A = torch.from_numpy(st["A"]).to(dev, dt)  # symmetric: rows == cols
            self.mu = torch.from_numpy(st["mu"]).to(dev, dt)
            self.beta = torch.from_numpy(st["sbeta"]).to(dev, dt)
            self.rho = torch.from_numpy(st["rho"]).to(dev, dt)
            self.alpha = torch.from_numpy(st["alpha"]).to(dev, dt)
            self.c = torch.from_numpy(st["c"]).to(dev, dt)
            self._update_unmixing_matrices()

    cls = FixInitNG if fixinit else AMICATorchNG
    m = cls(
        n_channels=nw, n_models=1, seed=0 if fixinit else seed, device=device,
        dtype=torch.float64, **PAMICA_KW,
    )  # fmt: skip
    t0 = time.time()
    m.fit(data, max_iter=max_iter, verbose=False)
    dt = time.time() - t0
    return {
        "W": np.asarray(m.get_unmixing_matrix(0)),
        "S": np.asarray(m.get_sphere()),
        "ll_history": np.asarray(m.ll_history),
        "final_ll": float(m.final_ll_),
        "stop_reason": str(m.stop_reason),
        "n_newton_fallbacks": int(m.n_newton_fallbacks),
        "end_newtrate": float(m.newtrate),
        "seconds": dt,
    }


def run_fortran(
    data: np.ndarray,
    seed: int | None,
    fixinit: bool,
    threads: int,
    work: Path,
    max_iter: int,
) -> dict:
    from pamica.native import resolver
    from pamica.native.engine import AMICANative

    binary = resolver.resolve("v0.3.3")
    t0 = time.time()
    if not fixinit:
        eng = AMICANative(
            binary=binary, threads=threads, max_threads=threads, timeout=6 * 3600,
            seed=seed, **{**FORTRAN_KW, "max_iter": max_iter},
        )  # fmt: skip
        eng.fit(data)
        out = eng.output_
        assert out is not None
        W = out.W[:, :, 0]
        S = out.S
        ll = np.asarray(out.LL)
    else:
        from pamica.tests.native_oracle import SeedState, run_seeded_reference

        nw, n_samples = data.shape
        fdt = work / "data_f32.fdt"
        if not fdt.exists():
            np.ascontiguousarray(data.T).astype("<f4").tofile(fdt)
        st = fixinit_state(nw)
        state = SeedState(mean=data.mean(axis=1), **st)
        kw = {
            k: v
            for k, v in FORTRAN_KW.items()
            if k not in ("num_models", "num_mix_comps", "max_iter")
        }
        ref = run_seeded_reference(
            state, fdt, work / "fixinit_run", n_samples=n_samples, max_iter=max_iter,
            threads=threads, timeout=6 * 3600, binary=binary, **kw,
        )  # fmt: skip
        # Reference layout: A(nw, nw) with component k in column k, so the
        # sphered unmixing matrix is inv(A); rows are components.
        W = np.linalg.inv(ref.A)
        S = ref.S
        ll = np.asarray(ref.LL)
    return {
        "W": np.asarray(W),
        "S": np.asarray(S),
        "ll_history": ll[ll != 0],
        "final_ll": float(ll[ll != 0][-1]),
        "seconds": time.time() - t0,
    }


def matched(Wa: np.ndarray, Wb: np.ndarray) -> np.ndarray:
    """Hungarian-matched |corr| per component (rows of the unmixing matrices),
    as in the #145 ``compare_local.py``."""
    a = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    b = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    c = np.abs(a @ b.T)
    r, cc = linear_sum_assignment(1 - c)
    return c[r, cc]


def summ(m: np.ndarray) -> dict:
    return {
        "mean": float(m.mean()),
        "min": float(m.min()),
        "n_below_0.9": int((m < 0.9).sum()),
        "worst5": [round(float(x), 3) for x in np.sort(m)[:5]],
    }


def compare(out_dir: Path) -> dict:
    runs = {
        p.stem: dict(np.load(p, allow_pickle=True))
        for p in sorted(out_dir.glob("*.npz"))
    }
    # Sphered-space unmixing rows, as #145 compared them: both sides compute
    # the same symmetric ZCA sphere from the same float32-rounded data.
    Wt = {k: v["W"] for k, v in runs.items()}
    report: dict = {
        "runs": {k: {"final_ll": float(v["final_ll"])} for k, v in runs.items()}
    }
    fort = sorted(k for k in Wt if k.startswith("fortran_seed"))
    pam = sorted(k for k in Wt if k.startswith("pamica_seed"))
    pairs = {}
    for a, b in itertools.combinations(fort, 2):
        pairs[f"{a} vs {b}"] = summ(matched(Wt[a], Wt[b]))
    for a in pam:
        for b in fort:
            pairs[f"{a} vs {b}"] = summ(matched(Wt[a], Wt[b]))
    for a, b in itertools.combinations(pam, 2):
        pairs[f"{a} vs {b}"] = summ(matched(Wt[a], Wt[b]))
    if "pamica_fixinit" in Wt and "fortran_fixinit" in Wt:
        pairs["pamica_fixinit vs fortran_fixinit"] = summ(
            matched(Wt["pamica_fixinit"], Wt["fortran_fixinit"])
        )
        for b in fort:
            pairs[f"fortran_fixinit vs {b}"] = summ(
                matched(Wt["fortran_fixinit"], Wt[b])
            )
        for a in pam:
            pairs[f"pamica_fixinit vs {a}"] = summ(matched(Wt["pamica_fixinit"], Wt[a]))
    report["pairs"] = pairs
    for k, v in runs.items():
        extra = {
            x: v[x].item()
            for x in ("stop_reason", "n_newton_fallbacks", "end_newtrate", "seconds")
            if x in v
        }
        report["runs"][k].update(
            {
                x: (y if not isinstance(y, bytes) else y.decode())
                for x, y in extra.items()
            }
        )
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pamica", "fortran", "compare"])
    p.add_argument("--seed", type=int)
    p.add_argument("--fixinit", action="store_true")
    p.add_argument("--data", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--device", default="cuda")
    # Smoke-test knobs only; the protocol is the full recording at 2000 iterations.
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--frames", type=int, default=None)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.cmd == "compare":
        rep = compare(a.out_dir)
        (a.out_dir / "newton_seeds_report.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
        return
    data = load_data(a.data)
    if a.frames:
        data = data[:, : a.frames]
    tag = "fixinit" if a.fixinit else f"seed{a.seed}"
    if a.cmd == "pamica":
        res = run_pamica(data, a.seed, a.fixinit, a.device, a.max_iter)
    else:
        res = run_fortran(data, a.seed, a.fixinit, a.threads, a.out_dir, a.max_iter)
    np.savez(a.out_dir / f"{a.cmd}_{tag}.npz", **res)
    print(
        f"{a.cmd} {tag}: final_ll={res['final_ll']:.6f} iters={len(res['ll_history'])} "
        + " ".join(
            f"{k}={res[k]}"
            for k in ("stop_reason", "n_newton_fallbacks", "end_newtrate")
            if k in res
        )
        + f" time={res['seconds']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
