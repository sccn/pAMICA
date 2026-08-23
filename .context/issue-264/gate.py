"""Pre-registered float32 go/no-go gate for MLX Newton (issue #264, epic #260).

Run from the repo root on Apple Silicon with MLX installed:

    uv run python .context/issue-264/gate.py

Criteria (fixed BEFORE the port was written; see the phase plan):

G1 formula     the finalized curvature and one warmed Newton M-step match a
               float64 AMICATorchNG twin from a matched state, within float32
               tolerance (rel ~1e-4, the sharing cross-backend precedent).
G2 stability   full-data MLX Newton fits at the default schedule and at the
               #24 sample_params schedule (newt_start=50, newtrate=1.0,
               lrate=0.05, block_size=512), >= 3 seeds, 100-150 iterations:
               all finite, stop_reason non-degenerate.
G3 quality     mlx.final_ll_ >= torch_f64_newton.final_ll_ - 0.05 at a matched
               100-iteration budget, and n_newton_fallbacks == 0 on the
               sample_params config (the PyTorch backend's own bar on this data).
G4 conditioning  min(prod - 1) over ACCEPTED source pairs across Newton
               iterations stays >= 1e-3 (the float64 reference measurement on
               this data is 2.09, .context/issue-145/setup_and_config.md:67-77).
G5 overshoot   report max(ll_history) - final_ll_ per seed (MLX has no
               keep_best; > 0.05 anywhere warrants a follow-up issue).

Decision rule: all pass -> GO float32. G1/G2 failing in a way traceable to
float32 curvature -> MIDDLE (float64 curvature on MLX's CPU stream). Both fail
-> NO-GO. Prints the verdict; the recorded numbers live in newton_findings.md.
"""

import logging
import time

import mlx.core as mx
import numpy as np
import torch

from pamica.mlx_impl.core import AMICAMLXNG
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

logging.basicConfig(level=logging.WARNING)

DATA = "pamica/sample_data/eeglab_data.fdt"
NW, FIELD, NMIX = 32, 30504, 3
SEEDS = (1, 3, 42)


def real_data(n=None):
    d = load_eeglab_data(DATA, data_dim=NW, field_dim=FIELD).astype(np.float64)
    return d if n is None else d[:, :n]


def relerr(a, b):
    scale = np.maximum(np.abs(b), np.abs(b).max() * 1e-6)
    return float(np.max(np.abs(a - b) / scale))


class _TracedMLX(AMICAMLXNG):
    """Records, per Newton iteration, the minimum ``prod - 1`` over the
    off-diagonal source pairs that were ACCEPTED (G4). Observation only: it
    recomputes the published quantity from the production method's own inputs
    and does not alter the fit."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.margins: list[float] = []

    def _newton_direction(self, dA_h, sigma2_h, lambda_h, kappa_h):
        H, posdef = super()._newton_direction(dA_h, sigma2_h, lambda_h, kappa_h)
        if posdef:
            s = np.array(sigma2_h, dtype=np.float64)
            k = np.array(kappa_h, dtype=np.float64)
            prod = (s[:, None] * k[None, :]) * (s[None, :] * k[:, None])
            off = ~np.eye(prod.shape[0], dtype=bool)
            self.margins.append(float((prod[off] - 1.0).min()))
        return H, posdef


# --- G1 ---------------------------------------------------------------------


def gate_g1():
    print("\n=== G1: formula vs float64 torch twin ===")
    data = real_data(4096)
    rows = []
    for warmup in (1, 55):
        m = AMICAMLXNG(
            n_channels=NW,
            n_mix=NMIX,
            seed=3,
            block_size=1024,
            do_newton=True,
            newt_start=0,
        )
        x = m._preprocess(data)
        m._initialize_parameters()
        for it in range(warmup):
            m.iteration = it
            m._update_parameters(m._accumulate_blocks(x), x.shape[1])
        m.iteration = warmup
        mx.eval(m.A, m.mu, m.alpha, m.beta, m.rho, m.gm, m.c)

        ng = AMICATorchNG(
            n_channels=NW,
            n_models=1,
            n_mix=NMIX,
            device="cpu",
            dtype=torch.float64,
            do_newton=True,
            newt_start=0,
            block_size=1024,
            seed=3,
            keep_best=False,
        )
        ng._initialize_parameters()
        for name in ("A", "mu", "alpha", "beta", "rho", "gm", "c"):
            setattr(
                ng,
                name,
                torch.from_numpy(np.array(getattr(m, name)).astype(np.float64)),
            )
        ng.comp_list = torch.from_numpy(np.array(m.comp_list).astype(np.int64))
        ng.sphere = torch.from_numpy(m._sphere_np.copy())
        ng._sphere_pinv = None
        ng.lrate, ng.lrate_cap = m.lrate, m.lrate_cap
        ng.newtrate, ng.rholrate = m.newtrate, m.rholrate
        ng.iteration, ng.sldet = m.iteration, m.sldet
        ng._update_unmixing_matrices()
        xt = torch.from_numpy(np.array(x).astype(np.float64))

        acc, nacc = m._accumulate_blocks(x), ng._accumulate_blocks(xt)
        errs = {}
        for name, a, b in zip(
            ("sigma2", "lambda", "kappa"),
            m._finalize_newton_stats(acc),
            ng._finalize_newton_stats(nacc),
        ):
            errs[name] = relerr(np.array(a, dtype=np.float64), b.numpy().T)
        m._update_parameters(acc, x.shape[1])
        ng._update_parameters(nacc, xt.shape[1])
        mx.eval(m.A)
        gap = float(np.abs(np.array(m.A, dtype=np.float64) - ng.A.numpy()).max())
        rows.append((warmup, errs, gap, m.n_newton_fallbacks, ng.n_newton_fallbacks))
        print(
            f"  warmup={warmup:3d}  sigma2 {errs['sigma2']:.2e}  "
            f"lambda {errs['lambda']:.2e}  kappa {errs['kappa']:.2e}  "
            f"A-gap {gap:.2e}  fallbacks mlx/torch {m.n_newton_fallbacks}/"
            f"{ng.n_newton_fallbacks}"
        )
    ok = all(max(e.values()) < 1e-4 and g < 1e-4 for _, e, g, _, _ in rows)
    print(f"  G1: {'PASS' if ok else 'FAIL'} (bar: rel < 1e-4, A-gap < 1e-4)")
    return ok


# --- G2 / G4 / G5 -----------------------------------------------------------


def gate_g2_g4_g5():
    print("\n=== G2/G4/G5: stability, conditioning, overshoot ===")
    data = real_data()
    configs = {
        "default (newt_start=20, newtrate=0.5, lrate=0.1, block=8192)": dict(
            do_newton=True
        ),
        "sample_params (newt_start=50, newtrate=1.0, lrate=0.05, block=512)": dict(
            do_newton=True,
            newt_start=50,
            newtrate=1.0,
            lrate=0.05,
            block_size=512,
        ),
    }
    finite_ok, margin_ok, overshoots, results = True, True, [], {}
    for label, kw in configs.items():
        print(f"  {label}")
        for seed in SEEDS:
            t0 = time.time()
            m = _TracedMLX(n_channels=NW, n_mix=NMIX, seed=seed, **kw)
            m.fit(data, max_iter=150, verbose=False)
            hist = np.asarray(m.ll_history, dtype=float)
            over = float(hist.max() - m.final_ll_)
            margin = min(m.margins) if m.margins else float("nan")
            degenerate = m.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS
            finite_ok &= bool(np.all(np.isfinite(hist))) and not degenerate
            margin_ok &= bool(m.margins) and margin >= 1e-3
            overshoots.append(over)
            results[(label, seed)] = (m.final_ll_, m.stop_reason, m.n_newton_fallbacks)
            print(
                f"    seed={seed:2d}  ll={m.final_ll_:.5f}  stop={m.stop_reason:<9s}"
                f"  iters={len(hist):3d}  fallbacks={m.n_newton_fallbacks}"
                f"  min(prod-1)={margin:.3f}  overshoot={over:.3e}"
                f"  ({time.time() - t0:.0f}s)"
            )
    print(f"  G2: {'PASS' if finite_ok else 'FAIL'} (all finite, non-degenerate stop)")
    print(f"  G4: {'PASS' if margin_ok else 'FAIL'} (bar: min(prod-1) >= 1e-3)")
    print(
        f"  G5: max overshoot {max(overshoots):.3e} over {len(overshoots)} fits "
        f"({'no keep_best follow-up needed' if max(overshoots) <= 0.05 else 'FOLLOW-UP WARRANTED'})"
    )
    return finite_ok, margin_ok, max(overshoots), results


# --- G3 ---------------------------------------------------------------------


def gate_g3():
    print("\n=== G3: quality vs float64 torch Newton, matched 100 iters ===")
    data = real_data()
    ok = True
    for label, mlx_kw, ng_kw in (
        ("default", dict(), dict()),
        (
            "sample_params",
            dict(newt_start=50, newtrate=1.0, lrate=0.05, block_size=512),
            dict(newt_start=50, newtrate=1.0, lrate=0.05, block_size=512),
        ),
    ):
        m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=3, do_newton=True, **mlx_kw)
        m.fit(data, max_iter=100, verbose=False)
        ng = AMICATorchNG(
            n_channels=NW,
            n_models=1,
            n_mix=NMIX,
            seed=3,
            device="cpu",
            dtype=torch.float64,
            do_newton=True,
            **ng_kw,
        )
        ng.fit(data, max_iter=100, verbose=False)
        ng_last = AMICATorchNG(
            n_channels=NW,
            n_models=1,
            n_mix=NMIX,
            seed=3,
            device="cpu",
            dtype=torch.float64,
            do_newton=True,
            keep_best=False,
            **ng_kw,
        )
        ng_last.fit(data, max_iter=100, verbose=False)
        passed = m.final_ll_ >= ng.final_ll_ - 0.05
        ok &= bool(passed)
        print(
            f"  {label:14s} mlx={m.final_ll_:.5f}  torch_f64(keep_best)="
            f"{ng.final_ll_:.5f}  torch_f64(last)={ng_last.final_ll_:.5f}  "
            f"delta={m.final_ll_ - ng.final_ll_:+.5f}  "
            f"mlx_fallbacks={m.n_newton_fallbacks}  {'PASS' if passed else 'FAIL'}"
        )
        if label == "sample_params":
            ok &= m.n_newton_fallbacks == 0
    print(f"  G3: {'PASS' if ok else 'FAIL'} (bar: >= torch_f64 - 0.05; 0 fallbacks)")
    return ok


if __name__ == "__main__":
    g1 = gate_g1()
    g2, g4, g5, _ = gate_g2_g4_g5()
    g3 = gate_g3()
    verdict = "GO (float32)" if all((g1, g2, g3, g4)) else "NOT GO -- see criteria"
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"G1={g1} G2={g2} G3={g3} G4={g4} G5 max overshoot={g5:.4f}")
