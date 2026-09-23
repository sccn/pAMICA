"""The ``share_comps`` figures of ``docs/guides/amica-differences.md``
("Component sharing compares and ties components"), re-measured under epic
#324 (issue #351 item 6). Protocols as the page and the Phase 8 run records
(PR 342) describe them, on the bundled sample EEG:

a. end to end: 2 models, seed 42, Newton on, 300 iterations, ``share_start``
   100, ``share_iter`` 100, ``comp_thresh`` 0.95, PyTorch defaults otherwise;
   every merge with its iteration and the |cos| of the two scalp maps, and the
   final log-likelihood with sharing on and off;
b. early scans: 2 models, seed 42, PyTorch defaults, one scan at iteration 8
   or 20 (``share_iter`` 100), fit to ``share_start + 2``; merges per threshold;
c. the reference's similarity on its own early state: the binary runs 8
   iterations from pamica's seed-42 initialization (the test suite's ``_OPT``
   optimizer settings), and the reference's formula (``Spinv2 = Spinv^T Spinv``,
   i.e. the scalp maps ``pinv(S) @ A``) on that state is compared with
   PyTorch's and NumPy's scans of their own 8-iteration states;
d. the collapse recipe of ``test_component_rows.py`` (4096 samples, seed 23,
   ``share_start`` = ``share_iter`` = 11, ``comp_thresh`` 0.9): merges at the
   first scan, the second model's ``gm`` over the next iterations, and the
   iteration where the fit goes non-finite.

    uv run python .context/issue-351/share_comps_counts.py OUT.json
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from pamica import AMICA_NumPy
from pamica.numpy_impl.utils import identify_shared_components
from pamica.tests.native_oracle import run_seeded_reference, seed_from_torch
from pamica.tests.test_component_rows import _COLLAPSE, _OPT, _REF_OPT, DATA_FILE
from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

NW, FIELD = 32, 30504
X = load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(np.float64)


def logged(base):
    """``base`` with every merge recorded: (1-based iteration, model, source,
    merged-into model, source, |cos| of the two scalp maps before the scan)."""

    class Logged(base):
        merge_log: list

        def _identify_shared_comps(self):
            before = self.comp_list.cpu().numpy().copy()
            maps = [self.get_sensor_mixing_matrix(h) for h in range(self.n_models)]
            maps = [q / np.linalg.norm(q, axis=0) for q in maps]
            super()._identify_shared_comps()
            after = self.comp_list.cpu().numpy()
            for hh in range(self.n_models):
                for ii in range(self.n_channels):
                    if after[ii, hh] != before[ii, hh]:
                        tgt = after[ii, hh]
                        h, i = next(
                            (h, i)
                            for h in range(self.n_models)
                            for i in range(self.n_channels)
                            if before[i, h] == tgt
                        )
                        cos = float(abs(maps[h][:, i] @ maps[hh][:, ii]))
                        self.merge_log.append((self.iteration + 1, h, i, hh, ii, cos))

    return Logged


def part_a() -> dict:
    kw = dict(n_channels=NW, n_models=2, n_mix=3, device="cpu", seed=42, do_newton=True)
    share = dict(share_comps=True, share_start=100, share_iter=100, comp_thresh=0.95)
    off = AMICATorchNG(**kw).fit(X, max_iter=300, verbose=False)
    m = logged(AMICATorchNG)(**kw, **share)
    m.merge_log = []
    m.fit(X, max_iter=300, verbose=False)
    groups = []
    for g in m.shared_components():
        v = [m.get_mixing_matrix(h)[:, i] for h, i in g]
        groups.append(
            {"group": g, "identical": all(np.array_equal(v[0], x) for x in v[1:])}
        )
    return {
        "share_off_final_ll": float(off.final_ll_),
        "share_off_stop": str(off.stop_reason),
        "share_off_newton_fallbacks": int(off.n_newton_fallbacks),
        "share_on_final_ll": float(m.final_ll_),
        "share_on_stop": str(m.stop_reason),
        "share_on_newton_fallbacks": int(m.n_newton_fallbacks),
        "merges": [
            {
                "iter": r[0],
                "model": r[1],
                "source": r[2],
                "into_model": r[3],
                "into_source": r[4],
                "map_abs_cos": r[5],
            }
            for r in m.merge_log
        ],  # fmt: skip
        "unique_components": int(m.comp_used.sum()),
        "groups": groups,
    }


def part_b() -> dict:
    out = {}
    for start in (8, 20):
        for thr in (0.9, 0.95, 0.99):
            m = AMICATorchNG(
                NW, n_models=2, seed=42, share_comps=True, share_start=start,
                share_iter=100, comp_thresh=thr, device="cpu",
            )  # fmt: skip
            m.fit(X, max_iter=start + 2, verbose=False)
            merged = 2 * NW - len(np.unique(m.comp_list.cpu().numpy()))
            out[f"scan_at_{start}_thr_{thr}"] = {
                "merged": int(merged),
                "final_ll": float(m.final_ll_),
                "stop": str(m.stop_reason),
            }
    return out


def part_c(work: Path) -> dict:
    pre = 8

    def torch_model(**kw):
        return AMICATorchNG(
            n_channels=NW, n_models=2, n_mix=3, seed=42, device="cpu",
            dtype=torch.float64, keep_best=False, **{**_OPT, **kw},
        )  # fmt: skip

    init = torch_model()
    init._preprocess(X)
    init._initialize_parameters()
    default = init.comp_list.numpy().copy()
    seed = dataclasses.replace(seed_from_torch(init), comp_list=default)
    ref8 = run_seeded_reference(
        seed, DATA_FILE, work / "c", n_samples=FIELD, max_iter=pre, num_models=2,
        **{**_REF_OPT, "share_start": pre, "share_iter": 100, "share_comps": 0},
    )  # fmt: skip
    t8 = torch_model(share_comps=False)
    t8.fit(X, max_iter=pre, verbose=False)
    with tempfile.TemporaryDirectory() as td:
        n8 = AMICA_NumPy(
            num_models=2, num_mix=3, max_iter=pre, seed=42, use_tqdm=False,
            outdir=str(td), do_opt_block=False, writestep=10**6, **_OPT,
        )  # fmt: skip
        n8.fit(X)
    ref_maps = np.linalg.pinv(ref8.S) @ ref8.A
    out = {
        "state_A_diff_torch_vs_ref": float(np.abs(t8.A.numpy().T - ref8.A).max()),
        "state_LL_diff_torch_vs_ref": float(
            np.abs(np.asarray(t8.ll_history) - ref8.LL).max()
        ),
    }
    for thr in (0.9, 0.95, 0.99):
        res = {}
        lists = {}
        for name, maps in (
            ("reference_state", ref_maps),
            ("torch_state", t8._component_sensor_maps()),
            ("numpy_state", n8._component_sensor_maps()),
        ):
            new, _ = identify_shared_components(maps, default.copy(), thr)
            lists[name] = new
            res[name] = int(2 * NW - len(np.unique(new)))
        ts = torch_model(
            share_comps=True, share_start=pre, share_iter=100, comp_thresh=thr
        )
        ts.fit(X, max_iter=pre, verbose=False)
        res["torch_fit_scan"] = int(2 * NW - len(np.unique(ts.comp_list.numpy())))
        res["same_comp_list_ref_torch_numpy_fit"] = [
            bool(np.array_equal(lists["reference_state"], lists["torch_state"])),
            bool(np.array_equal(lists["reference_state"], lists["numpy_state"])),
            bool(np.array_equal(lists["reference_state"], ts.comp_list.numpy())),
        ]
        out[f"thr_{thr}"] = res
    return out


def part_d() -> dict:
    Xs = X[:, :4096]
    out: dict = {}
    first = AMICATorchNG(**_COLLAPSE)
    first.fit(Xs, max_iter=_COLLAPSE["share_start"], verbose=False)
    out["first_scan_merges"] = int(2 * NW - len(np.unique(first.comp_list.numpy())))
    out["gm2_after_first_scan"] = float(first.gm.numpy()[1])
    for k in (1, 2):
        m = AMICATorchNG(**_COLLAPSE)
        m.fit(Xs, max_iter=_COLLAPSE["share_start"] + k, verbose=False)
        out[f"gm2_{k}_after_scan"] = float(m.gm.numpy()[1])
    second = AMICATorchNG(**_COLLAPSE)
    second.fit(
        Xs, max_iter=_COLLAPSE["share_start"] + _COLLAPSE["share_iter"], verbose=False
    )
    out["second_scan_stop"] = str(second.stop_reason)
    out["merges_after_second_scan"] = int(
        2 * NW - len(np.unique(second.comp_list.numpy()))
    )
    out["gm2_after_second_scan"] = float(second.gm.numpy()[1])
    long = AMICATorchNG(**_COLLAPSE)
    long.fit(Xs, max_iter=40, verbose=False)
    out["long_fit_stop"] = str(long.stop_reason)
    out["long_fit_iterations_recorded"] = len(long.ll_history)
    out["long_fit_last_iteration_index"] = int(long.iteration)
    return out


def main() -> None:
    out = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        rep = {"a": part_a(), "b": part_b(), "c": part_c(Path(td)), "d": part_d()}
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
