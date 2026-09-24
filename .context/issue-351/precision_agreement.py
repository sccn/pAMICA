"""Component-level float32 agreement at a matched iteration budget (issue #351
item 4c): MLX (float32) against PyTorch (float64) from the same start.

Both backends draw their initialization from numpy's ``RandomState(seed)``, so
with the same seed they start from the same point up to MLX's float32 cast, and
any difference after K iterations is float32 arithmetic. The harness settings
(``sample_params.json`` through ``validate_implementations._run_wrapper_amica``)
at K = 100 (the harness budget) and 200 (the fixture budget), seed 42.

    uv run python .context/issue-351/precision_agreement.py OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "vi_351p", REPO / "validate_implementations.py"
    )
    assert spec is not None and spec.loader is not None
    vi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vi)
    data, params = vi.load_sample_data()
    out = {}
    for k in (100, 200):
        p = dict(params, max_iter=k)
        t = vi._run_wrapper_amica("torch", data, p, 42)
        m = vi._run_wrapper_amica("mlx", data, p, 42)
        cmp = vi.compare_results(t, m, label="MLX")
        out[str(k)] = {
            "torch_final_ll": float(t["final_ll"]),
            "mlx_final_ll": float(m["final_ll"]),
            "abs_ll_difference": float(abs(t["final_ll"] - m["final_ll"])),
            "mean_matched_corr": float(cmp["mean_correlation"]),
            "min_matched_corr": float(cmp["min_correlation"]),
            "amari": float(vi.amari_distance(t["W"], m["W"])),
            "max_abs_W_diff": float(np.abs(t["W"] - m["W"]).max()),
        }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
