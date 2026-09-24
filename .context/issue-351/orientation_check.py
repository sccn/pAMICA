"""Check that ``native_oracle.seed_from_torch`` writes a tree's initialization in
the orientation the binary reads (issue #351, for ``harness_gap.py``).

For the tree at ``ROOT``, pamica's seed-42 initialization is written into the
binary's ``load_*`` files twice: as ``seed_from_torch`` writes it (the stored
``A`` transposed) and with the stored ``A`` itself. The binary runs ONE
iteration from each; its iteration-1 log-likelihood is the E-step at the loaded
state, before any update, so the right orientation reproduces pamica's own
iteration-1 log-likelihood to round-off and the wrong one does not.

    uv run python .context/issue-351/orientation_check.py ROOT
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
BINARY = Path.home() / ".cache/pamica/bin/v0.3.3/amica15-macos-arm64"


def main() -> None:
    from _reference_settings import REFERENCE_SETTINGS

    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    import numpy as np

    import pamica

    assert pamica.__file__.startswith(str(root)), pamica.__file__
    from pamica.torch_impl import AMICATorchNG

    spec = importlib.util.spec_from_file_location(
        "native_oracle_351", WORKTREE / "pamica/tests/native_oracle.py"
    )
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    sys.modules["native_oracle_351"] = oracle  # dataclasses need the module
    spec.loader.exec_module(oracle)
    vspec = importlib.util.spec_from_file_location(
        "vi_351", root / "validate_implementations.py"
    )
    assert vspec is not None and vspec.loader is not None
    vi = importlib.util.module_from_spec(vspec)
    vspec.loader.exec_module(vi)

    data, params = vi.load_sample_data()
    X = data.astype(np.float64)
    nw, n = X.shape
    kw = dict(n_channels=nw, n_models=1, n_mix=3, rho0=params["rho0"], seed=42)
    fit = AMICATorchNG(block_size=512, lrate=0.05, device="cpu", **kw)
    fit.fit(X, max_iter=1, verbose=False)
    pamica_ll1 = float(fit.ll_history[0])

    init = AMICATorchNG(device="cpu", **kw)
    init._preprocess(X)
    init._initialize_parameters()
    state = oracle.seed_from_torch(init)
    stored = dataclasses.replace(state, A=np.ascontiguousarray(state.A.T))
    fdt = root / "pamica/sample_data/eeglab_data.fdt"
    out = {}
    for name, st in (("seed_from_torch", state), ("stored A untransposed", stored)):
        with tempfile.TemporaryDirectory() as td:
            ref = oracle.run_seeded_reference(
                st, fdt, Path(td), n_samples=n, max_iter=1, threads=1, binary=BINARY,
                **{**REFERENCE_SETTINGS, "block_size": 512, "do_newton": 1,
                   "use_min_dll": 0, "use_grad_norm": 0},
            )  # fmt: skip
        out[name] = float(ref.LL[0])
    asym = float(np.abs(state.A - state.A.T).max())
    print(f"root {root}")
    print(f"max |A - A.T| of the initial A: {asym:.3e}")
    print(f"pamica iteration-1 LL: {pamica_ll1:.15f}")
    for name, ll in out.items():
        print(f"binary iteration-1 LL, {name}: {ll:.15f} (diff {ll - pamica_ll1:+.3e})")


if __name__ == "__main__":
    main()
