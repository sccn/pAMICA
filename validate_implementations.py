#!/usr/bin/env python
"""
Validation script to compare Fortran binary and PyTorch AMICA implementations.

This script ensures both implementations produce similar results when starting
from the same initialization and random seed.
"""

import numpy as np
import json
import os
import re
import subprocess
import torch
import inspect
from pathlib import Path
import shutil
import argparse
from typing import Dict, Tuple, Optional

from pamica import AMICA
from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

# Constructor kwargs accepted by AMICATorchNG, used to filter the sample
# params.json down to what the natural-gradient backend understands.
_NG_PARAMS = set(inspect.signature(AMICATorchNG).parameters) - {"n_channels"}

# params.json keys consumed explicitly (as AMICA()/fit() args or run metadata)
# rather than forwarded as AMICATorchNG constructor kwargs. Any key that is
# neither here nor an AMICATorchNG kwarg is a setting the NG backend cannot
# honor; run_pytorch_amica warns about those so a parity comparison against the
# Fortran run can't silently diverge.
_HANDLED_KEYS = {
    "files",
    "outdir",
    "data_dim",
    "field_dim",
    "num_models",
    "num_mix",
    "num_comps",
    "max_iter",
    "max_decs",
    "lrate",
    "do_mean",
    "do_sphere",
    "do_newton",
    "seed",
    "device",
}


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sample_data() -> Tuple[np.ndarray, Dict]:
    """Load the sample EEG data and parameters."""
    sample_dir = Path("pamica/sample_data")
    data_file = sample_dir / "eeglab_data.fdt"
    params_file = sample_dir / "sample_params.json"

    if not data_file.exists():
        raise FileNotFoundError(f"Sample data not found at {data_file}")

    with open(params_file, "r") as f:
        params = json.load(f)

    data = load_eeglab_data(
        str(data_file),
        data_dim=params["data_dim"],
        field_dim=params["field_dim"][0],
        dtype=np.float32,
    )

    return data, params


# params.json spellings that differ from the Fortran keyword of the same setting.
_FORTRAN_ALIASES = {
    "num_mix": "num_mix_comps",
    "share_int": "share_iter",
    "maxrej": "numrej",
}

# params.json keys that configure the Python side only, so having no Fortran
# keyword is expected rather than a dropped setting.
_PYTHON_ONLY_KEYS = {"device", "seed", "outdir", "files", "dtype"}


def fortran_accepted_keys(
    source: Path = Path("pamica/amica15.f90"),
) -> Optional[set]:
    """Keywords the reference binary's parameter parser accepts.

    Read from its own ``case('...')`` arms rather than hardcoded, so this cannot
    drift from the binary being run. Returns ``None`` if the source is
    unavailable, which callers treat as "forward everything and do not warn".
    """
    if not source.exists():
        return None
    return set(re.findall(r"^\s*case\('([^']+)'\)", source.read_text(), re.MULTILINE))


def write_fortran_param_file(
    template_lines,
    dest: Path,
    params: Dict,
    overrides: Optional[Dict] = None,
) -> None:
    """Write ``dest`` with every setting in ``params`` the binary understands.

    The Fortran and Python arms of a parity run must be configured identically.
    Rewriting only a hardcoded handful of keys (as this did before issue #228)
    left the rest at the template's values while the Python side honored them,
    which silently turns a parity comparison into an uncontrolled one.

    Keys absent from the template are appended, since the binary accepts more
    keywords than the shipped ``input.param`` lists. Keys it does not accept are
    reported, so a setting can never be dropped in silence.
    """
    accepted = fortran_accepted_keys()
    wanted, unsupported = {}, []
    for key, value in params.items():
        fortran_key = _FORTRAN_ALIASES.get(key, key)
        if accepted is not None and fortran_key not in accepted:
            if key not in _PYTHON_ONLY_KEYS:
                unsupported.append(key)
            continue
        # Fortran parses logicals as 0/1 integers.
        wanted[fortran_key] = int(value) if isinstance(value, bool) else value

    # Last, so the harness's own paths win over whatever params.json carries.
    wanted.update(overrides or {})

    if unsupported:
        print(
            "WARNING: params with no Fortran keyword are left at the template's "
            f"value for the reference run: {sorted(unsupported)}. The Python run "
            "may honor them, so the comparison would not be controlled."
        )

    written, out = set(), []
    for line in template_lines:
        key = line.split()[0] if line.split() else None
        if key in wanted:
            out.append(f"{key} {wanted[key]}\n")
            written.add(key)
        else:
            out.append(line)
    for key, value in wanted.items():
        if key not in written:
            out.append(f"{key} {value}\n")

    dest.write_text("".join(out))


def run_fortran_amica(
    data: np.ndarray,
    params: Dict,
    output_dir: Path,
    seed: int,
    binary_path: Optional[Path] = None,
) -> Optional[Dict]:
    """Run the AMICA reference binary and collect results.

    ``binary_path`` selects which reference binary to run. It defaults to the
    bundled macOS x86_64 fixture (``pamica/sample_data/amica15mac``); pass the
    cross-platform native-engine binary (see ``--native-engine`` in ``main``) to
    run the real Fortran reference on Linux/Windows/Apple-Silicon instead.
    """
    if binary_path is None:
        binary_path = Path("pamica/sample_data/amica15mac")
    # Resolve to an absolute path now, before the run chdirs into fortran_dir.
    binary_path = Path(binary_path).resolve()
    if not binary_path.exists():
        print(
            f"Warning: Fortran binary not found at {binary_path}. Skipping Fortran comparison."
        )
        return None

    # Create a temporary working directory for Fortran
    fortran_dir = output_dir / "fortran_run"
    fortran_dir.mkdir(exist_ok=True)

    # Copy the sample data file to working directory
    sample_data_file = Path("pamica/sample_data/eeglab_data.fdt")
    working_data_file = fortran_dir / "eeglab_data.fdt"
    shutil.copy(sample_data_file, working_data_file)

    # Copy and modify the parameter file
    sample_param_file = Path("pamica/sample_data/input.param")
    working_param_file = fortran_dir / "input.param"

    with open(sample_param_file, "r") as f:
        param_lines = f.readlines()

    write_fortran_param_file(
        param_lines,
        working_param_file,
        params,
        overrides={"files": "./eeglab_data.fdt", "outdir": "./fortran_output/"},
    )

    # Create output directory
    fortran_output = fortran_dir / "fortran_output"
    fortran_output.mkdir(exist_ok=True)

    # Run Fortran binary
    print("Running Fortran AMICA...")
    original_dir = os.getcwd()
    try:
        # Change to working directory to run
        os.chdir(fortran_dir)

        result = subprocess.run(
            [str(binary_path), "input.param"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        os.chdir(original_dir)

        if result.returncode != 0:
            print(f"Fortran AMICA failed: {result.stderr}")
            print(f"Stdout: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        print("Fortran AMICA timed out")
        return None
    except Exception as e:
        os.chdir(original_dir)
        print(f"Error running Fortran AMICA: {e}")
        return None

    # Parse results
    results = {}

    # Read convergence info from output
    out_file = fortran_output / "out.txt"
    if out_file.exists():
        with open(out_file, "r") as f:
            lines = f.readlines()

        # Extract final LL and iterations from Fortran output format
        for line in lines:
            # Look for iteration lines like: " iter    10 lrate = 0.0500000 LL = -3.4527"
            if line.strip().startswith("iter"):
                parts = line.split()
                if len(parts) >= 6:
                    iter_num = int(parts[1])
                    ll_idx = parts.index("LL") if "LL" in parts else -1
                    if ll_idx > 0 and ll_idx + 2 < len(parts):
                        ll_value = float(parts[ll_idx + 2])
                        results["final_iter"] = iter_num
                        results["final_ll"] = ll_value

    # Load mixing matrix W (unmixing weights)
    W_file = fortran_output / "W"
    if W_file.exists():
        # Fortran writes binary double precision files
        try:
            W = np.fromfile(W_file, dtype=np.float64)  # double precision
            n_sources = params["data_dim"]
            # W can be n_sources x n_sources x n_models
            if len(W) == n_sources * n_sources:
                results["W"] = W.reshape(
                    n_sources, n_sources, order="F"
                )  # Fortran order
            elif len(W) == n_sources * n_sources * params.get("num_models", 1):
                # Multiple models, take first one
                W_all = W.reshape(
                    n_sources, n_sources, params.get("num_models", 1), order="F"
                )
                results["W"] = W_all[:, :, 0]
        except Exception as e:
            print(f"Error loading W: {e}")

    # Load model parameters A (mixing matrix)
    A_file = fortran_output / "A"
    if A_file.exists():
        try:
            A = np.fromfile(A_file, dtype=np.float64)  # double precision
            n_sources = params["data_dim"]
            if len(A) == n_sources * n_sources:
                results["A"] = A.reshape(
                    n_sources, n_sources, order="F"
                )  # Fortran order
            elif len(A) == n_sources * n_sources * params.get("num_models", 1):
                # Multiple models, take first one
                A_all = A.reshape(
                    n_sources, n_sources, params.get("num_models", 1), order="F"
                )
                results["A"] = A_all[:, :, 0]
        except Exception as e:
            print(f"Error loading A: {e}")

    return results


def run_pytorch_amica(
    data: np.ndarray, params: Dict, output_dir: Path, seed: int
) -> Dict:
    """Run the PyTorch natural-gradient EM backend and collect results."""
    print("Running PyTorch AMICA (natural-gradient EM backend)...")

    # Set seed for reproducibility (AMICATorchNG also seeds its own init).
    set_all_seeds(seed)

    # AMICATorchNG cannot do PCA source reduction (n_sources == n_channels), so
    # a Fortran run with num_comps < data_dim would not be an apples-to-apples
    # comparison. Fail loudly rather than silently running full-rank.
    n_comps = params.get("num_comps", params["data_dim"])
    if n_comps != params["data_dim"]:
        raise ValueError(
            f"num_comps={n_comps} != data_dim={params['data_dim']}: "
            "AMICATorchNG does not support PCA source reduction, so the "
            "PyTorch<->Fortran comparison would not be apples-to-apples."
        )

    # Map the sample params.json onto AMICATorchNG constructor kwargs. The
    # backend seeds init, builds the symmetric-ZCA sphere, and starts from an
    # identity-plus-small-perturbation mixing matrix internally, so no manual
    # parameter poking is needed (unlike the removed basic backend). AMICA.fit()
    # handles device selection (and the MPS/float64 -> CPU fallback).
    ng_kwargs = {k: v for k, v in params.items() if k in _NG_PARAMS}
    if "max_decs" in params:  # json name -> AMICATorchNG's `maxdecs`
        ng_kwargs["maxdecs"] = params["max_decs"]
    # lrate/do_mean/do_sphere/do_newton/seed/device are passed explicitly to
    # AMICA()/fit(); drop them from **kwargs to avoid duplicate keyword args.
    for k in ("lrate", "do_mean", "do_sphere", "do_newton", "seed", "device"):
        ng_kwargs.pop(k, None)

    # A parity harness must not silently ignore requested settings: warn about
    # any params.json key the NG backend cannot honor (the Fortran run may use
    # them, so the two runs would then be configured differently).
    ignored = sorted(set(params) - set(ng_kwargs) - _HANDLED_KEYS)
    if ignored:
        print(
            "WARNING: params.json settings with no AMICATorchNG equivalent are "
            f"ignored (NG uses its own behavior): {ignored}. The Fortran run "
            "may honor them, so a parity comparison can differ."
        )

    model = AMICA(
        n_models=params.get("num_models", 1),
        n_mix=params.get("num_mix", 3),
        verbose=True,
    )
    model.fit(
        data,
        max_iter=params.get("max_iter", 100),
        lrate=params.get("lrate", 0.05),
        do_mean=params.get("do_mean", True),
        do_sphere=params.get("do_sphere", True),
        do_newton=params.get("do_newton", False),
        seed=seed,
        **ng_kwargs,
    )

    return {
        # final_ll_ is the LL of the fitted parameters (issue #51 best-iterate
        # safeguard); ll_history_[-1] is the raw last-iteration value, which can
        # sit below the returned iterate after a late overshoot.
        "final_ll": model.final_ll_,
        "final_iter": len(model.ll_history_),
        "W": model.get_unmixing_matrix(0),
        "A": model.get_mixing_matrix(0),
        "ll_history": model.ll_history_,
    }


def _amari_index(gain: np.ndarray) -> float:
    n = gain.shape[0]
    if n < 2:
        raise ValueError("amari_distance: matrices must be at least 2x2")
    abs_gain = np.abs(gain)
    row_max = abs_gain.max(axis=1)
    col_max = abs_gain.max(axis=0)
    if np.any(row_max == 0) or np.any(col_max == 0):
        raise ValueError("amari_distance: a row or column is all-zero")
    row_term = (abs_gain.sum(axis=1) / row_max - 1).sum()
    col_term = (abs_gain.sum(axis=0) / col_max - 1).sum()
    return (row_term + col_term) / (2 * n * (n - 1))


def amari_distance(Wa: np.ndarray, Wb: np.ndarray) -> float:
    """Amari distance between two square unmixing matrices (Amari et al. 1996).

    Permutation- and scale-invariant by construction, so unlike the
    Hungarian-matched correlation above it needs no assignment step: 0 for a
    perfect match up to row permutation/scaling, increasing with disagreement.
    The raw index is not symmetric under a Wa/Wb swap, so this averages both
    directions to give an actual (symmetric) distance.
    """
    forward = _amari_index(Wa @ np.linalg.pinv(Wb))
    backward = _amari_index(Wb @ np.linalg.pinv(Wa))
    return float((forward + backward) / 2)


def compare_results(fortran_results: Optional[Dict], pytorch_results: Dict) -> Dict:
    """Compare results from both implementations."""
    comparison = {}

    if fortran_results is None:
        print("\nNo Fortran results to compare. Showing PyTorch results only:")
        print(f"  Final LL: {pytorch_results['final_ll']:.6f}")
        print(f"  Iterations: {pytorch_results['final_iter']}")
        return {"status": "fortran_unavailable"}

    # Compare log-likelihood
    fortran_ll = fortran_results.get("final_ll", 0)
    pytorch_ll = pytorch_results["final_ll"]

    # Note: There may be a scaling difference
    ll_ratio = pytorch_ll / fortran_ll if fortran_ll != 0 else float("inf")
    comparison["ll_ratio"] = ll_ratio
    comparison["ll_difference"] = abs(pytorch_ll - fortran_ll)

    # Compare convergence speed
    comparison["fortran_iters"] = fortran_results.get("final_iter", 0)
    comparison["pytorch_iters"] = pytorch_results["final_iter"]

    # Compare mixing/unmixing matrices (if available)
    if "W" in fortran_results and "W" in pytorch_results:
        W_fortran = fortran_results["W"]
        W_pytorch = pytorch_results["W"]

        # Compute correlation between components
        if W_fortran.shape == W_pytorch.shape:
            # Normalize rows (components)
            W_fortran_norm = W_fortran / (
                np.linalg.norm(W_fortran, axis=1, keepdims=True) + 1e-10
            )
            W_pytorch_norm = W_pytorch / (
                np.linalg.norm(W_pytorch, axis=1, keepdims=True) + 1e-10
            )

            # Compute absolute correlations (components may have sign flip and permutation)
            correlations = np.abs(W_fortran_norm @ W_pytorch_norm.T)

            # Find best matching components using Hungarian algorithm for optimal assignment
            from scipy.optimize import linear_sum_assignment

            # Convert to cost matrix (maximize correlation = minimize negative correlation)
            cost_matrix = 1 - correlations
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Get the correlations for best matches
            best_correlations = correlations[row_ind, col_ind]
            comparison["component_correlations"] = best_correlations
            comparison["mean_correlation"] = best_correlations.mean()
            comparison["min_correlation"] = best_correlations.min()
            comparison["max_correlation"] = best_correlations.max()
            comparison["std_correlation"] = best_correlations.std()

            # Store permutation for component matching
            comparison["component_permutation"] = col_ind

            # Also compare A matrices if available
            if "A" in fortran_results and "A" in pytorch_results:
                A_fortran = fortran_results["A"]
                A_pytorch = pytorch_results["A"]

                if A_fortran.shape == A_pytorch.shape:
                    # Apply same permutation to PyTorch A for fair comparison
                    A_pytorch_perm = A_pytorch[:, col_ind]

                    # Compute reconstruction error
                    A_diff = np.linalg.norm(A_fortran - A_pytorch_perm, "fro")
                    A_norm = np.linalg.norm(A_fortran, "fro")
                    comparison["mixing_matrix_error"] = (
                        A_diff / A_norm if A_norm > 0 else float("inf")
                    )

    return comparison


def print_comparison_report(
    comparison: Dict, fortran_results: Optional[Dict], pytorch_results: Dict
):
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT: Fortran vs PyTorch AMICA")
    print("=" * 70)

    if comparison.get("status") == "fortran_unavailable":
        print("Fortran binary not available for comparison.")
        return

    # Log-likelihood comparison
    print("\n1. LOG-LIKELIHOOD COMPARISON:")
    print("-" * 40)
    if fortran_results:
        print(f"  Fortran Final LL: {fortran_results.get('final_ll', 'N/A')}")
    print(f"  PyTorch Final LL: {pytorch_results['final_ll']:.6f}")
    if "ll_ratio" in comparison:
        print(f"  LL Ratio (PyTorch/Fortran): {comparison['ll_ratio']:.4f}")
        print(f"  LL Absolute Difference: {comparison['ll_difference']:.6f}")

    # Convergence comparison
    print("\n2. CONVERGENCE COMPARISON:")
    print("-" * 40)
    if "fortran_iters" in comparison:
        print(f"  Fortran Iterations: {comparison['fortran_iters']}")
    print(f"  PyTorch Iterations: {comparison['pytorch_iters']}")

    # Component correlation
    if "mean_correlation" in comparison:
        print("\n3. COMPONENT CORRELATION:")
        print("-" * 40)
        print(f"  Mean Correlation: {comparison['mean_correlation']:.4f}")
        print(f"  Min Correlation: {comparison['min_correlation']:.4f}")
        print(f"  Max Correlation: {comparison['max_correlation']:.4f}")
        print(f"  Std Correlation: {comparison['std_correlation']:.4f}")

        if comparison["mean_correlation"] > 0.9:
            print("  ✓ Components are highly correlated (>0.9)")
        elif comparison["mean_correlation"] > 0.7:
            print("  ⚠ Components are moderately correlated (0.7-0.9)")
        else:
            print("  ✗ Components have low correlation (<0.7)")

        if "mixing_matrix_error" in comparison:
            print(
                f"\n  Mixing Matrix Relative Error: {comparison['mixing_matrix_error']:.4f}"
            )
            if comparison["mixing_matrix_error"] < 0.1:
                print("  ✓ Mixing matrices are very similar (<10% error)")
            elif comparison["mixing_matrix_error"] < 0.3:
                print("  ⚠ Mixing matrices are moderately similar (10-30% error)")
            else:
                print("  ✗ Mixing matrices differ significantly (>30% error)")

    # Overall assessment
    print("\n4. OVERALL ASSESSMENT:")
    print("-" * 40)

    issues = []
    if "ll_ratio" in comparison:
        if abs(comparison["ll_ratio"] - 1.0) > 0.1:
            issues.append("Log-likelihood values differ significantly")

    if "mean_correlation" in comparison:
        if comparison["mean_correlation"] < 0.9:
            issues.append("Component correlations are below threshold")

    if issues:
        print("  Issues detected:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Note: Differences may be due to:")
        print("    - Different numerical precision")
        print("    - Different optimization paths")
        print("    - Scaling differences in LL computation")
    else:
        print("  ✓ Implementations produce comparable results")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate AMICA implementations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--skip-fortran", action="store_true", help="Skip Fortran comparison"
    )
    parser.add_argument(
        "--native-engine",
        action="store_true",
        help="Run the reference through pamica.native: resolve the binary via "
        "PAMICA_NATIVE_BINARY or the cached/downloaded cross-platform release "
        "binary, instead of the bundled macOS-only sample_data/amica15mac. This "
        "is how to run the real Fortran reference on Linux/Windows/Apple Silicon.",
    )
    parser.add_argument(
        "--fortran-binary",
        type=str,
        default=None,
        help="Explicit path to the AMICA reference binary (overrides the bundled "
        "fixture; ignored when --native-engine is given).",
    )

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("validation_output")

    output_dir.mkdir(exist_ok=True)

    print(f"Validation with seed={args.seed}, max_iter={args.max_iter}")
    print(f"Output directory: {output_dir}")

    try:
        # Load data
        data, params = load_sample_data()
        print(f"Loaded data: {data.shape}")

        # Update parameters
        params["max_iter"] = args.max_iter

        # Resolve which reference binary to run: the bundled macOS fixture by
        # default, or the cross-platform native-engine binary on request.
        fortran_binary = None
        if not args.skip_fortran and args.native_engine:
            from pamica.native import resolver

            try:
                fortran_binary = resolver.resolve()
                print(f"Reference binary (native engine): {fortran_binary}")
            except Exception as e:
                print(
                    f"Could not resolve a native AMICA binary ({e}); "
                    "skipping Fortran comparison."
                )
                args.skip_fortran = True
        elif args.fortran_binary:
            fortran_binary = Path(args.fortran_binary)

        # Run Fortran implementation
        fortran_results = None
        if not args.skip_fortran:
            fortran_results = run_fortran_amica(
                data, params, output_dir, args.seed, binary_path=fortran_binary
            )

        # Run PyTorch implementation
        pytorch_results = run_pytorch_amica(data, params, output_dir, args.seed)

        # Compare results
        comparison = compare_results(fortran_results, pytorch_results)

        # Print report
        print_comparison_report(comparison, fortran_results, pytorch_results)

        # Save comparison to file
        report_file = output_dir / "validation_report.txt"
        with open(report_file, "w") as f:
            import sys

            original_stdout = sys.stdout
            sys.stdout = f
            print_comparison_report(comparison, fortran_results, pytorch_results)
            sys.stdout = original_stdout

        print(f"\nReport saved to: {report_file}")

    except Exception as e:
        print(f"\nERROR during validation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
