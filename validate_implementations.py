#!/usr/bin/env python
"""
Validation script to compare the Fortran reference binary with pamica's backends.

Every backend (``--backend torch``, ``numpy`` or ``mlx``; a comma-separated
list; or ``all``) runs the bundled sample EEG with the same parameters as the
reference run (``sample_params.json``, read through the shared canonical
reader) and is compared against it with the same metrics: final
log-likelihood, Hungarian-matched component correlation, mixing-matrix error
and Amari distance. The default, ``torch``, prints exactly the report this
script has always printed.
"""

import numpy as np
import os
import re
import subprocess
import sys
import time
import torch
import inspect
from pathlib import Path
import shutil
import argparse
from contextlib import redirect_stdout
from typing import Callable, Dict, List, Tuple, Optional

from pamica import AMICA, AMICA_NumPy
from pamica.fortran_params import (
    JSON_ALIAS_TO_CANONICAL,
    PAMICA_KEY_TO_FORTRAN_KEY,
    read_params_file,
)
from pamica.numpy_impl.core import _CANONICAL_TO_NUMPY_KEY, _CONSUMED_KEYS
from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

# Backends the harness can compare against the reference, in report order, with
# the label each one's report uses and the precision it computes in.
BACKENDS = ("torch", "numpy", "mlx")
_BACKEND_LABELS = {"torch": "PyTorch", "numpy": "NumPy", "mlx": "MLX"}
_BACKEND_PRECISION = {"torch": "float64", "numpy": "float64", "mlx": "float32"}

# Constructor kwargs accepted by AMICATorchNG, used to filter the sample
# params.json down to what the natural-gradient backend understands.
_NG_PARAMS = set(inspect.signature(AMICATorchNG).parameters) - {"n_channels"}

# Data-location keys: they tell the reference binary where the data live, but
# every Python run here is handed the loaded array directly.
_DATA_LOCATION_KEYS = {"files", "outdir", "data_dim", "field_dim"}

# params.json keys consumed explicitly (as AMICA()/fit() args or run metadata)
# rather than forwarded as AMICATorchNG constructor kwargs. Any key that is
# neither here nor a key the backend run applies is a setting that backend
# cannot honor; each run_*_amica warns about those (_warn_ignored) so a parity
# comparison against the Fortran run can't silently diverge. `max_decs` is not
# listed: load_sample_data
# now reads params.json through read_params_file (issue #304), which already
# renames it (and min_grad_norm/share_int) to the canonical maxdecs/min_nd/
# share_iter spelling -- the same spelling _NG_PARAMS filters on -- so those
# three land in ng_kwargs automatically instead of needing a special case here.
_HANDLED_KEYS = {
    "files",
    "outdir",
    "data_dim",
    "field_dim",
    "num_models",
    "num_mix",
    "num_comps",
    "max_iter",
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

    # Issue #304: read through the shared canonical reader rather than a raw
    # json.load, so params.json's own alias spellings (min_grad_norm/max_decs/
    # share_int) are translated to the canonical min_nd/maxdecs/share_iter
    # names -- the same spelling _NG_PARAMS filters run_pytorch_amica's
    # ng_kwargs on -- instead of silently landing in the "ignored" warning.
    params = read_params_file(params_file)

    data = load_eeglab_data(
        str(data_file),
        data_dim=params["data_dim"],
        field_dim=params["field_dim"][0],
        dtype=np.float32,
    )

    return data, params


# params.json/pamica-canonical spellings that differ from the Fortran keyword
# of the same setting, keyed both ways so write_fortran_param_file accepts
# either spelling with no hand-maintained entry (issue #304): the canonical
# ("pamica-key") entries come straight from PAMICA_KEY_TO_FORTRAN_KEY (the
# single source of truth for pamica-key -> Fortran-keyword, itself the
# inverse of FORTRAN_TO_PAMICA_KEY's renames); the JSON-schema-keyed entries
# (e.g. "share_int") are composed by chaining each JSON_ALIAS_TO_CANONICAL
# alias through the same table (JSON alias -> canonical -> Fortran keyword;
# a canonical that PAMICA_KEY_TO_FORTRAN_KEY has no rename for, like
# share_iter, falls back to its own spelling, which already matches Fortran).
_FORTRAN_ALIASES = {
    json_alias: PAMICA_KEY_TO_FORTRAN_KEY.get(canonical, canonical)
    for json_alias, canonical in JSON_ALIAS_TO_CANONICAL.items()
}
_FORTRAN_ALIASES.update(PAMICA_KEY_TO_FORTRAN_KEY)

# params.json keys that configure the Python side only, so having no Fortran
# keyword is expected rather than a dropped setting.
_PYTHON_ONLY_KEYS = {"device", "outdir", "files", "dtype"}

# Keywords the tracked amica15.f90 carries but sccn/amica master does not, so
# they are absent when this parses an upstream source (issue #228). The binary's
# parser has no `case default`, so an unknown keyword is silently ignored --
# passing `seed` to an unpatched legacy binary is harmless, it just has no effect.
_PAMICA_EXTRA_KEYS = {"seed"}


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
    found = set(re.findall(r"^\s*case\('([^']+)'\)", source.read_text(), re.MULTILINE))
    return found | _PAMICA_EXTRA_KEYS


def _fortran_value(value) -> str:
    """Render a params.json value the way the Fortran parser reads it.

    Logicals are 0/1 integers, and the per-file lists (``files``, ``field_dim``)
    are whitespace-separated -- writing Python's ``repr`` for those puts brackets
    in the file and the parser aborts.
    """
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return " ".join(_fortran_value(v) for v in value)
    return str(value)


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
        wanted[fortran_key] = _fortran_value(value)

    # Last, so the harness's own paths win over whatever params.json carries.
    wanted.update({k: _fortran_value(v) for k, v in (overrides or {}).items()})

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


LEGACY_BINARY = Path("pamica/sample_data/amica15mac")


def default_reference_binary(*, download: bool = True) -> Path:
    """The reference binary to compare against, preferring the native engine.

    The native engine is built from a source carrying the ``seed`` option
    (sccn/amica PR #54), so its runs are reproducible; the bundled
    ``amica15mac`` fixture predates it and re-randomizes its initialization on
    every run, which makes any comparison against it a comparison with a random
    draw (issue #228). Falls back to the fixture, loudly, when the native engine
    cannot be resolved.
    """
    try:
        from pamica.native import resolver

        return resolver.resolve(download=download)
    except Exception as exc:  # network, unsupported platform, missing cache
        print(
            f"WARNING: native engine unavailable ({exc}); falling back to "
            f"{LEGACY_BINARY}, which cannot be seeded. Reference runs will not "
            "be reproducible and single-run comparisons against them are not "
            "controlled (issue #228)."
        )
        return LEGACY_BINARY


def run_fortran_amica(
    data: np.ndarray,
    params: Dict,
    output_dir: Path,
    seed: int,
    binary_path: Optional[Path] = None,
) -> Optional[Dict]:
    """Run the AMICA reference binary and collect results.

    ``binary_path`` selects which reference binary to run. It defaults to the
    native engine (see :func:`default_reference_binary`), which is seedable and
    therefore reproducible; pass ``LEGACY_BINARY`` for the bundled macOS x86_64
    fixture instead.
    """
    if binary_path is None:
        binary_path = default_reference_binary()
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

    # seed and max_threads are pinned, not taken from params: an unseeded or
    # multi-threaded reference run is not reproducible (issue #228), which makes
    # any comparison against it a comparison with a random draw.
    write_fortran_param_file(
        param_lines,
        working_param_file,
        params,
        overrides={
            "files": "./eeglab_data.fdt",
            "outdir": "./fortran_output/",
            "seed": int(seed),
            "max_threads": 1,
        },
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

        start = time.perf_counter()
        result = subprocess.run(
            [str(binary_path), "input.param"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        runtime_s = time.perf_counter() - start

        os.chdir(original_dir)

        if result.returncode != 0:
            print(f"Fortran AMICA failed: {result.stderr}")
            print(f"Stdout: {result.stdout}")
            return None

        # A binary predating sccn/amica PR #54 has no `seed` case and its parser
        # has no `case default`, so it ignores the keyword in silence and
        # re-randomizes instead. Detect that rather than reporting a comparison
        # against a random draw as if it were controlled (issue #228).
        if "seed =" not in result.stdout:
            print(
                f"WARNING: {binary_path} did not acknowledge the seed, so it "
                "predates the seedable build. Its initialization is random per "
                "run and this comparison is not controlled (issue #228)."
            )

    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        print("Fortran AMICA timed out")
        return None
    except Exception as e:
        os.chdir(original_dir)
        print(f"Error running Fortran AMICA: {e}")
        return None

    # Parse results. runtime_s is the binary's wall-clock (single-threaded, per
    # the max_threads pin above), process start-up included.
    results = {"runtime_s": runtime_s}

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


def _check_full_rank_comps(params: Dict, label: str) -> None:
    """Refuse a reference run with ``num_comps != data_dim``.

    ``num_comps`` is a reference-binary keyword with no pamica counterpart:
    every pamica backend sizes its model from the data rank and
    ``pcakeep``/``pcadb`` instead. A Fortran run configured with a different
    ``num_comps`` would therefore fit a differently sized model than the
    Python run, so fail loudly rather than report an uncontrolled comparison.
    """
    n_comps = params.get("num_comps", params["data_dim"])
    if n_comps != params["data_dim"]:
        raise ValueError(
            f"num_comps={n_comps} != data_dim={params['data_dim']}: pamica "
            "has no num_comps (its backends size the model from the data rank "
            f"and pcakeep/pcadb), so the {label}<->Fortran comparison would "
            "not be apples-to-apples."
        )


def _warn_ignored(params: Dict, applied: set, backend_class: str, short: str) -> None:
    """Name every params.json setting a backend run cannot honor.

    A parity harness must not silently ignore requested settings: the
    Fortran run may use them, so the two runs would then be configured
    differently.
    """
    ignored = sorted(set(params) - applied - _HANDLED_KEYS)
    if ignored:
        print(
            f"WARNING: params.json settings with no {backend_class} equivalent are "
            f"ignored ({short} uses its own behavior): {ignored}. The Fortran run "
            "may honor them, so a parity comparison can differ."
        )


def run_pytorch_amica(
    data: np.ndarray, params: Dict, output_dir: Path, seed: int
) -> Dict:
    """Run the PyTorch natural-gradient EM backend and collect results."""
    print("Running PyTorch AMICA (natural-gradient EM backend)...")

    # Set seed for reproducibility (AMICATorchNG also seeds its own init).
    set_all_seeds(seed)

    _check_full_rank_comps(params, "PyTorch")

    # Map the sample params.json onto AMICATorchNG constructor kwargs. The
    # backend seeds init, builds the symmetric-ZCA sphere, and starts from an
    # identity-plus-small-perturbation mixing matrix internally, so no manual
    # parameter poking is needed (unlike the removed basic backend). AMICA.fit()
    # handles device selection (and the MPS/float64 -> CPU fallback). `params`
    # is already canonical-keyed (read_params_file, issue #304), so maxdecs/
    # min_nd/share_iter land here via the plain _NG_PARAMS filter -- no
    # special case needed for the json schema's max_decs/min_grad_norm/
    # share_int spellings.
    ng_kwargs = {k: v for k, v in params.items() if k in _NG_PARAMS}
    # lrate/do_mean/do_sphere/do_newton/seed/device are passed explicitly to
    # AMICA()/fit(); drop them from **kwargs to avoid duplicate keyword args.
    for k in ("lrate", "do_mean", "do_sphere", "do_newton", "seed", "device"):
        ng_kwargs.pop(k, None)

    _warn_ignored(params, set(ng_kwargs), "AMICATorchNG", "NG")

    model = AMICA(
        n_models=params.get("num_models", 1),
        n_mix=params.get("num_mix", 3),
        verbose=True,
    )
    start = time.perf_counter()
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
    runtime_s = time.perf_counter() - start

    return {
        # final_ll_ is the LL of the fitted parameters (issue #51 best-iterate
        # safeguard); ll_history_[-1] is the raw last-iteration value, which can
        # sit below the returned iterate after a late overshoot.
        "final_ll": model.final_ll_,
        "final_iter": len(model.ll_history_),
        "W": model.get_unmixing_matrix(0),
        "A": model.get_mixing_matrix(0),
        "ll_history": model.ll_history_,
        "runtime_s": runtime_s,
    }


def run_numpy_amica(
    data: np.ndarray, params: Dict, output_dir: Path, seed: int
) -> Dict:
    """Run the legacy NumPy backend (``AMICA_NumPy``) and collect results.

    ``params`` is canonical-keyed (``read_params_file``, issue #304); the
    three settings this backend spells differently (``min_nd``, ``maxdecs``,
    ``share_iter``) go through the backend's own ``_CANONICAL_TO_NUMPY_KEY``
    table rather than a copy of it. The backend computes in float64, so the
    float32 sample is widened first, as its own parity test does.
    """
    print("Running NumPy AMICA (legacy reference backend)...")

    _check_full_rank_comps(params, "NumPy")

    numpy_kwargs, applied = {}, set()
    for key, value in params.items():
        numpy_key = _CANONICAL_TO_NUMPY_KEY.get(key, key)
        if key in _DATA_LOCATION_KEYS or numpy_key not in _CONSUMED_KEYS:
            continue
        numpy_kwargs[numpy_key] = value
        applied.add(key)
    _warn_ignored(params, applied, "AMICA_NumPy", "NumPy")

    # The backend writes out.txt at construction and checkpoints every
    # writestep, so point it at this run's own directory (its default,
    # ./output, would land in the working directory).
    numpy_kwargs["outdir"] = str(output_dir / "numpy_run")
    numpy_kwargs["seed"] = seed
    model = AMICA_NumPy(**numpy_kwargs)

    start = time.perf_counter()
    model.fit(data.astype(np.float64))
    runtime_s = time.perf_counter() - start
    if not model.converged:
        raise RuntimeError(
            f"AMICA_NumPy fit ended degenerate (stop_reason={model.stop_reason!r})"
        )

    W = model.get_weights()
    assert model.A is not None and model.comp_list is not None
    return {
        # ll[-1] is this backend's final_ll_ (it has no keep_best restore).
        "final_ll": float(model.ll[-1]),
        "final_iter": len(model.ll),
        "W": W,
        # get_weights() is the stored W transposed (issue #24 convention), so
        # the true mixing is the stored A's model-0 columns transposed, the
        # same composition AMICATorchNG.get_mixing_matrix returns.
        "A": model.A[:, model.comp_list[:, 0]].T,
        "ll_history": list(model.ll),
        "runtime_s": runtime_s,
    }


def mlx_unavailable_reason() -> Optional[str]:
    """Why the MLX backend cannot run here, or ``None`` when it can."""
    try:
        import pamica.mlx_impl  # noqa: F401
    except ImportError as exc:
        return (
            f"--backend mlx needs the optional MLX backend, which failed to "
            f"import ({exc}). MLX runs on Apple Silicon only; install it with "
            "`uv sync --extra mlx`."
        )
    return None


def run_mlx_amica(data: np.ndarray, params: Dict, output_dir: Path, seed: int) -> Dict:
    """Run the MLX backend (``AMICAMLXNG``, float32 on the Apple GPU) and
    collect results.

    Its constructor carries ``AMICATorchNG``'s parameter names, so the
    canonical params map onto it the same way they map onto the PyTorch run.
    """
    from pamica.mlx_impl import AMICAMLXNG

    print("Running MLX AMICA (Apple-GPU float32 backend)...")

    _check_full_rank_comps(params, "MLX")

    mlx_params = set(inspect.signature(AMICAMLXNG).parameters) - {
        "n_channels",
        "n_models",
        "n_mix",
        "seed",
    }
    mlx_kwargs = {k: v for k, v in params.items() if k in mlx_params}
    _warn_ignored(params, set(mlx_kwargs), "AMICAMLXNG", "MLX")

    model = AMICAMLXNG(
        n_channels=data.shape[0],
        n_models=params.get("num_models", 1),
        n_mix=params.get("num_mix", 3),
        seed=seed,
        **mlx_kwargs,
    )
    start = time.perf_counter()
    model.fit(data.astype(np.float32), max_iter=params.get("max_iter", 100))
    runtime_s = time.perf_counter() - start
    if model.stop_reason in AMICAMLXNG._DEGENERATE_STOP_REASONS:
        raise RuntimeError(
            f"AMICAMLXNG fit ended degenerate (stop_reason={model.stop_reason!r})"
        )

    return {
        "final_ll": model.final_ll_,
        "final_iter": len(model.ll_history),
        # float64 copies, so the comparison arithmetic below runs at the same
        # precision for every backend.
        "W": model.get_unmixing_matrix(0).astype(np.float64),
        "A": model.get_mixing_matrix(0).astype(np.float64),
        "ll_history": list(model.ll_history),
        "runtime_s": runtime_s,
    }


_RUNNERS: Dict[str, Callable[[np.ndarray, Dict, Path, int], Dict]] = {
    "torch": run_pytorch_amica,
    "numpy": run_numpy_amica,
    "mlx": run_mlx_amica,
}


def run_backend(
    name: str, data: np.ndarray, params: Dict, output_dir: Path, seed: int
) -> Dict:
    """Run backend ``name`` (one of :data:`BACKENDS`) on ``data`` with the
    harness's ``params`` and return its results dict (``final_ll``,
    ``final_iter``, ``W``, ``A``, ``ll_history``, ``runtime_s``)."""
    if name not in _RUNNERS:
        raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
    return _RUNNERS[name](data, params, output_dir, seed)


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


def compare_results(
    fortran_results: Optional[Dict], backend_results: Dict, label: str = "PyTorch"
) -> Dict:
    """Compare a backend's results (labeled ``label`` in the printout) with
    the reference run's."""
    comparison = {}

    if fortran_results is None:
        print(f"\nNo Fortran results to compare. Showing {label} results only:")
        print(f"  Final LL: {backend_results['final_ll']:.6f}")
        print(f"  Iterations: {backend_results['final_iter']}")
        return {"status": "fortran_unavailable"}

    # Compare log-likelihood
    fortran_ll = fortran_results.get("final_ll", 0)
    backend_ll = backend_results["final_ll"]

    # Note: There may be a scaling difference
    ll_ratio = backend_ll / fortran_ll if fortran_ll != 0 else float("inf")
    comparison["ll_ratio"] = ll_ratio
    comparison["ll_difference"] = abs(backend_ll - fortran_ll)

    # Compare convergence speed
    comparison["fortran_iters"] = fortran_results.get("final_iter", 0)
    comparison["backend_iters"] = backend_results["final_iter"]

    # Compare mixing/unmixing matrices (if available)
    if "W" in fortran_results and "W" in backend_results:
        W_fortran = fortran_results["W"]
        W_backend = backend_results["W"]

        # Compute correlation between components
        if W_fortran.shape == W_backend.shape:
            # Normalize rows (components)
            W_fortran_norm = W_fortran / (
                np.linalg.norm(W_fortran, axis=1, keepdims=True) + 1e-10
            )
            W_backend_norm = W_backend / (
                np.linalg.norm(W_backend, axis=1, keepdims=True) + 1e-10
            )

            # Compute absolute correlations (components may have sign flip and permutation)
            correlations = np.abs(W_fortran_norm @ W_backend_norm.T)

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

            # The assignment-free second metric (see amari_distance).
            comparison["amari_distance"] = amari_distance(W_fortran, W_backend)

            # Also compare A matrices if available
            if "A" in fortran_results and "A" in backend_results:
                A_fortran = fortran_results["A"]
                A_backend = backend_results["A"]

                if A_fortran.shape == A_backend.shape:
                    # Apply same permutation to the backend's A for fair comparison
                    A_backend_perm = A_backend[:, col_ind]

                    # Compute reconstruction error
                    A_diff = np.linalg.norm(A_fortran - A_backend_perm, "fro")
                    A_norm = np.linalg.norm(A_fortran, "fro")
                    comparison["mixing_matrix_error"] = (
                        A_diff / A_norm if A_norm > 0 else float("inf")
                    )

    return comparison


def print_comparison_report(
    comparison: Dict,
    fortran_results: Optional[Dict],
    backend_results: Dict,
    label: str = "PyTorch",
):
    """Print a formatted comparison report for the backend labeled ``label``."""
    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT: Fortran vs {label} AMICA")
    print("=" * 70)

    if comparison.get("status") == "fortran_unavailable":
        print("Fortran binary not available for comparison.")
        return

    # Log-likelihood comparison
    print("\n1. LOG-LIKELIHOOD COMPARISON:")
    print("-" * 40)
    if fortran_results:
        print(f"  Fortran Final LL: {fortran_results.get('final_ll', 'N/A')}")
    print(f"  {label} Final LL: {backend_results['final_ll']:.6f}")
    if "ll_ratio" in comparison:
        print(f"  LL Ratio ({label}/Fortran): {comparison['ll_ratio']:.4f}")
        print(f"  LL Absolute Difference: {comparison['ll_difference']:.6f}")

    # Convergence comparison
    print("\n2. CONVERGENCE COMPARISON:")
    print("-" * 40)
    if "fortran_iters" in comparison:
        print(f"  Fortran Iterations: {comparison['fortran_iters']}")
    print(f"  {label} Iterations: {comparison['backend_iters']}")

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


def _fmt(value, spec: str) -> str:
    """``value`` formatted with ``spec``, or ``n/a`` when it is missing."""
    return "n/a" if value is None else format(value, spec)


def format_parity_summary(
    fortran_results: Optional[Dict], rows: List[Tuple[str, Dict, Dict]]
) -> str:
    """One Markdown table row per backend run, plus the reference's own row.

    ``rows`` holds ``(backend name, results, comparison)`` triples as produced
    by :func:`run_backend` and :func:`compare_results`. Runtime is each fit's
    wall-clock; the reference's includes process start-up and is
    single-threaded (``max_threads 1``).
    """
    lines = [
        "| backend | precision | iterations | final LL | abs. LL difference "
        "| mean matched corr | min matched corr | Amari distance | runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if fortran_results is not None:
        lines.append(
            "| Fortran (reference) | float64 | "
            f"{_fmt(fortran_results.get('final_iter'), 'd')} | "
            f"{_fmt(fortran_results.get('final_ll'), '.6f')} | | | | | "
            f"{_fmt(fortran_results.get('runtime_s'), '.1f')} |"
        )
    for name, results, comparison in rows:
        lines.append(
            f"| {_BACKEND_LABELS[name]} | {_BACKEND_PRECISION[name]} | "
            f"{results['final_iter']} | {results['final_ll']:.6f} | "
            f"{_fmt(comparison.get('ll_difference'), '.6f')} | "
            f"{_fmt(comparison.get('mean_correlation'), '.4f')} | "
            f"{_fmt(comparison.get('min_correlation'), '.4f')} | "
            f"{_fmt(comparison.get('amari_distance'), '.4f')} | "
            f"{results['runtime_s']:.1f} |"
        )
    return "\n".join(lines)


def parse_backends(value: str) -> List[str]:
    """Parse ``--backend``: one backend, a comma-separated list, or ``all``.

    Returns the names in the order given, without duplicates.
    """
    if value.strip() == "all":
        return list(BACKENDS)
    names = [name.strip() for name in value.split(",")]
    unknown = [name for name in names if name not in BACKENDS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown backend(s) {unknown}; choose from {', '.join(BACKENDS)}, "
            "a comma-separated list of them, or 'all'"
        )
    return list(dict.fromkeys(names))


def build_parser() -> argparse.ArgumentParser:
    """The harness's command-line interface."""
    parser = argparse.ArgumentParser(description="Validate AMICA implementations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--backend",
        type=parse_backends,
        default=None,
        metavar="{torch,numpy,mlx}[,...]|all",
        help="pamica backend(s) to compare against the reference: torch (the "
        "default), numpy, mlx (Apple Silicon, float32), a comma-separated list "
        "such as torch,mlx, or all. Each backend gets its own report; passing "
        "this flag also prints and saves a one-row-per-backend parity summary.",
    )
    parser.add_argument(
        "--skip-fortran", action="store_true", help="Skip Fortran comparison"
    )
    parser.add_argument(
        "--native-engine",
        action="store_true",
        help="Run the reference through pamica.native: resolve the binary via "
        "PAMICA_NATIVE_BINARY or the cached/downloaded cross-platform release "
        "binary. This is also what the default (no reference flag) resolves "
        "to, falling back to the bundled macOS x86_64 sample_data/amica15mac "
        "only when no native binary can be resolved; with this flag that "
        "failure skips the Fortran comparison instead.",
    )
    parser.add_argument(
        "--fortran-binary",
        type=str,
        default=None,
        help="Explicit path to the AMICA reference binary (overrides the "
        "native-engine resolution; ignored when --native-engine is given).",
    )
    return parser


def _report_file(output_dir: Path, backend: str) -> Path:
    """Where a backend's report is saved. The PyTorch report keeps the name
    the harness has always used."""
    if backend == "torch":
        return output_dir / "validation_report.txt"
    return output_dir / f"validation_report_{backend}.txt"


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # The default run (no --backend) is exactly the historical torch-only run.
    backends = args.backend if args.backend is not None else ["torch"]
    if "mlx" in backends:
        reason = mlx_unavailable_reason()
        if reason is not None:
            print(f"ERROR: {reason}", file=sys.stderr)
            return 2

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

        # Resolve which reference binary to run: the native engine by default
        # (see default_reference_binary), or an explicit binary on request.
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

        # Run Fortran implementation (once, shared by every backend below)
        fortran_results = None
        if not args.skip_fortran:
            fortran_results = run_fortran_amica(
                data, params, output_dir, args.seed, binary_path=fortran_binary
            )

        rows = []
        for backend in backends:
            label = _BACKEND_LABELS[backend]
            results = run_backend(backend, data, params, output_dir, args.seed)

            # Compare results
            comparison = compare_results(fortran_results, results, label)

            # Print report
            print_comparison_report(comparison, fortran_results, results, label)

            # Save comparison to file
            report_file = _report_file(output_dir, backend)
            with open(report_file, "w") as f, redirect_stdout(f):
                print_comparison_report(comparison, fortran_results, results, label)

            print(f"\nReport saved to: {report_file}")
            rows.append((backend, results, comparison))

        if args.backend is not None:
            summary = format_parity_summary(fortran_results, rows)
            print(
                "\nPARITY SUMMARY (seed={}, max_iter={}):".format(
                    args.seed, args.max_iter
                )
            )
            print(summary)
            summary_file = output_dir / "parity_summary.md"
            summary_file.write_text(summary + "\n")
            print(f"\nSummary saved to: {summary_file}")

    except Exception as e:
        print(f"\nERROR during validation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
