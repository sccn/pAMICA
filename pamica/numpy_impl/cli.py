#!/usr/bin/env python3
"""
Command-line interface for AMICA (Adaptive Mixture ICA).

This module provides a command-line interface for running the AMICA algorithm.
It handles:
1. Parameter loading from a JSON or Fortran-format (``input.param``) file
2. Data loading from binary files
3. Model initialization and training
4. Result saving

Example usage:
    python -m pamica.numpy_impl.cli params.json  --outdir results --seed 42  # Use -m flag to run as module

The parameter file may be pamica's own JSON schema or the literal Fortran
``input.param`` text format, auto-detected by content the same way
``AMICA(params_file=...)`` itself does (:func:`pamica.fortran_params.
read_params_file`, issue #304). Either way it must include:
- files: List of binary data files to process
- data_dim: Number of channels/dimensions
- field_dim: Number of samples per channel for each file

Optional parameters can be included in the file:
- num_models: Number of models (default: 1)
- num_mix: Number of mixture components (default: 3)
- max_iter: Maximum iterations (default: 100)
And many others as documented in the AMICA class.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .core import AMICA, _read_numpy_keyed_params
from .data import load_multiple_files


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for AMICA execution.

    Returns
    -------
    args : argparse.Namespace
        Parsed command line arguments with the following attributes:
        - paramfile: Path to a JSON or Fortran-format (input.param) parameter file
        - outdir: Output directory for results (default: 'output')
        - seed: Random seed for reproducibility (optional)
        - verbose: Flag for detailed per-line progress output (disables tqdm progress bar)
    """
    parser = argparse.ArgumentParser(description="AMICA: Adaptive Mixture ICA")

    # Required arguments
    parser.add_argument(
        "paramfile",
        help="Parameter file (pamica JSON schema or Fortran input.param text)",
    )

    # Optional arguments
    parser.add_argument(
        "--outdir", help="Output directory (default: output)", default="output"
    )
    parser.add_argument("--seed", help="Random seed", type=int)
    parser.add_argument(
        "--verbose",
        help="Verbose output with detailed per-line progress (disables tqdm progress bar)",
        action="store_true",
    )

    return parser.parse_args()


def load_params(
    paramfile: str, default_paramfile: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load and validate AMICA parameters from a params file.

    The function loads default parameters from default_paramfile (if provided),
    then updates them with user-provided parameters from paramfile. Both are
    read through :func:`pamica.numpy_impl.core._read_numpy_keyed_params`
    (the same helper ``AMICA(params_file=...)`` itself uses, issue #304), so
    either a pamica JSON-schema file or the literal Fortran ``input.param``
    text format works here -- not just JSON, as before -- and the returned
    keys are already mapped to this backend's own spellings (no separate
    key-mapping table duplicated in this module). ``files`` is returned
    exactly as the params file spells it, still relative to the current
    working directory: this function performs no path resolution, matching
    the reference Fortran binary's own semantics.

    Required parameters in paramfile are:
    - files: List of data files to process
    - data_dim: Number of channels/dimensions
    - field_dim: List of samples per channel for each file

    Parameters
    ----------
    paramfile : str
        Path to a JSON or Fortran-format parameter file with user settings.
    default_paramfile : str, optional
        Path to a JSON or Fortran-format file with default parameters.

    Returns
    -------
    params : dict
        Dictionary of parameters.
    """
    # Load default parameters if provided.
    params = (
        dict(_read_numpy_keyed_params(default_paramfile)) if default_paramfile else {}
    )

    # Update with user parameters.
    params.update(_read_numpy_keyed_params(paramfile))

    # Required parameters
    required = {"files", "data_dim", "field_dim"}

    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    return params


def setup_logging(verbose: bool = False):
    """
    Configure logging for AMICA execution.

    Sets up logging with appropriate level and format. When verbose is True,
    DEBUG level messages are included, otherwise only INFO and above are shown.
    The verbose flag also affects the progress display, switching from tqdm
    progress bar to detailed per-line output.

    Parameters
    ----------
    verbose : bool
        Whether to enable verbose (DEBUG level) logging and detailed per-line progress
    """
    # Get the root logger and remove any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Get the AMICA logger and remove any existing handlers
    logger = logging.getLogger("AMICA")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set level based on verbose flag
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)


def main():
    """
    Main entry point for AMICA command-line execution.

    This function orchestrates the AMICA workflow:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads and validates parameters
    4. Loads data from binary files
    5. Initializes and fits the AMICA model
    6. Saves results to the specified output directory

    The execution can be customized through the parameter file and
    command line arguments.
    """
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger("AMICA")

    # Load parameters
    logger.info(f"Loading parameters from {args.paramfile}")
    params = load_params(args.paramfile)

    # Load data
    logger.info("Loading data files:")
    for f in params["files"]:
        logger.info(f"  {f}")

    data = load_multiple_files(params["files"], params["data_dim"], params["field_dim"])

    # Create output directory
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # Initialize AMICA
    model = AMICA(
        params_file=args.paramfile,
        outdir=str(outdir),
        seed=args.seed,
        use_tqdm=not args.verbose,  # Use tqdm by default, but disable if verbose
        verbose=args.verbose,
    )

    # Fit model
    logger.info("Fitting AMICA model")
    model.fit(data)

    # Report the outcome honestly: on a terminal non-finite likelihood the fit
    # diverged and nothing was written, so do not claim success.
    if getattr(model, "converged", True):
        logger.info(f"Results saved to {outdir}")
    else:
        logger.error(
            "AMICA did not converge (non-finite likelihood); no results were "
            "written to %s. Try a different --seed.",
            outdir,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
