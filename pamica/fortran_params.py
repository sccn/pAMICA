"""Translator from the Fortran ``input.param`` text format to pamica parameters
(issue #132).

The reference Fortran binary (``amica15.f90``) and ``AMICA.from_params_file``
(``amica.py``) both configure a run from a parameter file, but they use two
different formats and, for a handful of settings, two different spellings of
the same keyword. This module parses the literal Fortran text format --
whitespace-separated ``key value`` lines, ``#`` full-line comments, ints/
floats/strings, boolean flags as ``0``/``1`` -- into a dict shaped exactly
like ``sample_data/sample_params.json`` (the format ``from_params_file``
already understands), so a single ``input.param`` can drive both
implementations for a parity run instead of maintaining a hand-translated
JSON copy.

This is a translator, not a second parameter-handling implementation: no
constructor/backend logic lives here, only ``key value`` text -> dict, plus
the renames documented below. The mapping was built by reading every
``case('...')`` arm of ``amica15.f90``'s ``get_cmd_args`` (~amica15.f90:3100-
3700) against ``pamica.torch_impl.core.AMICATorchNG``'s constructor and
``validate_implementations.py``'s ``_NG_PARAMS``/``_HANDLED_KEYS``, the
authoritative pamica-side parameter names.

``FORTRAN_TO_PAMICA_KEY`` lists every Fortran keyword this module can
translate (53 keys covering 52 distinct pamica-side names -- Fortran accepts
both ``num_mix_comps`` and ``num_mix`` for the same setting). Of those, three
are renamed because the JSON schema spells them differently:

======================  ===================  ==========================
Fortran keyword         pamica/JSON key       Note
======================  ===================  ==========================
``num_mix_comps``       ``num_mix``           also accepts bare ``num_mix``
``share_iter``          ``share_int``         JSON schema's own spelling
``numrej``              ``maxrej``            matches ``AMICATorchNG.maxrej``
======================  ===================  ==========================

Every other translated key keeps its Fortran spelling.

``FORTRAN_UNSUPPORTED_KEYS`` lists every other keyword the Fortran parser
accepts (checkpoint warm-start, per-family EM freeze toggles, the
``do_opt_block`` block-size search, FIR/DFT pre-filtering, console/file
reporting cadence, ...) that has no pamica equivalent; these are
*deliberately* unmapped, not missed, and ``read_fortran_param_file`` warns
about them by name rather than dropping them in silence. A keyword absent
from *both* dicts is not a Fortran keyword this module knows about at all
(a typo, or a different ``amica15.f90`` revision than the one bundled here)
and is also warned about, separately.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# The bundled reference source, read to discover every keyword the reference
# binary's parameter parser accepts (its own `case('...')` arms), so this
# module's "recognized keyword" set cannot drift from the binary being
# targeted. See `_fortran_accepted_keys`.
_DEFAULT_SOURCE = Path(__file__).resolve().parent / "amica15.f90"

# ---------------------------------------------------------------------------
# Value types, read off each `case('key')` arm's `read(tmparg, '(fmt)')` in
# amica15.f90's get_cmd_args (~lines 3100-3700): '(i12)' + a `k == 1` branch
# is a boolean flag, plain '(i12)' is an integer, '(f15.12)'/'(e15.3)' is a
# float, '(a)' is a bare string, and `files`/`field_dim`/`num_samples` are
# whitespace-separated per-file lists.
# ---------------------------------------------------------------------------

_BOOL_KEYS = frozenset(
    {
        "use_grad_norm",
        "use_min_dll",
        "do_newton",
        "do_rho",
        "load_rho",
        "dble_data",
        "load_rej",
        "print_debug",
        "update_A",
        "load_A",
        "update_c",
        "load_c",
        "share_comps",
        "load_comp_list",
        "do_mean",
        "do_sphere",
        "do_approx_sphere",
        "load_all_param",
        "load_sphere",
        "load_mean",
        "update_mu",
        "load_mu",
        "update_beta",
        "load_beta",
        "update_alpha",
        "load_alpha",
        "update_gm",
        "load_gm",
        "write_nd",
        "write_LLt",
        "do_reject",
        "do_history",
        "do_opt_block",
        "fix_init",
        "doscaling",
    }
)

_FLOAT_KEYS = frozenset(
    {
        "min_grad_norm",
        "min_dll",
        "comp_thresh",
        "pcadb",
        "invsigmax",
        "invsigmin",
        "mineig",
        "lrate",
        "minlrate",
        "lratefact",
        "rholrate",
        "newtrate",
        "rholratefact",
        "rho0",
        "minrho",
        "maxrho",
        "rejsig",
    }
)

_STR_KEYS = frozenset({"outdir", "indir"})

_LIST_KEYS = frozenset({"files", "field_dim", "num_samples"})

# Every other keyword the parser accepts (`num_samples`/`data_dim`/... down to
# `seed`) reads as a plain integer; see `_coerce_value`.

# ---------------------------------------------------------------------------
# Fortran keyword -> pamica/JSON key. Identity unless noted otherwise above.
# ---------------------------------------------------------------------------

FORTRAN_TO_PAMICA_KEY: dict = {
    # Data-location metadata: not AMICATorchNG constructor kwargs, but carried
    # through unchanged since a caller needs them to load the same data
    # (matches `sample_params.json`'s own `files`/`outdir`/`data_dim`/
    # `field_dim`; `indir`/`num_samples` have no JSON-schema precedent but are
    # equally metadata-only, so they pass through the same way).
    "files": "files",
    "outdir": "outdir",
    "indir": "indir",
    "data_dim": "data_dim",
    "field_dim": "field_dim",
    "num_samples": "num_samples",
    "num_comps": "num_comps",
    # Model size / run length.
    "num_models": "num_models",
    "num_mix_comps": "num_mix",
    "num_mix": "num_mix",
    "max_iter": "max_iter",
    # Preprocessing.
    "do_mean": "do_mean",
    "do_sphere": "do_sphere",
    "do_approx_sphere": "do_approx_sphere",
    "pcakeep": "pcakeep",
    "pcadb": "pcadb",
    "mineig": "mineig",
    # Learning rate / convergence stops (issue #207: use_min_dll/min_dll and
    # use_grad_norm/min_grad_norm are the two independent per-iteration stops;
    # max_decs is the lrate-decrease-count stop).
    "lrate": "lrate",
    "minlrate": "minlrate",
    "lratefact": "lratefact",
    "use_min_dll": "use_min_dll",
    "min_dll": "min_dll",
    "use_grad_norm": "use_grad_norm",
    "min_grad_norm": "min_grad_norm",
    "max_decs": "max_decs",
    "block_size": "block_size",
    # Newton preconditioner.
    "do_newton": "do_newton",
    "newt_start": "newt_start",
    "newt_ramp": "newt_ramp",
    "newtrate": "newtrate",
    # Outlier rejection.
    "do_reject": "do_reject",
    "rejsig": "rejsig",
    "rejstart": "rejstart",
    "rejint": "rejint",
    "numrej": "maxrej",
    # GG shape (rho) adaptation.
    "rho0": "rho0",
    "minrho": "minrho",
    "maxrho": "maxrho",
    "rholrate": "rholrate",
    "rholratefact": "rholratefact",
    # Source-density family / kurtosis switch schedule.
    "pdftype": "pdftype",
    "kurt_start": "kurt_start",
    "num_kurt": "num_kurt",
    "kurt_int": "kurt_int",
    # Component sharing (multi-model).
    "share_comps": "share_comps",
    "comp_thresh": "comp_thresh",
    "share_start": "share_start",
    "share_iter": "share_int",
    # Scaling / mixing-matrix bounds.
    "doscaling": "doscaling",
    "scalestep": "scalestep",
    "invsigmax": "invsigmax",
    "invsigmin": "invsigmin",
    # Reproducibility.
    "seed": "seed",
}

# ---------------------------------------------------------------------------
# Every other keyword the reference parser accepts, with no pamica
# equivalent: DELIBERATELY unmapped, not missed. `read_fortran_param_file`
# warns about these by name rather than dropping them in silence.
# ---------------------------------------------------------------------------

FORTRAN_UNSUPPORTED_KEYS: dict = {
    # Raw-data I/O format: pamica's loaders take an already-typed array/dtype
    # from the caller instead of reading these from the param file.
    "dble_data": "raw-data byte format; pamica infers dtype from the array it is given",
    "byte_size": "raw-data byte width; see dble_data",
    # Pre-filtering pipeline, not ported.
    "filter_length": "FIR pre-filtering pipeline, not ported",
    "dft_length": "DFT pre-filtering pipeline, not ported",
    # Threading: PyTorch manages its own thread pool.
    "max_threads": "Fortran thread-pool size; PyTorch manages its own threading",
    # Optimal block-size search (do_opt_block sweep): pamica uses a single
    # fixed block_size (see AGENTS.md's block_size performance note).
    "blk_min": "do_opt_block block-size search, not ported",
    "blk_max": "do_opt_block block-size search, not ported",
    "blk_step": "do_opt_block block-size search, not ported",
    "do_opt_block": "block-size search toggle, not ported",
    # Checkpoint warm-start: Fortran can resume a run from a prior run's saved
    # parameter files. pamica has no equivalent load path from a param file
    # (AMICA.load restores its own saved state instead).
    "load_rho": "checkpoint warm-start, not ported",
    "load_rej": "checkpoint warm-start, not ported",
    "load_A": "checkpoint warm-start, not ported",
    "load_c": "checkpoint warm-start, not ported",
    "load_comp_list": "checkpoint warm-start, not ported",
    "load_all_param": "checkpoint warm-start, not ported",
    "load_sphere": "checkpoint warm-start, not ported",
    "load_mean": "checkpoint warm-start, not ported",
    "load_mu": "checkpoint warm-start, not ported",
    "load_beta": "checkpoint warm-start, not ported",
    "load_alpha": "checkpoint warm-start, not ported",
    "load_gm": "checkpoint warm-start, not ported",
    # Per-parameter-family EM freeze toggles: Fortran can hold a family fixed
    # during EM; pamica exposes no equivalent per-family freeze knobs.
    "update_A": "per-family EM freeze toggle, not exposed",
    "update_c": "per-family EM freeze toggle, not exposed",
    "update_mu": "per-family EM freeze toggle, not exposed",
    "update_beta": "per-family EM freeze toggle, not exposed",
    "update_alpha": "per-family EM freeze toggle, not exposed",
    "update_gm": "per-family EM freeze toggle, not exposed",
    "do_rho": (
        "GG shape (rho) freeze toggle; pamica derives it from pdftype "
        "(dorho = pdftype == 0), not independently settable"
    ),
    # Fortran-side console/output-file reporting: not applicable to the
    # Python API, which returns arrays in memory instead of writing them.
    "print_debug": "Fortran console verbosity, no pamica equivalent",
    "write_nd": "Fortran output-file toggle (weight/gradient dumps)",
    "write_LLt": "Fortran output-file toggle (per-sample log-likelihood dumps)",
    "writestep": "Fortran output-file write cadence",
    "do_history": "Fortran periodic weight-history dump",
    "histstep": "Fortran periodic weight-history dump cadence",
    # Misc.
    "decwindow": "decrease-count window size; pamica exposes only max_decs/maxdecs",
    "fix_init": "fixed initialization scheme, not ported",
}


def _fortran_accepted_keys(source: Optional[Path] = None) -> Optional[set]:
    """Keywords the reference binary's parameter parser accepts.

    Read from its own ``case('...')`` arms rather than hardcoded, so this
    cannot drift from the bundled binary's source. Returns ``None`` if the
    source is unavailable, which :func:`read_fortran_param_file` treats as
    "translate anything this module has a mapping for and do not warn about
    unrecognized keywords" (mirrors ``validate_implementations.fortran_accepted_keys``).

    ``source`` defaults to the module-level ``_DEFAULT_SOURCE`` looked up at
    call time (not bound as a default argument), so tests can point it at a
    missing path via ``monkeypatch.setattr``.
    """
    if source is None:
        source = _DEFAULT_SOURCE
    if not source.exists():
        return None
    text = source.read_text()
    return set(re.findall(r"^\s*case\('([^']+)'\)", text, re.MULTILINE))


def _coerce_value(key: str, raw: str):
    """Cast one value string to the type its Fortran ``read`` format implies."""
    if key in _LIST_KEYS:
        tokens = raw.split()
        if key == "files":
            return tokens
        try:
            return [int(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError(f"{key!r} expects integers, got {raw!r}") from exc
    if key in _BOOL_KEYS:
        try:
            flag = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{key!r} expects an integer 0/1 flag, got {raw!r}"
            ) from exc
        # Fortran's own semantics (amica15.f90: `if (k == 1) then ... .true.`):
        # only exactly 1 is true, so an out-of-range flag (e.g. 2) is false,
        # not an error, matching the reference reader.
        return flag == 1
    if key in _FLOAT_KEYS:
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{key!r} expects a float, got {raw!r}") from exc
    if key in _STR_KEYS:
        return raw
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key!r} expects an integer, got {raw!r}") from exc


def read_fortran_param_file(path: Union[str, Path]) -> dict:
    """Parse a Fortran ``input.param`` file into pamica/JSON-schema kwargs.

    Reads the literal Fortran text format -- whitespace-separated
    ``key value`` lines, ``#`` full-line comments (Fortran's own comment
    marker; see ``amica15.f90``'s ``get_cmd_args``), blank lines, ints/
    floats/strings, boolean flags as ``0``/``1`` -- and returns a dict shaped
    like ``sample_data/sample_params.json``: the same format
    ``AMICA.from_params_file`` already consumes for the JSON path (see
    ``AMICA.from_params_file`` in ``amica.py``, which auto-detects this
    format and routes through this function).

    Unlike the Fortran parser itself, which silently ``cycle``s past a line
    with no value, a malformed line here raises ``ValueError`` -- silently
    dropping a configured setting would contradict this project's no-silent-
    failures policy. A recognized-but-unsupported keyword (see
    ``FORTRAN_UNSUPPORTED_KEYS``) or an unrecognized one is instead a
    ``logger.warning`` naming the keyword, never a silent drop.

    Parameters
    ----------
    path : str or Path
        Path to the Fortran-format parameter file.

    Returns
    -------
    dict
        Parsed settings, keyed by their pamica/JSON name (see
        ``FORTRAN_TO_PAMICA_KEY``). When the same setting is repeated (e.g.
        both ``num_mix_comps`` and ``num_mix``), the last line in the file
        wins, matching the Fortran parser's own sequential overwrite.
    """
    path = Path(path)
    accepted = _fortran_accepted_keys()
    result: dict = {}
    unrecognized: set = set()
    unsupported: set = set()

    with path.open("r") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                raise ValueError(
                    f"{path}:{lineno}: malformed line, expected '<key> <value>': "
                    f"{raw_line.rstrip()!r}"
                )
            key, raw_value = parts[0], parts[1].strip()

            if accepted is not None and key not in accepted:
                unrecognized.add(key)
                continue

            json_key = FORTRAN_TO_PAMICA_KEY.get(key)
            if json_key is None:
                unsupported.add(key)
                continue

            try:
                result[json_key] = _coerce_value(key, raw_value)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc

    if unrecognized:
        logger.warning(
            "%s: %d keyword(s) not recognized by the reference Fortran parser "
            "(dropped -- a typo, or a different amica15.f90 revision than the "
            "one bundled with pamica): %s",
            path,
            len(unrecognized),
            sorted(unrecognized),
        )
    if unsupported:
        logger.warning(
            "%s: %d Fortran keyword(s) have no pamica equivalent and are "
            "deliberately dropped (see pamica.fortran_params."
            "FORTRAN_UNSUPPORTED_KEYS and docs/guides/validation.md#parameter-files): %s",
            path,
            len(unsupported),
            sorted(unsupported),
        )
    return result
