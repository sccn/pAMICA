"""Translator from the Fortran ``input.param`` text format to pamica parameters
(issue #132).

The reference Fortran binary (``amica15.f90``) and ``AMICA.from_params_file``
(``amica.py``) both configure a run from a parameter file, but they use two
different formats and, for a handful of settings, two different spellings of
the same keyword. This module parses the literal Fortran text format --
whitespace-separated ``key value`` lines, ``#`` full-line comments, ints/
floats/strings, boolean flags as ``0``/``1`` -- into a dict of pamica
constructor/``fit()`` keyword arguments (see below), which
``AMICA.from_params_file`` applies the same way it already applies a JSON
parameter file, so a single ``input.param`` can drive both implementations
for a parity run instead of maintaining a hand-translated JSON copy.

This is a translator, not a second parameter-handling implementation: no
constructor/backend logic lives here, only ``key value`` text -> dict, plus
the renames documented below. The mapping was built by reading every
``case('...')`` arm of ``amica15.f90``'s ``get_cmd_args`` (~amica15.f90:3100-
3700) against ``pamica.torch_impl.core.AMICATorchNG``'s constructor and
``validate_implementations.py``'s ``_NG_PARAMS``/``_HANDLED_KEYS``, the
authoritative pamica-side parameter names.

The output dict's keys target the actual Python call surface --
``AMICA.fit``'s named parameters (``max_iter``, ``lrate``, ``do_mean``,
``do_sphere``, ``do_newton``) and ``AMICATorchNG`` constructor keywords --
because ``AMICA.from_params_file`` forwards this dict into ``fit()`` as
per-call defaults (see ``amica.py``). A key that does not spell one of those
names exactly would silently fail to forward (or raise ``TypeError`` if
forwarded blindly), so where the JSON schema (``sample_params.json``) itself
spells a setting differently from the constructor, this module targets the
*constructor's* spelling, not the JSON schema's.

``FORTRAN_TO_PAMICA_KEY`` lists every Fortran keyword this module can
translate (53 keys covering 52 distinct pamica-side names -- Fortran accepts
both ``num_mix_comps`` and ``num_mix`` for the same setting). Of those, three
are renamed because Fortran spells them differently from the pamica-side
name:

======================  =========================  ================================
Fortran keyword         pamica key                 Note
======================  =========================  ================================
``min_grad_norm``       ``min_nd``                 matches ``AMICATorchNG.min_nd``
``max_decs``            ``maxdecs``                matches ``AMICATorchNG.maxdecs``
``numrej``              ``maxrej``                 matches ``AMICATorchNG.maxrej``
======================  =========================  ================================

``num_mix_comps``/``num_mix`` both collapse to the pamica key ``num_mix``,
which is not itself a constructor keyword (the constructor takes ``n_mix``);
``AMICA.from_params_file`` reads ``num_mix``/``num_models`` directly to size
the instance at construction time (unchanged pre-existing behavior), so
those two keys are consumed *before* the rest of the dict ever reaches
``fit()``'s per-call-default merge. ``share_iter`` is **not** renamed here --
it already matches ``AMICATorchNG.share_iter`` exactly. This is a deliberate
divergence from ``sample_params.json``, whose own schema spells the same
setting ``share_int`` (and, along with its ``max_decs``/``min_grad_norm``
keys, does not match the constructor either -- a pre-existing gap in that
JSON file, out of scope here, that ``AMICA.fit``'s per-call-default merge
now surfaces as a "not applied" warning when fitting from it).

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
# Fortran keyword -> pamica constructor/fit() key. Identity unless noted
# otherwise in the module docstring's rename table.
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
    # use_grad_norm/min_nd are the two independent per-iteration stops;
    # maxdecs is the lrate-decrease-count stop). min_grad_norm -> min_nd and
    # max_decs -> maxdecs are renamed to match AMICATorchNG's constructor
    # spelling (see the module docstring's rename table).
    "lrate": "lrate",
    "minlrate": "minlrate",
    "lratefact": "lratefact",
    "use_min_dll": "use_min_dll",
    "min_dll": "min_dll",
    "use_grad_norm": "use_grad_norm",
    "min_grad_norm": "min_nd",
    "max_decs": "maxdecs",
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
    # Component sharing (multi-model). share_iter is identity: it already
    # matches AMICATorchNG's constructor spelling (sample_params.json's own
    # "share_int" spelling does not -- see the module docstring).
    "share_comps": "share_comps",
    "comp_thresh": "comp_thresh",
    "share_start": "share_start",
    "share_iter": "share_iter",
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
        # Fortran double-precision literals may use 'd'/'D' as the exponent
        # marker instead of 'e'/'E' (e.g. "1.5d-3"); normalize before parsing.
        try:
            return float(raw.replace("D", "e").replace("d", "e"))
        except ValueError as exc:
            raise ValueError(f"{key!r} expects a float, got {raw!r}") from exc
    if key in _STR_KEYS:
        return raw
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key!r} expects an integer, got {raw!r}") from exc


def read_fortran_param_file(path: Union[str, Path]) -> dict:
    """Parse a Fortran ``input.param`` file into pamica constructor/``fit()`` kwargs.

    Reads the literal Fortran text format -- whitespace-separated
    ``key value`` lines, ``#`` full-line comments (Fortran's own comment
    marker; see ``amica15.f90``'s ``get_cmd_args``), blank lines, ints/
    floats/strings, boolean flags as ``0``/``1`` -- and returns a dict keyed
    by pamica's own parameter names (see ``FORTRAN_TO_PAMICA_KEY``), which
    ``AMICA.from_params_file`` (``amica.py``, which auto-detects this format
    and routes through this function) applies the same way it already
    applies a JSON parameter file.

    Unlike the Fortran parser itself, which silently ``cycle``s past a line
    with no value, a malformed line here raises ``ValueError`` -- silently
    dropping a configured setting would contradict this project's no-silent-
    failures policy. A recognized-but-unsupported keyword (see
    ``FORTRAN_UNSUPPORTED_KEYS``) or an unrecognized one is instead a
    ``logger.warning`` naming the keyword, never a silent drop. If a
    non-empty file has content lines but not one keyword is recognized as
    valid Fortran syntax at all (e.g. a JSON file mistakenly handed to this
    reader instead of ``json.load``), that is also a hard ``ValueError``
    rather than a silent all-defaults return; a file of only known-but-
    unsupported keywords is unaffected (it is a real, if useless, param
    file, not a format mismatch).

    Fortran's own comment marker is a line-*leading* ``#`` only; this reader
    is deliberately more permissive (fail-safe direction) and also strips an
    inline ``" #..."`` trailing comment from a value before type coercion.
    Float values additionally accept Fortran's ``d``/``D`` double-precision
    exponent marker (e.g. ``"1.5d-3"``), normalized to ``e`` before
    ``float()``.

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
    content_lines = 0
    recognized_lines = 0

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
            content_lines += 1
            key, raw_value = parts[0], parts[1].strip()
            # Strip a trailing inline comment before type coercion (deliberately
            # more permissive than Fortran's own line-leading-only `#`).
            hash_idx = raw_value.find(" #")
            if hash_idx != -1:
                raw_value = raw_value[:hash_idx].rstrip()
            if not raw_value:
                raise ValueError(
                    f"{path}:{lineno}: malformed line, expected '<key> <value>': "
                    f"{raw_line.rstrip()!r}"
                )

            if accepted is not None and key not in accepted:
                unrecognized.add(key)
                continue
            recognized_lines += 1

            json_key = FORTRAN_TO_PAMICA_KEY.get(key)
            if json_key is None:
                unsupported.add(key)
                continue

            try:
                result[json_key] = _coerce_value(key, raw_value)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc

    # Only meaningful when `accepted` is known: a file where not a single
    # content line's keyword was recognized as valid Fortran syntax at all is
    # very likely the wrong format entirely (e.g. JSON handed to this reader
    # by mistake), so this is a hard error rather than a silent all-defaults
    # result. A file of only known-but-unsupported keywords (recognized, just
    # not translatable) is a legitimate, if useless, param file and does NOT
    # hit this -- see FORTRAN_UNSUPPORTED_KEYS's warning below instead.
    if accepted is not None and content_lines and recognized_lines == 0:
        raise ValueError(
            f"{path}: {content_lines} content line(s) but not one keyword was "
            "recognized by the reference Fortran parser -- this usually means "
            "the file is not actually the Fortran input.param text format "
            "(e.g. JSON handed to read_fortran_param_file by mistake). "
            "Refusing to silently return an all-defaults result."
        )

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
