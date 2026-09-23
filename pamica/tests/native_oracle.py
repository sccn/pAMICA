"""Seed the native reference binary from a pamica state and read back its raw output.

A bit-level oracle for the M-step: pamica's own initialization (or any state
reached from it) is written in the reference's ``load_*`` file formats, the
binary runs a few iterations from exactly that state, and its raw output arrays
are read back without the variance ordering or normalization that ``loadmodout``
applies. Any per-iteration difference then shows up element by element instead
of hiding behind scale-blind endpoint metrics (issue #333, epic #324 Phase 7).

Not a test module (no ``test_`` prefix, so pytest does not collect it). Callers
gate on ``AMICA_RUN_FORTRAN=1`` like the other binary-driven tests.

Reference facts this relies on (``amica15.f90``):

* Layout: the reference mixing matrix ``A(nw, num_comps)`` holds source ``i`` of
  model ``h`` in column ``comp_list(i, h)``. pamica stores one component per
  row (issue #334, ADR 0007), so the reference ``A`` is pamica's stored ``A``
  transposed, merged ``comp_list`` or not.
* Every ``load_*`` file is a raw little-endian float64 array in column-major
  order under ``indir``. A loaded ``A`` is used as is (no normalization,
  :793-800), unlike a drawn one (:818-819).
* ``load_sphere`` is not used: its load path dereferences an unallocated
  temporary and crashes (``.context/issue-24/findings.md``). The binary computes
  its own sphere, the same symmetric zero-phase component analysis (ZCA) sphere
  pamica builds from the same data, so ``S`` is returned for the caller to check.
* ``load_comp_list`` reads the file named ``c``, not ``comp_list`` (:831, a
  reference bug), so a comp_list seed and a ``c`` seed cannot be combined. With
  ``comp_list`` given, :func:`run_seeded_reference` writes it to ``c`` and leaves
  the bias at the binary's own zero initialization, which requires a zero ``c``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pamica.native import resolver
from pamica.native.engine import _DEFAULT_PARAMS, _render_param

# The release whose native binary the seeded oracles run (resolver cache, else a
# verified download); PAMICA_NATIVE_BINARY overrides it with a local build.
REFERENCE_VERSION = "v0.3.3"

# Every output file :func:`run_seeded_reference` reads back.
_OUTPUT_FILES = ("A", "mu", "sbeta", "rho", "alpha", "gm", "c", "comp_list", "LL", "S")
# stdout lines of a run that did not carry the seeded state to the end
# (amica15.f90): an "Error: ..." before a bare ``stop``, which exits with
# status 0 (:264, :2452, :3077, ...), a NaN exit (:1054), or a NaN restart,
# which draws a fresh ``A`` and so discards the seed (:1021-1046, printed as
# "Reinitializaing"). A gfortran runtime error goes to stderr.
_STDOUT_FAILURE = re.compile(r"\berror\b|Got NaN|Reinitiali", re.IGNORECASE)
_STDERR_FAILURE = re.compile(r"Fortran runtime error", re.IGNORECASE)


@dataclass(frozen=True)
class SeedState:
    """A model state in the reference's layout, ready for the ``load_*`` files.

    ``A`` is ``(nw, num_comps)`` with source ``i`` of model ``h`` in column
    ``comp_list[i, h]``; ``mu``/``sbeta``/``rho``/``alpha`` are
    ``(num_mix, num_comps)``, ``gm`` is ``(num_models,)``, ``c`` is
    ``(nw, num_models)`` and ``mean`` is ``(nx,)``. ``comp_list`` is 0-based
    ``(nw, num_models)``; ``None`` means the reference's default
    ``comp_list(i, h) = (h-1)*nw + i``.
    """

    A: np.ndarray
    mean: np.ndarray
    mu: np.ndarray
    sbeta: np.ndarray
    rho: np.ndarray
    alpha: np.ndarray
    gm: np.ndarray
    c: np.ndarray
    comp_list: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ReferenceOutput:
    """The binary's raw output arrays, in the reference layout (see
    :class:`SeedState`), plus the per-iteration ``LL`` and its sphere ``S``."""

    A: np.ndarray
    mu: np.ndarray
    sbeta: np.ndarray
    rho: np.ndarray
    alpha: np.ndarray
    gm: np.ndarray
    c: np.ndarray
    comp_list: np.ndarray
    LL: np.ndarray
    S: np.ndarray
    stdout: str


def reference_mixing(A_stored: np.ndarray) -> np.ndarray:
    """pamica's stored component-row ``A`` in the reference layout
    ``(nw, num_comps)``: its transpose (issue #334), for any ``comp_list``."""
    return np.ascontiguousarray(np.asarray(A_stored).T)


def seed_from_torch(model) -> SeedState:
    """The reference-layout state of an initialized (or fitted) ``AMICATorchNG``.

    ``comp_list`` is left at the reference's default; pass a merged one with
    ``dataclasses.replace`` (the ``load_comp_list`` path, which needs ``c == 0``).
    """
    return SeedState(
        A=reference_mixing(model.A.cpu().numpy()),
        mean=model.mean.cpu().numpy().reshape(-1),
        mu=model.mu.cpu().numpy(),
        sbeta=model.beta.cpu().numpy(),
        rho=model.rho.cpu().numpy(),
        alpha=model.alpha.cpu().numpy(),
        gm=model.gm.cpu().numpy(),
        c=model.c.cpu().numpy(),
    )


def _write_f64(path: Path, arr: np.ndarray) -> None:
    np.asarray(arr, dtype="<f8").flatten(order="F").tofile(path)


def _read_f64(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    need = int(np.prod(shape))
    if raw.size < need:
        raise RuntimeError(f"{path.name}: {raw.size} values, expected {need}")
    return raw[:need].reshape(shape, order="F")


def history_mixing(workdir: Path, nw: int, num_comps: int) -> dict[int, np.ndarray]:
    """The reference-layout ``A`` the binary wrote after each iteration.

    A run with ``do_history=1`` and ``histstep=1`` passed to
    :func:`run_seeded_reference` writes its state after every iteration's
    update to ``out/history/<iter>/`` (``write_history``, amica15.f90:2287-2324,
    called at :1130). Keys are the reference's 1-based iterations; an iteration
    that ends the run on a stop takes no update and writes no history.
    """
    root = Path(workdir) / "out" / "history"
    return {
        int(d.name): _read_f64(d / "A", (nw, num_comps))
        for d in root.iterdir()
        if d.name.isdigit() and (d / "A").is_file()
    }


def run_seeded_reference(
    state: SeedState,
    data_file: Path,
    workdir: Path,
    *,
    n_samples: int,
    max_iter: int,
    threads: int = 2,
    timeout: float = 600.0,
    binary: Optional[Path] = None,
    **params: object,
) -> ReferenceOutput:
    """Run the reference binary for ``max_iter`` iterations from ``state``.

    ``data_file`` is the raw float32 column-major recording
    (``nx x n_samples``) the pamica side fitted; it is linked into ``workdir``.
    ``params`` override the engine defaults by Fortran name (``do_newton``,
    ``doscaling``, ``block_size``, ...).

    A reused ``workdir`` is safe: its ``out`` and ``init`` directories are
    removed first, so every file read back was written by this run. Raises
    ``RuntimeError`` with the tails of the binary's stdout and stderr if it
    exits nonzero, prints a failure line (see ``_STDOUT_FAILURE``), or leaves
    any output file this helper reads unwritten.
    """
    nw, num_comps = state.A.shape
    num_mix = state.mu.shape[0]
    num_models = state.gm.shape[0]
    nx = state.mean.shape[0]
    if nx != nw:
        raise ValueError("seeded reference runs are full rank (nx == nw) only")

    workdir.mkdir(parents=True, exist_ok=True)
    indir = workdir / "init"
    out = workdir / "out"
    # Old output or seed files from an earlier run must never be read back;
    # the binary creates ``out`` itself (amica15.f90:82).
    for stale in (out, indir):
        if stale.exists():
            shutil.rmtree(stale)
    indir.mkdir()
    _write_f64(indir / "A", state.A)
    _write_f64(indir / "mean", state.mean)
    _write_f64(indir / "mu", state.mu)
    _write_f64(indir / "sbeta", state.sbeta)
    _write_f64(indir / "rho", state.rho)
    _write_f64(indir / "alpha", state.alpha)
    _write_f64(indir / "gm", state.gm)
    if state.comp_list is None:
        _write_f64(indir / "c", state.c)
        load = {"load_c": 1, "load_comp_list": 0}
    else:
        if np.any(state.c != 0.0):
            raise ValueError(
                "load_comp_list reads the file 'c' (amica15.f90:831), so a "
                "comp_list seed needs c == 0 (the binary's own initialization)"
            )
        # Integer comp_list, 1-based, in a record of 2*nbyte*nw*num_models bytes.
        rec = np.zeros(2 * nw * num_models, dtype="<i4")
        rec[: nw * num_models] = (state.comp_list + 1).flatten(order="F")
        rec.tofile(indir / "c")
        load = {"load_c": 0, "load_comp_list": 1}

    _link_data(workdir, data_file)
    param = {
        # `files` must come first: amica15.f90 stops if other keys precede it.
        "files": "./data.fdt",
        "outdir": "./out/",
        **_DEFAULT_PARAMS,
        "indir": "./init",
        "data_dim": nx,
        "field_dim": n_samples,
        "pcakeep": nw,
        "num_models": num_models,
        "num_mix_comps": num_mix,
        "max_iter": max_iter,
        "max_threads": threads,
        "write_LLt": 0,
        "writestep": max_iter + 1,
        "load_mean": 1,
        "load_sphere": 0,
        "load_A": 1,
        "load_mu": 1,
        "load_beta": 1,
        "load_rho": 1,
        "load_alpha": 1,
        "load_gm": 1,
        **load,
        **params,
    }
    return _run_reference(
        param,
        workdir,
        nw=nw,
        num_comps=num_comps,
        num_mix=num_mix,
        num_models=num_models,
        max_iter=max_iter,
        threads=threads,
        timeout=timeout,
        binary=binary,
    )


def run_drawn_reference(
    data_file: Path,
    workdir: Path,
    *,
    nw: int,
    n_samples: int,
    num_models: int,
    num_mix: int,
    seed: int,
    max_iter: int = 0,
    threads: int = 2,
    timeout: float = 600.0,
    binary: Optional[Path] = None,
    **params: object,
) -> ReferenceOutput:
    """Run the reference binary from its OWN drawn initialization.

    Nothing is loaded: the binary draws ``A`` (and ``mu``, ``sbeta``) with
    gfortran's ``random_number``, seeded deterministically from ``seed``
    (amica15.f90:222-238). With ``max_iter=0`` its loop exits before the first
    iteration (:955) and ``write_output`` writes the initialization itself, so
    the returned ``A`` is the drawn initial mixing matrix (issue #341).
    Full rank only (``nw`` channels, ``pcakeep = nw``); ``params`` override
    the engine defaults by Fortran name. Raises like
    :func:`run_seeded_reference`.
    """
    num_comps = nw * num_models
    workdir.mkdir(parents=True, exist_ok=True)
    # Old output from an earlier run must never be read back.
    out = workdir / "out"
    if out.exists():
        shutil.rmtree(out)
    _link_data(workdir, data_file)
    param = {
        # `files` must come first: amica15.f90 stops if other keys precede it.
        "files": "./data.fdt",
        "outdir": "./out/",
        **_DEFAULT_PARAMS,
        "data_dim": nw,
        "field_dim": n_samples,
        "pcakeep": nw,
        "num_models": num_models,
        "num_mix_comps": num_mix,
        "max_iter": max_iter,
        "max_threads": threads,
        "write_LLt": 0,
        "writestep": max_iter + 1,
        "seed": seed,
        **params,
    }
    return _run_reference(
        param,
        workdir,
        nw=nw,
        num_comps=num_comps,
        num_mix=num_mix,
        num_models=num_models,
        max_iter=max_iter,
        threads=threads,
        timeout=timeout,
        binary=binary,
    )


def _link_data(workdir: Path, data_file: Path) -> None:
    """Link the raw recording into ``workdir`` as ``data.fdt``."""
    link = workdir / "data.fdt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(data_file).resolve())


def _run_reference(
    param: dict,
    workdir: Path,
    *,
    nw: int,
    num_comps: int,
    num_mix: int,
    num_models: int,
    max_iter: int,
    threads: int,
    timeout: float,
    binary: Optional[Path],
) -> ReferenceOutput:
    """Write ``param`` as ``input.param`` in ``workdir``, run the binary there
    and read its raw output back (full rank: the sphere is ``nw x nw``). Raises
    as :func:`run_seeded_reference` describes."""
    out = workdir / "out"
    (workdir / "input.param").write_text(_render_param(param))

    exe = binary if binary is not None else resolver.resolve(REFERENCE_VERSION)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    proc = subprocess.run(
        [str(exe), "input.param"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    missing = [name for name in _OUTPUT_FILES if not (out / name).is_file()]
    failure = _STDOUT_FAILURE.search(proc.stdout) or _STDERR_FAILURE.search(proc.stderr)
    if proc.returncode != 0 or missing or failure:
        raise RuntimeError(
            f"native AMICA failed (exit {proc.returncode}; "
            f"missing output: {', '.join(missing) or 'none'}; "
            f"failure line: {failure.group(0) if failure else 'none'}).\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
        )
    comp_list = np.fromfile(out / "comp_list", dtype="<i4")[: nw * num_models]
    return ReferenceOutput(
        A=_read_f64(out / "A", (nw, num_comps)),
        mu=_read_f64(out / "mu", (num_mix, num_comps)),
        sbeta=_read_f64(out / "sbeta", (num_mix, num_comps)),
        rho=_read_f64(out / "rho", (num_mix, num_comps)),
        alpha=_read_f64(out / "alpha", (num_mix, num_comps)),
        gm=_read_f64(out / "gm", (num_models,)),
        c=_read_f64(out / "c", (nw, num_models)),
        comp_list=comp_list.reshape(nw, num_models, order="F") - 1,
        LL=_read_f64(out / "LL", (max_iter,)),
        S=_read_f64(out / "S", (nw, nw)),
        stdout=proc.stdout,
    )
