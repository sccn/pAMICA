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
  model ``h`` in column ``comp_list(i, h)``. pamica stores each model's block
  transposed (issue #24 convention, ADR 0006), so the reference block is the
  transpose of pamica's stored block.
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


def reference_mixing(A_stored: np.ndarray, comp_list: np.ndarray) -> np.ndarray:
    """pamica's stored ``A`` in the reference layout: block ``h`` transposed.

    Only defined for a disjoint ``comp_list`` (no shared columns): in pamica's
    current layout a merged column belongs to rows of several blocks, which has
    no reference counterpart (issue #334).
    """
    if np.unique(comp_list).size != comp_list.size:
        raise ValueError("reference_mixing needs a comp_list without shared columns")
    A_ref = np.empty_like(A_stored)
    for h in range(comp_list.shape[1]):
        idx = comp_list[:, h]
        A_ref[:, idx] = A_stored[:, idx].T
    return A_ref


def seed_from_torch(model) -> SeedState:
    """The reference-layout state of an initialized (or fitted) ``AMICATorchNG``."""
    comp_list = model.comp_list.cpu().numpy()
    return SeedState(
        A=reference_mixing(model.A.cpu().numpy(), comp_list),
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
    ``doscaling``, ``block_size``, ...). Raises ``RuntimeError`` with the
    binary's output tail if it fails or writes no ``A``.
    """
    nw, num_comps = state.A.shape
    num_mix = state.mu.shape[0]
    num_models = state.gm.shape[0]
    nx = state.mean.shape[0]
    if nx != nw:
        raise ValueError("seeded reference runs are full rank (nx == nw) only")

    workdir.mkdir(parents=True, exist_ok=True)
    indir = workdir / "init"
    indir.mkdir(exist_ok=True)
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

    link = workdir / "data.fdt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(data_file).resolve())

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
    out = workdir / "out"
    if proc.returncode != 0 or not (out / "A").exists():
        raise RuntimeError(
            f"native AMICA failed (exit {proc.returncode}).\n"
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
        S=_read_f64(out / "S", (nx, nx)),
        stdout=proc.stdout,
    )
