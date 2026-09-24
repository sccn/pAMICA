"""``AMICANative``: run the native AMICA Fortran binary as a pamica backend.

The fourth run engine, alongside the NumPy, PyTorch (``AMICATorchNG``) and MLX
backends. It writes the data and an ``input.param``, runs the dependency-free
binary resolved by :mod:`pamica.native.resolver`, and reads the result back
through :func:`pamica.numpy_impl.load.loadmodout` -- so its output is an
``AmicaOutput`` with the same accessors (``.sources``, ``.W``, ``.A``, ...) as a
loaded fit. This is the Fortran reference itself, so it is the parity oracle the
Python backends are validated against.
"""

from __future__ import annotations

import functools
import inspect
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..fortran_params import FORTRAN_TO_PAMICA_KEY
from ..numpy_impl.load import AmicaOutput, loadmodout
from . import resolver

# Settings the binary reads that no pamica backend has: its thread count, the
# raw-data format, its output files and checkpoint cadence, the checkpoint
# loading switches and the per-family update switches. They keep the values of
# the bundled sample_data/input.param, as before issue #354. doPCA,
# field_blocksize and load_W are keys that file carries and this amica15.f90
# does not parse (pamica.fortran_params warns about them the same way).
_NATIVE_ONLY_PARAMS: dict[str, object] = {
    "max_threads": 10,
    "num_samples": 1,
    "field_blocksize": 1,
    "do_history": 0,
    "histstep": 10,
    "writestep": 20,
    "write_nd": 0,
    "write_LLt": 1,
    "decwindow": 1,
    "fix_init": 0,
    "update_A": 1,
    "update_c": 1,
    "update_gm": 1,
    "update_alpha": 1,
    "update_mu": 1,
    "update_beta": 1,
    "do_rho": 1,
    "load_rej": 0,
    "load_W": 0,
    "load_c": 0,
    "load_gm": 0,
    "load_alpha": 0,
    "load_mu": 0,
    "load_beta": 0,
    "load_rho": 0,
    "load_comp_list": 0,
    "doPCA": 1,
    "byte_size": 4,
}

# The two canonical pamica keys (pamica.fortran_params) that AMICATorchNG
# spells differently in its constructor.
_TORCH_SPELLING = {"num_models": "n_models", "num_mix": "n_mix"}


@functools.cache
def _shared_default_params() -> dict[str, object]:
    """pamica's shared defaults under the binary's keywords (issue #354).

    Every keyword of ``pamica.fortran_params.FORTRAN_TO_PAMICA_KEY`` whose
    pamica setting ``AMICATorchNG`` has takes ``AMICATorchNG``'s default
    (``max_iter`` from its ``fit``), the same value the other backends hold
    (``pamica/tests/test_default_settings.py``). A setting whose default is
    ``None`` (``pcakeep``, ``pcadb``, ``seed``) is left out of this table;
    ``_input_params`` then fills ``pcakeep`` with the data's channel count.
    pamica settings with no binary keyword (``keep_best``, ``mineig_rel``,
    ``n_restarts``, ...) have nothing to write. Then the native-only knobs.
    Imported lazily: the torch backend's signature is the shared source.
    """
    from ..torch_impl.core import AMICATorchNG

    shared = {
        name: param.default
        for name, param in inspect.signature(AMICATorchNG).parameters.items()
    }
    shared["max_iter"] = (
        inspect.signature(AMICATorchNG.fit).parameters["max_iter"].default
    )
    params: dict[str, object] = {}
    for fortran_key, pamica_key in FORTRAN_TO_PAMICA_KEY.items():
        if fortran_key == "num_mix":  # the binary's second spelling of num_mix_comps
            continue
        value = shared.get(_TORCH_SPELLING.get(pamica_key, pamica_key))
        if value is not None:
            params[fortran_key] = value
    params.update(_NATIVE_ONLY_PARAMS)
    return params


def _default_params() -> dict[str, object]:
    """A fresh copy of :func:`_shared_default_params`, for a caller to modify."""
    return dict(_shared_default_params())


# Friendly kwarg -> Fortran param-name aliases (match the Python backends' names).
_ALIASES = {"n_models": "num_models", "n_mix": "num_mix_comps"}


def _render_param(params: dict[str, object]) -> str:
    return "".join(f"{k} {_fmt(v)}\n" for k, v in params.items())


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.6e}" if (v != 0 and abs(v) < 1e-3) else f"{v:.6f}"
    return str(v)


class AMICANative:
    """Run the native AMICA binary on data and expose the result as an
    ``AmicaOutput``.

    Parameters
    ----------
    binary : path-like, optional
        Explicit binary path; otherwise resolved for the host (downloaded from the
        release on first use). Equivalent to setting ``PAMICA_NATIVE_BINARY``.
    version : str, default "latest"
        Release tag to resolve the binary from when not given explicitly.
    threads : int, optional
        ``OMP_NUM_THREADS`` for the run (default: the binary's own default).
    timeout : float, optional
        Seconds before the subprocess is killed (default: no timeout).
    **params
        Any Fortran ``input.param`` field (or a friendly alias: ``n_models``,
        ``n_mix``), overriding the defaults; e.g. ``max_iter``, ``lrate``,
        ``pdftype``, ``do_newton``.

    Notes
    -----
    The defaults are pamica's shared ones since issue #354, the values the
    PyTorch, MLX and NumPy backends default to, under the binary's keywords
    (``lrate`` 0.1, Newton off, ``max_iter`` 100, ``do_opt_block`` off,
    ``pdftype`` 0, ...). pamica settings the binary has no keyword for, such
    as ``keep_best`` and ``mineig_rel``, are not written, so the binary
    returns its last iterate and uses the absolute ``mineig`` floor.

    The binary's ``block_size`` counts one thread's share of a block, and it
    processes no block at all when ``max_threads * block_size`` exceeds the
    samples (an all-NaN fit, issue #292). Unless ``block_size`` is given, the
    engine therefore writes pamica's 8192-sample block, capped at the data's
    length, divided by ``max_threads`` (10 unless given): 819 for a long
    recording. A given ``block_size`` is written as is. Before issue #354 the defaults were the bundled
    ``sample_data/input.param``'s (``lrate`` 0.05, Newton on from iteration
    50, ``max_iter`` 2000, ``block_size`` 512); pass that file's settings as
    keywords to run it (see ``docs/api/native-backend.md``).
    """

    def __init__(
        self,
        binary: Optional[Union[str, Path]] = None,
        *,
        version: str = "latest",
        threads: Optional[int] = None,
        timeout: Optional[float] = None,
        **params: object,
    ) -> None:
        # resolve() now: the subprocess runs with cwd set to a tempdir, so a
        # relative binary path would pass the existence check but fail to launch.
        self.binary = Path(binary).resolve() if binary is not None else None
        self.version = version
        self.threads = threads
        self.timeout = timeout
        self.params = params
        self.output_: Optional[AmicaOutput] = None

    def _resolve_binary(self) -> Path:
        if self.binary is not None:
            if not self.binary.exists():
                raise FileNotFoundError(f"binary not found: {self.binary}")
            return self.binary
        return resolver.resolve(self.version)

    def _input_params(
        self, n_channels: int, n_samples: int, params: dict[str, object]
    ) -> dict[str, object]:
        """The ``input.param`` settings :meth:`fit` writes for data of this
        shape: pamica's shared defaults, then the constructor's and ``fit``'s
        keywords (friendly aliases mapped), then the data's dimensions."""
        merged = _default_params()
        given = set()
        for src in (self.params, params):
            for key, value in src.items():
                merged[_ALIASES.get(key, key)] = value
                given.add(_ALIASES.get(key, key))
        if "block_size" not in given:
            # pamica's block_size counts the samples of one block; the
            # binary's counts one thread's share of it, and it runs
            # n_samples // (max_threads * block_size) blocks, so a block larger
            # than the data would leave it none and an all-NaN fit (issue
            # #292). The default is therefore pamica's block, at most the
            # data, split over the threads.
            # int(str(...)): a value passed as a file's text ("10") reads the same.
            threads = int(str(merged["max_threads"]))
            block = min(int(str(merged["block_size"])), n_samples)
            merged["block_size"] = max(1, block // threads)
        merged["data_dim"] = n_channels
        merged["field_dim"] = n_samples
        if not merged.get("pcakeep"):
            merged["pcakeep"] = n_channels  # default: keep all components
        # `files` must come first: amica15.f90 hard-stops if it parses other
        # keys before the data file. Dict insertion order preserves that.
        return {"files": "./data.fdt", "outdir": "./amicaout/", **merged}

    def fit(self, X: np.ndarray, **params: object) -> "AMICANative":
        """Run AMICA on ``X`` (shape ``(n_channels, n_samples)``) and store the
        result as ``self.output_`` (an ``AmicaOutput``)."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_channels, n_samples); got {X.shape}")
        n_channels, n_samples = X.shape
        param = self._input_params(n_channels, n_samples, params)

        binary = self._resolve_binary()

        with tempfile.TemporaryDirectory(prefix="amica_native_") as td:
            work = Path(td)
            # AMICA reads the data as raw byte_size floats in column-major order
            # (numpy_impl/data.py: reshape order="F"); write it that way.
            byte_size = param["byte_size"]
            dtype = (
                np.float32
                if isinstance(byte_size, int) and byte_size == 4
                else np.float64
            )
            X.astype(dtype).ravel(order="F").tofile(work / "data.fdt")

            outdir = work / "amicaout"
            outdir.mkdir()
            (work / "input.param").write_text(_render_param(param))

            env = None
            if self.threads is not None:
                import os

                env = {**os.environ, "OMP_NUM_THREADS": str(self.threads)}

            proc = subprocess.run(
                [str(binary), "input.param"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"native AMICA failed (exit {proc.returncode}).\n"
                    f"stdout tail:\n{proc.stdout[-2000:]}\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}"
                )
            if not (outdir / "W").exists():
                raise RuntimeError(
                    "native AMICA produced no output (no 'W' file); stdout tail:\n"
                    f"{proc.stdout[-2000:]}"
                )
            # A collapsed fit writes NaN weights (and zero model probabilities);
            # loadmodout's pinv(W@S) would then fail with an opaque SVD error, so
            # detect it here and report it as the degenerate fit it is (cf. the
            # #50 degenerate-fit contract for the Python backends). An empty W
            # (truncated write) is caught too -- an all-NaN check vacuously passes
            # on a zero-length array.
            w_raw = np.fromfile(outdir / "W")
            if w_raw.size == 0 or not np.all(np.isfinite(w_raw)):
                raise RuntimeError(
                    "native AMICA produced a degenerate fit (non-finite weights); "
                    "the run did not converge. Try more iterations, more data, or "
                    "fewer models."
                )
            self.output_ = loadmodout(outdir)
        return self

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """Source activations for ``X`` from the fitted model (delegates to
        ``AmicaOutput.sources``)."""
        if self.output_ is None:
            raise RuntimeError("call fit() before transform().")
        return self.output_.sources(X, model_idx)
