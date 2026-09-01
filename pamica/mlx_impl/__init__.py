"""MLX backend for pamica (Apple-Silicon GPU).

Mirrors ``torch_impl`` but targets Apple's MLX array framework. MLX is an
*optional* dependency (Apple Silicon only); importing this subpackage requires
``mlx`` to be installed, so the top-level ``pamica`` package never imports it
eagerly. Install with ``uv pip install mlx`` (or the ``mlx`` extra).

The backend (:class:`AMICAMLXNG`) runs the natural-gradient EM E/M-step on the
Apple GPU in float32 (Apple GPUs have no FP64), with the small per-iteration
linear algebra on MLX's CPU stream (issues #76/#81, epic #74 Phase C/D). It
supports single- and multi-model, all five ``amica15.f90`` source-density
families (``pdftype`` 0/1/2/3/4, including the extended-Infomax adaptive
switcher, issue #265), natural gradient, component sharing (``share_comps``,
issue #263), the Newton preconditioner (``do_newton``, issue #264 -- float32
throughout, validated against a float64 PyTorch twin), source extraction and
persistence (``transform``, ``state_dict``/``.npz`` ``save``/``load``, issue
#287), the best-iterate safeguard (``keep_best``, issue #288), outlier
rejection (``do_reject``, issue #123's mechanism), the EEGLAB
``write_amica_output`` export, and the MIR/PMI diagnostics (issue #137) --
epic #278 Phase 3 (issue #289). The EEGLAB back-projected-variance
component order (``variance_order``) landed in the epic's post-Phase-3
polish round, ahead of merge to ``dev``, closing the one accessor gap
Phase 3 left open: every ``AMICATorchNG``-supported feature this backend
can support (float32 GPU limits aside) is now ported, so epic #278 is
complete.
"""

from .core import PDFTYPE_NAMES, AMICAMLXNG

__all__ = ["AMICAMLXNG", "PDFTYPE_NAMES"]
