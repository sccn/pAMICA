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
issue #263) and the Newton preconditioner (``do_newton``, issue #264 --
float32 throughout, validated against a float64 PyTorch twin); outlier
rejection and save/load remain fast-follows.
"""

from .core import AMICAMLXNG

__all__ = ["AMICAMLXNG"]
