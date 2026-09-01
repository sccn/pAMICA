"""Legacy NumPy reference implementation of AMICA.

Topic-named modules (``core``, ``pdf``, ``data``, ``load``, ``viz``,
``utils``, ``cli``), renamed from the old ``pamica.py``/``amica_*.py`` sprawl in
issue #34. The scikit-learn-style :class:`~pamica.numpy_impl.core.AMICA` is also
exposed at the package root as ``pamica.AMICA_NumPy``.
"""

from .core import AMICA

__all__ = ["AMICA"]
