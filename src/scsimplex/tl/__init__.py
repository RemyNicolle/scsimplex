"""Tools namespace for scsimplex."""

from ._classifier import map_multinomial_nb
from ._multiplets import detect_multiplets

__all__ = ["detect_multiplets", "map_multinomial_nb"]
