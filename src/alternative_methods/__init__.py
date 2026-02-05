"""
Alternative Reserving Methods

This module contains alternative reserving methods beyond standard chain-ladder:

- Cape Cod: Uses on-level premium and expected loss ratios
"""

from .cape_cod import CapeCod

__all__ = [
    'CapeCod'
]
