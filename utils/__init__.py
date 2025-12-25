"""
utils - 工具函数模块

包含:
- geometry: 几何计算工具
- integrals: 数值积分工具
"""

from .geometry import normalize, spherical_to_cartesian, cartesian_to_spherical
from .integrals import adaptive_simpson, arc_length_integral

__all__ = [
    "normalize",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "adaptive_simpson",
    "arc_length_integral",
]
