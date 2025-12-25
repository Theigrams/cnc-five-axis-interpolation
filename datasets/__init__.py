"""
datasets - 测试数据集

包含:
- ijms2021: IJMS 2021 扇形刀具路径
- jcde2022: JCDE 2022 双B样条刀具路径
"""

from .ijms2021 import ijms2021_fan_shaped_path, IJMS2021Constraints
from .jcde2022 import jcde2022_dual_bspline_path, JCDE2022Constraints

__all__ = [
    "ijms2021_fan_shaped_path",
    "IJMS2021Constraints",
    "jcde2022_dual_bspline_path",
    "JCDE2022Constraints",
]
