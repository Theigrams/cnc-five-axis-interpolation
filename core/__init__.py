"""
core - 核心算法模块

包含:
- position_spline: C³ 连续位置样条生成
- orientation_spline: C³ 连续姿态样条生成
- kinematics: 五轴逆运动学
"""

from .position_spline import PositionSpline
from .orientation_spline import OrientationSpline
from .kinematics import inverse_kinematics_ac

__all__ = ["PositionSpline", "OrientationSpline", "inverse_kinematics_ac"]
