"""
cnc_five_axis_interpolation - 五轴CNC平滑轨迹生成算法库

基于论文: Yuen, Zhang, Altintas (2013)
"Smooth trajectory generation for five-axis machine tools"

该库实现了 C³ 连续的五轴样条插补技术，通过解耦刀尖位置和刀具姿态的插补，
实现平滑的五轴轨迹生成。
"""

from .algorithm import FiveAxisPath

__version__ = "0.1.0"
__all__ = ["FiveAxisPath"]
