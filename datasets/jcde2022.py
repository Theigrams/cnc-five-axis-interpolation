"""
jcde2022 - JCDE 2022 双B样条刀具路径数据

数据来源:
Paper: JCDE 2022 "Optimal feedrate planning on a five-axis parametric tool path..."
Authors: Hong-Yu Ma et al.
Data Source: Appendix 1 (Flank Milling Tool Path) & Table 4 (Kinematic Limits)

数据说明:
该数据集使用双三次 B 样条 (Dual Cubic B-spline) 定义刀具路径。
- Bottom Curve P(β): 定义刀尖轨迹
- Top Curve H(β): 定义刀具轴线上的第二个点
- 刀轴方向 O(β) = (H - P) / ||H - P||
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class JCDE2022Constraints:
    """JCDE 2022 机床运动学约束参数 (Table 4)"""

    interpolation_period_s: float = 0.002  # 插补周期 (s)
    chord_error_limit_mm: float = 0.000125  # 弦误差限制 (mm), 论文中 0.125 μm

    # 线性轴限制 [X, Y, Z] (mm, mm/s, mm/s^2, mm/s^3)
    vel_linear_mm_s: tuple = (100.0, 100.0, 100.0)
    acc_linear_mm_s2: tuple = (500.0, 500.0, 500.0)
    jerk_linear_mm_s3: tuple = (3000.0, 3000.0, 3000.0)

    # 旋转轴限制 [A, C] (rad, rad/s, rad/s^2, rad/s^3)
    vel_rotary_rad_s: tuple = (0.4, 0.8)
    acc_rotary_rad_s2: tuple = (0.5, 0.5)
    jerk_rotary_rad_s3: tuple = (1.5, 1.5)


# 节点向量 (Knot vector)
_KNOT_VECTOR = np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1], dtype=np.float64)
_DEGREE = 3

# 底部曲线控制点 P (Tool Tip) - (N, 3) 格式
_CTRL_PTS_P = np.array(
    [
        [5.0, 0.0, 0.0],
        [-10.0, 20.0, 0.0],
        [10.0, 20.0, 0.0],
        [20.0, 30.0, 0.0],
        [30.0, 30.0, 0.0],
        [40.0, 30.0, 0.0],
        [50.0, 20.0, 0.0],
        [55.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# 顶部曲线控制点 H (Second Point on Axis) - (N, 3) 格式
_CTRL_PTS_H = np.array(
    [
        [0.0, 0.0, 15.0],
        [-15.0, 20.0, 15.0],
        [5.0, 25.0, 15.0],
        [15.0, 35.0, 15.0],
        [30.0, 35.0, 15.0],
        [45.0, 35.0, 15.0],
        [55.0, 25.0, 15.0],
        [60.0, 0.0, 15.0],
    ],
    dtype=np.float64,
)


def jcde2022_dual_bspline_path(
    num_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, JCDE2022Constraints]:
    """
    获取 JCDE 2022 双B样条刀具路径数据。

    通过采样双三次 B 样条曲线生成离散化路径点。

    Args:
        num_samples: 采样点数

    Returns:
        positions: (N, 3) 刀尖位置数组 (mm)
        orientations: (N, 3) 刀轴姿态数组 (归一化单位向量)
        constraints: 机床运动学约束参数
    """
    # 创建 B 样条对象
    spline_P = BSpline(_KNOT_VECTOR, _CTRL_PTS_P, _DEGREE)
    spline_H = BSpline(_KNOT_VECTOR, _CTRL_PTS_H, _DEGREE)

    # 采样参数 β ∈ [0, 1]
    beta = np.linspace(0, 1, num_samples)

    # 计算曲线上对应点
    positions = spline_P(beta)
    H_vals = spline_H(beta)

    # 计算刀轴矢量 O = H - P 并归一化
    orientations = H_vals - positions
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    orientations = orientations / norms

    constraints = JCDE2022Constraints()

    return positions, orientations, constraints


def get_bspline_curves() -> tuple[BSpline, BSpline]:
    """
    获取原始 B 样条曲线对象。

    Returns:
        spline_P: 刀尖轨迹 B 样条
        spline_H: 刀具轴线第二点 B 样条
    """
    spline_P = BSpline(_KNOT_VECTOR, _CTRL_PTS_P, _DEGREE)
    spline_H = BSpline(_KNOT_VECTOR, _CTRL_PTS_H, _DEGREE)
    return spline_P, spline_H


if __name__ == "__main__":
    positions, orientations, constraints = jcde2022_dual_bspline_path(100)
    print("=== JCDE 2022 双B样条刀具路径 ===")
    print(f"点数: {len(positions)}")
    print(f"位置范围: X[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] mm")
    print(f"          Y[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] mm")
    print(f"          Z[{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}] mm")
    print(f"\n刀轴归一化验证: {np.allclose(np.linalg.norm(orientations, axis=1), 1.0)}")
