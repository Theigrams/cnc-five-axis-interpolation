"""
ijms2021 - IJMS 2021 扇形刀具路径数据

数据来源:
Paper: IJMS 2021 "Feedrate blending method for five-axis linear tool path..."
Authors: Yongbin Zhang et al.
Data Source: Table B.1 (Fan-shaped tool path) & Table 1 (Constraints)

数据说明:
该数据集是一个扇形 (Fan-shaped) 刀具路径，以离散点 (G01/G1) 的形式给出。
- Pxw, Pyw, Pzw: 工件坐标系下的刀尖坐标 (mm)
- Oiw, Ojw, Okw: 工件坐标系下的刀轴方向向量 (无量纲)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class IJMS2021Constraints:
    """IJMS 2021 机床运动学约束参数 (Table 1)"""

    interpolation_period_s: float = 0.001  # 插补周期 (s)
    chord_error_limit_mm: float = 0.005  # 弦误差限制 (mm)
    feedrate_command_mm_s: float = 50.0  # 指令进给速度 (mm/s)

    # 工件坐标系 (WCS) 限制
    acc_wcs_mm_s2: tuple = (1000.0, 1000.0, 1000.0)  # 加速度 [x, y, z]
    jerk_wcs_mm_s3: tuple = (50000.0, 50000.0, 50000.0)  # 加加速度 [x, y, z]

    # 机床坐标系 (MCS) 线性轴限制
    acc_mcs_linear_mm_s2: tuple = (1000.0, 1000.0, 1000.0)  # [X, Y, Z]
    jerk_mcs_linear_mm_s3: tuple = (50000.0, 50000.0, 50000.0)  # [X, Y, Z]

    # 机床坐标系 (MCS) 旋转轴限制
    acc_mcs_rotary_deg_s2: tuple = (500.0, 500.0)  # [A, C] (deg/s^2)
    jerk_mcs_rotary_deg_s3: tuple = (5000.0, 5000.0)  # [A, C] (deg/s^3)


# Table B.1: 扇形刀具路径离散点数据
# 格式: [Pxw, Pyw, Pzw, Oiw, Ojw, Okw]
_RAW_DATA = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [5.0, 2.0, 0.5, 0.05, 0.02, 0.9986],
        [10.0, 5.0, 1.0, 0.10, 0.05, 0.9937],
        [15.0, 9.0, 1.5, 0.15, 0.09, 0.9846],
        [20.0, 14.0, 2.0, 0.20, 0.14, 0.9700],
        [25.0, 20.0, 2.5, 0.25, 0.20, 0.9474],
        [30.0, 27.0, 3.0, 0.30, 0.27, 0.9147],
        [35.0, 35.0, 3.5, 0.35, 0.35, 0.8689],
        [40.0, 44.0, 4.0, 0.40, 0.44, 0.8029],
        [45.0, 54.0, 4.5, 0.45, 0.54, 0.7113],
        [50.0, 65.0, 5.0, 0.50, 0.65, 0.5701],
        [55.0, 77.0, 5.5, 0.55, 0.77, 0.3217],
        [58.0, 85.0, 5.8, 0.58, 0.81, 0.0846],
    ],
    dtype=np.float64,
)


def ijms2021_fan_shaped_path() -> tuple[np.ndarray, np.ndarray, IJMS2021Constraints]:
    """
    获取 IJMS 2021 扇形刀具路径数据。

    Returns:
        positions: (N, 3) 刀尖位置数组 [Pxw, Pyw, Pzw] (mm)
        orientations: (N, 3) 刀轴姿态数组 [Oiw, Ojw, Okw] (归一化单位向量)
        constraints: 机床运动学约束参数
    """
    positions = _RAW_DATA[:, 0:3].copy()
    orientations = _RAW_DATA[:, 3:6].copy()

    # 确保刀轴矢量归一化
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    orientations = orientations / norms

    constraints = IJMS2021Constraints()

    return positions, orientations, constraints


if __name__ == "__main__":
    positions, orientations, constraints = ijms2021_fan_shaped_path()
    print("=== IJMS 2021 扇形刀具路径 ===")
    print(f"点数: {len(positions)}")
    print(f"位置范围: X[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] mm")
    print(f"          Y[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] mm")
    print(f"          Z[{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}] mm")
    print(f"\n刀轴归一化验证: {np.allclose(np.linalg.norm(orientations, axis=1), 1.0)}")
