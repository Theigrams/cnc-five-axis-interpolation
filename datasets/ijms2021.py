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
    chord_error_limit_mm: float = 0.001  # 弦误差限制 (mm)
    feedrate_command_mm_s: float = 50.0  # 指令进给速度 (mm/s)

    # 工件坐标系 (WCS) 限制
    acc_wcs_mm_s2: tuple = (500.0, 500.0, 400.0)  # 加速度 [x, y, z]
    jerk_wcs_mm_s3: tuple = (5000.0, 5000.0, 4000.0)  # 加加速度 [x, y, z]

    # 机床坐标系 (MCS) 线性轴限制
    acc_mcs_linear_mm_s2: tuple = (500.0, 500.0, 400.0)  # [X, Y, Z]
    jerk_mcs_linear_mm_s3: tuple = (5000.0, 5000.0, 4000.0)  # [X, Y, Z]

    # 机床坐标系 (MCS) 旋转轴限制
    acc_mcs_rotary_deg_s2: tuple = (300.0, 500.0)  # [A, C] (deg/s^2)
    jerk_mcs_rotary_deg_s3: tuple = (3000.0, 5000.0)  # [A, C] (deg/s^3)


# Table B.1: 扇形刀具路径离散点数据
# 格式: [Pxw, Pyw, Pzw, Oiw, Ojw, Okw]
_RAW_DATA = np.array(
    [
        [113.5608, 7.7353, -2.2093, -0.1073, 0.6249, 0.7733],
        [117.8649, -10.9501, -0.9741, -0.0030, 0.6530, 0.7573],
        [115.5029, -34.8088, 0.7796, 0.1350, 0.6488, 0.7489],
        [104.0860, -55.8401, 2.6826, 0.2639, 0.5968, 0.7578],
        [95.2257, -63.9596, 4.0735, 0.3172, 0.5518, 0.7713],
        [88.8206, -65.9723, 6.0218, 0.3295, 0.5161, 0.7906],
        [80.1628, -65.0964, 7.0315, 0.3268, 0.4780, 0.8153],
        [72.4782, -61.1095, 6.8631, 0.3137, 0.4461, 0.8382],
        [65.9858, -54.6664, 6.1430, 0.2887, 0.4164, 0.8621],
        [54.2520, -39.5681, 4.9312, 0.2170, 0.3575, 0.9084],
        [38.0390, -23.1115, 3.5361, 0.1295, 0.2618, 0.9564],
        [31.6791, -18.7113, 3.0176, 0.1055, 0.2209, 0.9696],
        [26.1926, -16.7813, 2.4549, 0.0968, 0.1849, 0.9780],
        [22.3378, -16.4864, 1.9074, 0.0979, 0.1597, 0.9823],
        [18.7698, -17.9379, 0.2587, 0.1106, 0.1379, 0.9843],
        [17.0042, -21.3334, -1.3755, 0.1335, 0.1275, 0.9828],
        [16.7631, -27.0829, -2.5738, 0.1711, 0.1277, 0.9769],
        [20.0593, -38.2107, -3.5736, 0.2390, 0.1533, 0.9588],
        [26.9912, -67.7863, -5.5703, 0.3999, 0.2013, 0.8942],
        [28.0807, -86.6480, -6.2051, 0.4879, 0.2082, 0.8477],
        [21.8131, -103.5714, -4.4425, 0.5679, 0.1822, 0.8027],
        [6.1563, -114.1796, -2.0441, 0.6287, 0.0985, 0.7714],
        [-12.6615, -117.9888, -0.8723, 0.6542, -0.0066, 0.7563],
        [-31.8162, -116.3240, 0.5145, 0.6520, -0.1172, 0.7491],
        [-49.4389, -108.7844, 2.0895, 0.6189, -0.2239, 0.7529],
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
