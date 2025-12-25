"""
kinematics - 五轴逆运动学

基于论文 Section 4, Eq.42: A-C 配置逆运动学变换

将工件坐标系下的刀尖位置和刀轴姿态转换为机床坐标系下的五轴坐标 (X, Y, Z, A, C)。
"""

import numpy as np


def inverse_kinematics_ac(
    P: np.ndarray,
    O: np.ndarray,
    L_ac_z: float = 70.0,
    L_Tya_z: float = 150.0,
) -> tuple[np.ndarray, float, float]:
    """
    A-C 配置五轴逆运动学变换。

    参考论文 Eq.42:
        A = arccos(O_k)
        C = arctan2(O_i, O_j)
        X = -cos(C)*P_x - sin(C)*P_y
        Y = cos(A)*sin(C)*P_x - cos(A)*cos(C)*P_y - sin(A)*P_z - sin(A)*L_ac_z
        Z = sin(A)*sin(C)*P_x - sin(A)*cos(C)*P_y + cos(A)*P_z + cos(A)*L_ac_z + L_Tya_z

    Args:
        P: (3,) 工件坐标系下的刀尖位置 [P_x, P_y, P_z]
        O: (3,) 工件坐标系下的刀轴姿态 [O_i, O_j, O_k]，单位向量
        L_ac_z: A轴到C轴的Z方向偏移 (mm)
        L_Tya_z: 主轴到A轴的Z方向偏移 (mm)

    Returns:
        XYZ: (3,) 机床线性轴坐标 [X, Y, Z]
        A: A轴角度 (rad)
        C: C轴角度 (rad)
    """
    P_x, P_y, P_z = P
    O_i, O_j, O_k = O

    # 计算旋转轴角度
    A = np.arccos(np.clip(O_k, -1.0, 1.0))
    C = np.arctan2(O_i, O_j)

    # 预计算三角函数
    cos_A, sin_A = np.cos(A), np.sin(A)
    cos_C, sin_C = np.cos(C), np.sin(C)

    # 计算线性轴位置
    X = -cos_C * P_x - sin_C * P_y
    Y = cos_A * sin_C * P_x - cos_A * cos_C * P_y - sin_A * P_z - sin_A * L_ac_z
    Z = sin_A * sin_C * P_x - sin_A * cos_C * P_y + cos_A * P_z + cos_A * L_ac_z + L_Tya_z

    return np.array([X, Y, Z]), A, C


def batch_inverse_kinematics_ac(
    positions: np.ndarray,
    orientations: np.ndarray,
    L_ac_z: float = 70.0,
    L_Tya_z: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    批量计算A-C配置逆运动学。

    Args:
        positions: (N, 3) 刀尖位置数组
        orientations: (N, 3) 刀轴姿态数组

    Returns:
        XYZ: (N, 3) 机床线性轴坐标
        A: (N,) A轴角度
        C: (N,) C轴角度
    """
    N = len(positions)
    XYZ = np.zeros((N, 3))
    A = np.zeros(N)
    C = np.zeros(N)

    for i in range(N):
        XYZ[i], A[i], C[i] = inverse_kinematics_ac(
            positions[i], orientations[i], L_ac_z, L_Tya_z
        )

    return XYZ, A, C


if __name__ == "__main__":
    # 测试逆运动学
    print("=== 逆运动学测试 ===")

    # 刀轴沿z轴正方向 (A=0)
    P1 = np.array([10.0, 20.0, 30.0])
    O1 = np.array([0.0, 0.0, 1.0])
    XYZ1, A1, C1 = inverse_kinematics_ac(P1, O1)
    print(f"刀轴沿z轴:")
    print(f"  输入: P={P1}, O={O1}")
    print(f"  输出: XYZ={XYZ1}, A={np.degrees(A1):.1f}°, C={np.degrees(C1):.1f}°")

    # 刀轴倾斜45度
    O2 = np.array([0.0, np.sin(np.radians(45)), np.cos(np.radians(45))])
    XYZ2, A2, C2 = inverse_kinematics_ac(P1, O2)
    print(f"\n刀轴倾斜45°:")
    print(f"  输入: P={P1}, O={O2}")
    print(f"  输出: XYZ={XYZ2}, A={np.degrees(A2):.1f}°, C={np.degrees(C2):.1f}°")
