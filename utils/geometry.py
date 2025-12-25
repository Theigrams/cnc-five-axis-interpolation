"""
geometry - 几何计算工具函数

提供向量归一化、球坐标转换等基础几何操作。
"""

import numpy as np

EPSILON = 1e-16


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    将向量归一化为单位向量。

    Args:
        vectors: 单个向量 (n,) 或向量数组 (m, n)

    Returns:
        归一化后的单位向量，与输入形状相同
    """
    vectors = np.asarray(vectors)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / (norm + EPSILON)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norm + EPSILON)


def cartesian_to_spherical(orientation: np.ndarray) -> tuple[float, float]:
    """
    将笛卡尔坐标系下的方向向量转换为球坐标 (θ, φ)。

    参考论文 Eq.25:
        θ = arccos(o_k)  -- 极角，与z轴夹角
        φ = arctan2(o_j, o_i)  -- 方位角，在xy平面的投影与x轴夹角

    Args:
        orientation: (3,) 方向向量 [o_i, o_j, o_k]，需为单位向量

    Returns:
        theta: 极角 [0, π]
        phi: 方位角 [-π, π]

    Note:
        当 θ 接近 0 或 π 时（刀轴沿z轴方向），φ 存在奇异性。
    """
    o = normalize(orientation)
    o_i, o_j, o_k = o[0], o[1], o[2]

    # θ = arccos(o_k)，范围 [0, π]
    theta = np.arccos(np.clip(o_k, -1.0, 1.0))

    # φ = arctan2(o_j, o_i)，范围 [-π, π]
    phi = np.arctan2(o_j, o_i)

    return theta, phi


def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """
    将球坐标 (θ, φ) 转换为笛卡尔坐标系下的方向向量。

    参考论文 Eq.31:
        o_i = sin(θ) * cos(φ)
        o_j = sin(θ) * sin(φ)
        o_k = cos(θ)

    Args:
        theta: 极角 [0, π]
        phi: 方位角 [-π, π]

    Returns:
        orientation: (3,) 单位方向向量 [o_i, o_j, o_k]
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    o_i = sin_theta * cos_phi
    o_j = sin_theta * sin_phi
    o_k = cos_theta

    return np.array([o_i, o_j, o_k])


def batch_cartesian_to_spherical(orientations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    批量将笛卡尔方向向量转换为球坐标。

    Args:
        orientations: (N, 3) 方向向量数组

    Returns:
        theta: (N,) 极角数组
        phi: (N,) 方位角数组
    """
    o = normalize(orientations)
    theta = np.arccos(np.clip(o[:, 2], -1.0, 1.0))
    phi = np.arctan2(o[:, 1], o[:, 0])
    return theta, phi


def batch_spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    批量将球坐标转换为笛卡尔方向向量。

    Args:
        theta: (N,) 极角数组
        phi: (N,) 方位角数组

    Returns:
        orientations: (N, 3) 方向向量数组
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    o_i = sin_theta * cos_phi
    o_j = sin_theta * sin_phi
    o_k = cos_theta

    return np.column_stack([o_i, o_j, o_k])


def unwrap_angles(phi: np.ndarray) -> np.ndarray:
    """
    展开方位角序列，消除 ±π 处的不连续跳变。

    当方位角跨越 ±π 边界时，会产生 2π 的跳变，这会导致样条拟合失败。
    此函数通过累积相位差来消除这种不连续性。

    Args:
        phi: (N,) 方位角序列

    Returns:
        unwrapped_phi: (N,) 展开后的方位角序列
    """
    return np.unwrap(phi)


if __name__ == "__main__":
    # 测试球坐标转换
    print("=== 球坐标转换测试 ===")

    # 测试1: z轴正方向
    o1 = np.array([0.0, 0.0, 1.0])
    theta1, phi1 = cartesian_to_spherical(o1)
    o1_back = spherical_to_cartesian(theta1, phi1)
    print(f"z轴正方向: θ={np.degrees(theta1):.1f}°, φ={np.degrees(phi1):.1f}°")
    print(f"  逆转换: {o1_back}")

    # 测试2: x轴正方向
    o2 = np.array([1.0, 0.0, 0.0])
    theta2, phi2 = cartesian_to_spherical(o2)
    o2_back = spherical_to_cartesian(theta2, phi2)
    print(f"x轴正方向: θ={np.degrees(theta2):.1f}°, φ={np.degrees(phi2):.1f}°")
    print(f"  逆转换: {o2_back}")

    # 测试3: 任意方向
    o3 = normalize(np.array([1.0, 1.0, 1.0]))
    theta3, phi3 = cartesian_to_spherical(o3)
    o3_back = spherical_to_cartesian(theta3, phi3)
    print(f"对角方向: θ={np.degrees(theta3):.1f}°, φ={np.degrees(phi3):.1f}°")
    print(f"  原始: {o3}")
    print(f"  逆转换: {o3_back}")
    print(f"  误差: {np.linalg.norm(o3 - o3_back):.2e}")
