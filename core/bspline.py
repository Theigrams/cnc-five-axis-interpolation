"""
bspline - 公共B样条工具函数

提供五次B样条拟合的通用工具，供 position_spline 和 orientation_spline 共用。

实现:
1. 参数化方法 (Eq.4, Eq.28)
2. 节点向量计算 (Eq.5)
3. 基函数矩阵构造 (Eq.6) - 优化版本
4. 通用B样条插值拟合
"""

import numpy as np
from scipy.interpolate import BSpline


def centripetal_parameterization(points: np.ndarray) -> np.ndarray:
    """
    向心参数化方法 (Eq.4)。

    使用相邻点距离的平方根来分配参数值，相比弦长参数化能产生更好的拟合效果。

    Args:
        points: (N, D) 离散点坐标

    Returns:
        u_bar: (N,) 参数值数组, u_bar[0]=0, u_bar[-1]=1
    """
    N = len(points)
    if N < 2:
        return np.array([0.0]) if N == 1 else np.array([])

    sqrt_dists = np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1))
    d = np.sum(sqrt_dists)

    if d < 1e-12:
        return np.linspace(0, 1, N)

    u_bar = np.zeros(N)
    u_bar[1:] = np.cumsum(sqrt_dists) / d
    u_bar[-1] = 1.0

    return u_bar


def angular_parameterization(orientations: np.ndarray) -> np.ndarray:
    """
    角度参数化方法 (Eq.28)。

    使用相邻姿态向量之间角度变化的平方根来分配参数值。

    Args:
        orientations: (N, 3) 单位刀轴向量

    Returns:
        w_bar: (N,) 参数值数组, w_bar[0]=0, w_bar[-1]=1
    """
    N = len(orientations)
    if N < 2:
        return np.array([0.0]) if N == 1 else np.array([])

    dots = np.sum(orientations[:-1] * orientations[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    sqrt_angles = np.sqrt(angles)
    d = np.sum(sqrt_angles)

    if d < 1e-12:
        return np.linspace(0, 1, N)

    w_bar = np.zeros(N)
    w_bar[1:] = np.cumsum(sqrt_angles) / d
    w_bar[-1] = 1.0

    return w_bar


def compute_knot_vector(params: np.ndarray, degree: int) -> np.ndarray:
    """
    计算B样条节点向量 (Eq.5)。

    使用均值法从参数值计算内部节点。

    Args:
        params: (N,) 参数值数组
        degree: 样条阶数 (n=5 for quintic)

    Returns:
        knots: (N + degree + 1,) 节点向量
    """
    N = len(params) - 1
    n = degree
    num_knots = N + n + 2

    knots = np.zeros(num_knots)
    knots[:n + 1] = 0.0
    knots[-(n + 1):] = 1.0

    for j in range(1, N - n + 1):
        knots[j + n] = np.mean(params[j:j + n])

    return knots


def bspline_basis_matrix(params: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    计算B样条基函数矩阵 (Eq.6) - 优化版本。

    使用向量化计算代替 O(N²) 循环构造。

    Args:
        params: (N,) 参数值
        knots: 节点向量
        degree: 样条阶数

    Returns:
        Phi: (N, N) 基函数矩阵
    """
    N = len(params)
    coeffs = np.eye(N)
    basis_spline = BSpline(knots, coeffs, degree)
    return basis_spline(params)


def fit_interpolating_bspline(
    points: np.ndarray,
    params: np.ndarray,
    degree: int = 5,
) -> tuple[BSpline, np.ndarray]:
    """
    拟合插值B样条曲线。

    Args:
        points: (N, D) 待拟合点
        params: (N,) 参数值
        degree: 样条阶数

    Returns:
        spline: scipy BSpline 对象
        knots: 节点向量
    """
    knots = compute_knot_vector(params, degree)
    Phi = bspline_basis_matrix(params, knots, degree)
    control_points = np.linalg.solve(Phi, points)
    spline = BSpline(knots, control_points, degree)
    return spline, knots


def fit_quintic_bspline(points: np.ndarray) -> tuple[BSpline, np.ndarray, np.ndarray]:
    """
    拟合五次B样条曲线 (Eq.1-7)。

    使用向心参数化拟合五次B样条。

    Args:
        points: (N, D) 离散点

    Returns:
        spline: scipy BSpline 对象
        knots: 节点向量
        params: 参数值
    """
    params = centripetal_parameterization(points)
    spline, knots = fit_interpolating_bspline(points, params, degree=5)
    return spline, knots, params


if __name__ == "__main__":
    print("=== B样条工具测试 ===")

    points = np.array([
        [0, 0, 0],
        [1, 2, 0],
        [3, 3, 0],
        [4, 2, 0],
        [5, 0, 0],
        [6, -1, 0],
        [7, 0, 0],
    ], dtype=float)

    spline, knots, params = fit_quintic_bspline(points)
    print(f"点数: {len(points)}")
    print(f"节点数: {len(knots)}")
    print(f"参数范围: [{params[0]:.3f}, {params[-1]:.3f}]")

    errors = []
    for i, u in enumerate(params):
        interp = spline(u)
        error = np.linalg.norm(interp - points[i])
        errors.append(error)

    print(f"插值误差: max={max(errors):.2e}, mean={np.mean(errors):.2e}")
