"""
integrals - 数值积分工具函数

提供自适应数值积分方法，用于计算B样条曲线的弧长。

参考论文 Eq.8-10: 使用自适应Simpson法计算弧长积分。
"""

from typing import Callable

import numpy as np


def simpson(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Simpson法则计算定积分。

    参考论文 Eq.9:
        l(a,b) ≈ (b-a)/6 * [f(a) + 4f((a+b)/2) + f(b)]

    Args:
        f: 被积函数
        a: 积分下限
        b: 积分上限

    Returns:
        积分近似值
    """
    c = (a + b) / 2
    return (b - a) / 6 * (f(a) + 4 * f(c) + f(b))


def adaptive_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    intervals: list | None = None,
) -> float:
    """
    自适应Simpson积分法。

    参考论文 Eq.10: 当误差超过容差时，递归细分区间。
    |l(a1,b1) + l(a2,b2) - l(a,b)| < ε_l

    Args:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        tol: 误差容差
        intervals: 可选，用于记录最终的积分区间

    Returns:
        积分值
    """
    c = (a + b) / 2
    S_ab = simpson(f, a, b)
    S_ac = simpson(f, a, c)
    S_cb = simpson(f, c, b)

    if abs(S_ac + S_cb - S_ab) < tol:
        if intervals is not None:
            intervals.append((a, b, S_ac + S_cb))
        return S_ac + S_cb
    else:
        left = adaptive_simpson(f, a, c, tol / 2, intervals)
        right = adaptive_simpson(f, c, b, tol / 2, intervals)
        return left + right


def arc_length_integral(
    derivative_func: Callable[[float], np.ndarray],
    a: float,
    b: float,
    tol: float = 1e-8,
) -> float:
    """
    计算参数曲线的弧长积分。

    参考论文 Eq.8:
        l(b) - l(a) = ∫_a^b ||P'(u)|| du

    Args:
        derivative_func: 曲线的导数函数，返回 (n,) 向量
        a: 参数下限
        b: 参数上限
        tol: 积分误差容差

    Returns:
        弧长值
    """

    def integrand(u: float) -> float:
        deriv = derivative_func(u)
        return np.linalg.norm(deriv)

    return adaptive_simpson(integrand, a, b, tol)


def compute_arc_length_table(
    derivative_func: Callable[[float], np.ndarray],
    n_samples: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算参数-弧长对应表。

    将参数区间 [0, 1] 均匀采样，计算每个采样点对应的累积弧长。

    Args:
        derivative_func: 曲线的导数函数
        n_samples: 采样点数
        tol: 积分误差容差

    Returns:
        u_samples: (n_samples,) 参数值
        l_samples: (n_samples,) 对应的累积弧长
    """
    u_samples = np.linspace(0, 1, n_samples)
    l_samples = np.zeros(n_samples)

    for i in range(1, n_samples):
        segment_length = arc_length_integral(derivative_func, u_samples[i - 1], u_samples[i], tol)
        l_samples[i] = l_samples[i - 1] + segment_length

    return u_samples, l_samples


if __name__ == "__main__":
    # 测试自适应积分
    print("=== 自适应积分测试 ===")

    # 测试1: e^x 在 [0, 1] 上的积分，解析解为 e - 1
    def exp_func(x):
        return np.exp(x)

    result = adaptive_simpson(exp_func, 0, 1)
    exact = np.e - 1
    print(f"∫e^x dx from 0 to 1:")
    print(f"  计算值: {result:.10f}")
    print(f"  精确值: {exact:.10f}")
    print(f"  误差: {abs(result - exact):.2e}")

    # 测试2: 单位圆弧长 (四分之一圆)
    def circle_derivative(t):
        # 参数化: x = cos(πt/2), y = sin(πt/2)
        # 导数: dx/dt = -π/2 * sin(πt/2), dy/dt = π/2 * cos(πt/2)
        return np.array([-np.pi / 2 * np.sin(np.pi * t / 2), np.pi / 2 * np.cos(np.pi * t / 2)])

    arc_len = arc_length_integral(circle_derivative, 0, 1)
    exact_arc = np.pi / 2  # 四分之一圆周长
    print(f"\n四分之一圆弧长:")
    print(f"  计算值: {arc_len:.10f}")
    print(f"  精确值: {exact_arc:.10f}")
    print(f"  误差: {abs(arc_len - exact_arc):.2e}")
