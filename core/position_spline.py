"""
position_spline - C³ 连续位置样条生成

基于论文 Section 2: C³ tool tip position spline generation

实现:
1. 五次B样条拟合刀尖位置 (Eq.1-7)
2. 9阶多项式进给校正 (Eq.11-22)
3. 自适应细分 (Eq.23)
"""

import numpy as np
from scipy.interpolate import BSpline

from ..utils.integrals import adaptive_simpson
from .bspline import fit_quintic_bspline


class FeedCorrectionPolynomial:
    """
    9阶进给校正多项式 (Eq.11-22)。

    将弧长 l 映射到样条参数 u，同时满足 C³ 边界条件。
    支持分段拟合，每段映射到 [u_start, u_end] 范围。
    """

    def __init__(self, coeffs: np.ndarray, total_length: float, u_start: float = 0.0, u_end: float = 1.0):
        """
        Args:
            coeffs: (10,) 归一化多项式系数 [a_9, a_8, ..., a_0]
            total_length: 段弧长 S
            u_start: 段起始 u 值
            u_end: 段结束 u 值
        """
        self.coeffs = coeffs
        self.S = total_length
        self.u_start = u_start
        self.u_end = u_end

    def __call__(self, l: float | np.ndarray) -> float | np.ndarray:
        """计算 u(l)，映射到 [u_start, u_end]"""
        sigma = np.asarray(l) / self.S
        # 使用 Horner 法计算归一化多项式 (输出 [0, 1])
        u_normalized = np.zeros_like(sigma)
        for a in self.coeffs:
            u_normalized = u_normalized * sigma + a
        # 映射到实际 u 范围
        return self.u_start + u_normalized * (self.u_end - self.u_start)

    def derivative(self, l: float | np.ndarray, order: int = 1) -> float | np.ndarray:
        """计算 du/dl 的各阶导数"""
        sigma = np.asarray(l) / self.S

        if order == 1:
            # du/dl = (1/S) * (9*a_9*σ^8 + 8*a_8*σ^7 + ... + a_1)
            powers = np.arange(9, 0, -1)
            derivs = self.coeffs[:-1] * powers
            result = np.zeros_like(sigma)
            for d in derivs:
                result = result * sigma + d
            return result / self.S

        elif order == 2:
            # d²u/dl² = (1/S²) * (72*a_9*σ^7 + 56*a_8*σ^6 + ... + 2*a_2)
            powers = np.arange(9, 1, -1)
            powers2 = np.arange(8, 0, -1)
            derivs = self.coeffs[:-2] * powers[:-1] * powers2
            result = np.zeros_like(sigma)
            for d in derivs:
                result = result * sigma + d
            return result / (self.S**2)

        elif order == 3:
            # d³u/dl³
            powers = np.arange(9, 2, -1)
            powers2 = np.arange(8, 1, -1)
            powers3 = np.arange(7, 0, -1)
            derivs = self.coeffs[:-3] * powers[:-2] * powers2[:-1] * powers3
            result = np.zeros_like(sigma)
            for d in derivs:
                result = result * sigma + d
            return result / (self.S**3)

        else:
            raise ValueError(f"Order {order} not supported")


def compute_derivatives_at_u(spline: BSpline, u: float, S: float) -> np.ndarray:
    """
    计算指定 u 处的导数值 (Eq.16)。

    使用 B样条解析导数计算 f(u) = ||P'(u)|| 及其导数。

    Args:
        spline: 位置 B 样条
        u: 参数值
        S: 段弧长

    Returns:
        [u_l * S, u_ll * S², u_lll * S³] 用于边界约束
    """
    # 避免精确边界点的数值问题
    u_eval = max(1e-8, min(u, 1 - 1e-8))

    # 获取解析导数样条
    spline_d1 = spline.derivative(1)  # P'(u)
    spline_d2 = spline.derivative(2)  # P''(u)
    spline_d3 = spline.derivative(3)  # P'''(u)

    P1 = spline_d1(u_eval)  # P'(u)
    P2 = spline_d2(u_eval)  # P''(u)
    P3 = spline_d3(u_eval)  # P'''(u)

    # f(u) = ||P'(u)||
    f_val = np.linalg.norm(P1)
    if f_val < 1e-12:
        return np.array([S, 0.0, 0.0])

    # f'(u) = (P' · P'') / ||P'||
    fp_val = np.dot(P1, P2) / f_val

    # f''(u) = (||P''||² + P' · P''' - (P' · P'')² / ||P'||²) / ||P'||
    fpp_val = (np.dot(P2, P2) + np.dot(P1, P3) - fp_val**2) / f_val

    # u_l = 1/f (Eq.16)
    u_l = 1.0 / f_val

    # u_ll = -f'/f³ (Eq.16)
    u_ll = -fp_val / (f_val**3)

    # u_lll = (3f'² - f''f) / f⁵ (Eq.16)
    u_lll = (3 * fp_val**2 - fpp_val * f_val) / (f_val**5)

    return np.array([u_l * S, u_ll * S**2, u_lll * S**3])


def fit_feed_correction_polynomial(
    arc_lengths: np.ndarray,
    u_values: np.ndarray,
    spline: BSpline,
    u_start: float | None = None,
    u_end: float | None = None,
) -> FeedCorrectionPolynomial:
    """
    拟合9阶进给校正多项式 (Eq.11-22)。

    使用约束最小二乘法，满足 C³ 边界条件。
    支持分段拟合：多项式在归一化域 [0,1] 上拟合，输出映射到 [u_start, u_end]。

    Args:
        arc_lengths: (M,) 累积弧长 [0, l_1, ..., l_M=S]（相对于段起点）
        u_values: (M,) 对应的参数值
        spline: 位置 B 样条
        u_start: 段起始 u 值，默认使用 u_values[0]
        u_end: 段结束 u 值，默认使用 u_values[-1]

    Returns:
        FeedCorrectionPolynomial 对象
    """
    if u_start is None:
        u_start = u_values[0]
    if u_end is None:
        u_end = u_values[-1]

    S = arc_lengths[-1]
    sigma = arc_lengths / S  # 归一化弧长

    # 将 u_values 归一化到 [0, 1]
    u_range = u_end - u_start
    if u_range < 1e-12:
        # 退化情况：段太短
        coeffs = np.zeros(10)
        coeffs[9] = 0.0  # a_0 = 0
        return FeedCorrectionPolynomial(coeffs, S, u_start, u_end)

    u_normalized = (u_values - u_start) / u_range

    # 构造 Vandermonde 矩阵 Φ (Eq.13)
    powers = np.arange(9, -1, -1)
    Phi = sigma[:, np.newaxis] ** powers

    # 计算边界导数 (Eq.16)
    derivs_0 = compute_derivatives_at_u(spline, u_start, S)
    derivs_1 = compute_derivatives_at_u(spline, u_end, S)

    # 导数需要按 u_range 缩放（链式法则）
    derivs_0_scaled = derivs_0 / u_range
    derivs_1_scaled = derivs_1 / u_range

    # 构造约束矩阵 Ω (Eq.18)
    # 8 个约束: u(0), u'(0), u''(0), u'''(0), u(1), u'(1), u''(1), u'''(1)
    Omega = np.zeros((8, 10))

    # u(0) = 0: a_0 = 0 -> [0,0,0,0,0,0,0,0,0,1]
    Omega[0, 9] = 1

    # u'(0) = u_l(0)*S: a_1 = u_l(0)*S -> [0,0,0,0,0,0,0,0,1,0]
    Omega[1, 8] = 1

    # u''(0) = u_ll(0)*S²: 2*a_2 = u_ll(0)*S² -> [0,0,0,0,0,0,0,2,0,0]
    Omega[2, 7] = 2

    # u'''(0) = u_lll(0)*S³: 6*a_3 = u_lll(0)*S³ -> [0,0,0,0,0,0,6,0,0,0]
    Omega[3, 6] = 6

    # u(1) = 1: sum(a_i) = 1 -> [1,1,1,1,1,1,1,1,1,1]
    Omega[4, :] = 1

    # u'(1): 9*a_9 + 8*a_8 + ... + a_1 = u_l(1)*S
    Omega[5, :] = np.arange(9, -1, -1)

    # u''(1): 72*a_9 + 56*a_8 + ... + 2*a_2 = u_ll(1)*S²
    coeff_2nd = np.array([72, 56, 42, 30, 20, 12, 6, 2, 0, 0])
    Omega[6, :] = coeff_2nd

    # u'''(1): 504*a_9 + 336*a_8 + ... + 6*a_3 = u_lll(1)*S³
    coeff_3rd = np.array([504, 336, 210, 120, 60, 24, 6, 0, 0, 0])
    Omega[7, :] = coeff_3rd

    # 右侧向量 η（归一化域）
    eta = np.array(
        [
            0,  # u(0) = 0
            derivs_0_scaled[0],  # u'(0)*S
            derivs_0_scaled[1],  # u''(0)*S²
            derivs_0_scaled[2],  # u'''(0)*S³
            1,  # u(1) = 1
            derivs_1_scaled[0],  # u'(1)*S
            derivs_1_scaled[1],  # u''(1)*S²
            derivs_1_scaled[2],  # u'''(1)*S³
        ]
    )

    # 构造 KKT 系统 (Eq.22)
    PhiTPhi = Phi.T @ Phi
    PhiTu = Phi.T @ u_normalized

    n_constraints = 8
    n_coeffs = 10

    KKT = np.zeros((n_coeffs + n_constraints, n_coeffs + n_constraints))
    KKT[:n_coeffs, :n_coeffs] = PhiTPhi
    KKT[:n_coeffs, n_coeffs:] = Omega.T
    KKT[n_coeffs:, :n_coeffs] = Omega

    rhs = np.zeros(n_coeffs + n_constraints)
    rhs[:n_coeffs] = PhiTu
    rhs[n_coeffs:] = eta

    # 求解（带条件数检查）
    try:
        cond = np.linalg.cond(KKT)
        if cond > 1e12:
            solution = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        else:
            solution = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    coeffs = solution[:n_coeffs]

    return FeedCorrectionPolynomial(coeffs, S, u_start, u_end)


class PositionSpline:
    """
    C³ 连续位置样条。

    将离散刀尖位置拟合为五次B样条，并通过9阶多项式实现弧长参数化。
    """

    def __init__(self, points: np.ndarray, mse_tolerance: float = 1e-6):
        """
        初始化位置样条。

        Args:
            points: (N, 3) 刀尖位置点
            mse_tolerance: 进给校正多项式的MSE容差
        """
        self.points = np.asarray(points)
        self.mse_tolerance = mse_tolerance
        self.N = len(points)

        self.spline: BSpline | None = None
        self.knots: np.ndarray | None = None
        self.u_bar: np.ndarray | None = None
        self.feed_corrections: list[tuple[float, float, FeedCorrectionPolynomial]] = []
        self.length: float = 0.0

        self._arc_lengths: np.ndarray | None = None
        self._u_values: np.ndarray | None = None

    def fit(self):
        """拟合五次B样条和进给校正多项式。返回 self 以支持链式调用。"""
        # Step 1: 拟合五次 B 样条
        self.spline, self.knots, self.u_bar = fit_quintic_bspline(self.points)

        # Step 2: 计算弧长表
        self._compute_arc_length_table()

        # Step 3: 自适应拟合进给校正多项式
        self._fit_feed_correction_adaptive()

        return self

    def _compute_arc_length_table(self):
        """使用自适应Simpson积分计算弧长表 (Eq.8-10)。"""
        spline_deriv = self.spline.derivative()

        def arc_length_derivative(u):
            return np.linalg.norm(spline_deriv(u))

        # 收集积分过程中的区间
        intervals = []
        self.length = adaptive_simpson(arc_length_derivative, 0.0, 1.0, tol=1e-9, intervals=intervals)

        # 从区间构建弧长表
        u_values = [0.0]
        arc_lengths = [0.0]
        cumulative = 0.0

        for _, end, length in intervals:
            cumulative += length
            u_values.append(end)
            arc_lengths.append(cumulative)

        self._u_values = np.array(u_values)
        self._arc_lengths = np.array(arc_lengths)

    def _fit_feed_correction_adaptive(self):
        """自适应拟合进给校正多项式 (Eq.23)。"""
        self.feed_corrections = []
        self._fit_segment(0, len(self._arc_lengths) - 1)

    def _fit_segment(self, start_idx: int, end_idx: int):
        """递归拟合单个段的进给校正多项式。"""
        arc_lengths = self._arc_lengths[start_idx : end_idx + 1]
        u_values = self._u_values[start_idx : end_idx + 1]

        # 如果点数不足，直接拟合
        if len(arc_lengths) <= 10:
            poly = fit_feed_correction_polynomial(arc_lengths - arc_lengths[0], u_values, self.spline)
            self.feed_corrections.append((arc_lengths[0], arc_lengths[-1], poly))
            return

        # 尝试拟合
        poly = fit_feed_correction_polynomial(arc_lengths - arc_lengths[0], u_values, self.spline)

        # 计算 MSE (Eq.23)
        predicted = poly(arc_lengths - arc_lengths[0])
        mse = np.mean((u_values - predicted) ** 2)

        if mse < self.mse_tolerance:
            self.feed_corrections.append((arc_lengths[0], arc_lengths[-1], poly))
        else:
            # 分割并递归
            mid_idx = (start_idx + end_idx) // 2
            self._fit_segment(start_idx, mid_idx)
            self._fit_segment(mid_idx, end_idx)

    def get_u_from_length(self, l: float | np.ndarray) -> float | np.ndarray:
        """根据弧长获取样条参数 u。支持标量和数组输入。"""
        scalar_input = np.isscalar(l)
        l = np.atleast_1d(np.clip(l, 0, self.length))

        u = np.zeros_like(l)
        segment_ends = np.array([seg[1] for seg in self.feed_corrections])

        # 使用 searchsorted 找到每个 l 所属的段
        indices = np.searchsorted(segment_ends, l, side='left')
        indices = np.clip(indices, 0, len(self.feed_corrections) - 1)

        for i, (l_start, _, poly) in enumerate(self.feed_corrections):
            mask = indices == i
            if np.any(mask):
                u[mask] = poly(l[mask] - l_start)

        return float(u[0]) if scalar_input else u

    def evaluate(self, l: float) -> np.ndarray:
        """
        在弧长 l 处评估位置。

        Args:
            l: 弧长参数

        Returns:
            (3,) 位置向量
        """
        u = self.get_u_from_length(l)
        return self.spline(u)

    def evaluate_batch(self, l_values: np.ndarray) -> np.ndarray:
        """
        批量在弧长处评估位置（向量化版本）。

        Args:
            l_values: (M,) 弧长数组

        Returns:
            (M, 3) 位置数组
        """
        u_values = self.get_u_from_length(l_values)
        return self.spline(u_values)


if __name__ == "__main__":
    from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

    positions, _, _ = ijms2021_fan_shaped_path()

    print("=== 位置样条测试 ===")
    print(f"输入点数: {len(positions)}")

    spline = PositionSpline(positions, mse_tolerance=1e-8)
    spline.fit()

    print(f"总弧长: {spline.length:.4f} mm")
    print(f"进给校正多项式段数: {len(spline.feed_corrections)}")

    # 验证插值精度
    errors = []
    for i, u in enumerate(spline.u_bar):
        interp_pos = spline.spline(u)
        error = np.linalg.norm(interp_pos - positions[i])
        errors.append(error)

    print(f"插值误差: max={max(errors):.2e}, mean={np.mean(errors):.2e}")

    # 测试弧长参数化
    l_samples = np.linspace(0, spline.length, 50)
    pos_samples = spline.evaluate_batch(l_samples)
    print(f"弧长采样点数: {len(pos_samples)}")
