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


def centripetal_parameterization(points: np.ndarray) -> np.ndarray:
    """
    向心参数化方法 (Eq.4)。

    使用相邻点距离的平方根来分配参数值，相比弦长参数化能产生更好的拟合效果。

    Args:
        points: (N, 3) 离散点坐标

    Returns:
        u_bar: (N,) 参数值数组, u_bar[0]=0, u_bar[-1]=1
    """
    N = len(points)
    u_bar = np.zeros(N)

    # 计算相邻点距离的平方根之和
    sqrt_dists = np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1))
    d = np.sum(sqrt_dists)

    if d < 1e-12:
        # 所有点重合，使用均匀参数化
        return np.linspace(0, 1, N)

    # 累积参数化
    u_bar[0] = 0.0
    for k in range(1, N):
        u_bar[k] = u_bar[k - 1] + sqrt_dists[k - 1] / d
    u_bar[-1] = 1.0  # 确保端点精确

    return u_bar


def compute_knot_vector(u_bar: np.ndarray, degree: int) -> np.ndarray:
    """
    计算B样条节点向量 (Eq.5)。

    使用均值法从参数值计算内部节点。

    Args:
        u_bar: (N,) 参数值数组
        degree: 样条阶数 (n=5 for quintic)

    Returns:
        U: (N + degree + 1,) 节点向量
    """
    N = len(u_bar) - 1  # 点数为 N+1
    n = degree
    num_knots = N + n + 2  # 总节点数 = N + n + 1 + 1

    U = np.zeros(num_knots)

    # 端点节点重复 n+1 次
    U[: n + 1] = 0.0
    U[-(n + 1) :] = 1.0

    # 内部节点使用均值法 (Eq.5)
    for j in range(1, N - n + 1):
        U[j + n] = np.mean(u_bar[j : j + n])

    return U


def bspline_basis_matrix(u_bar: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    计算B样条基函数矩阵 (Eq.6)。

    构造 (N+1) x (N+1) 的基函数矩阵 Φ，用于求解控制点。

    Args:
        u_bar: (N+1,) 参数值
        knots: 节点向量
        degree: 样条阶数

    Returns:
        Phi: (N+1, N+1) 基函数矩阵
    """
    N = len(u_bar)
    Phi = np.zeros((N, N))

    for row, u in enumerate(u_bar):
        # 使用 scipy 的 B 样条基函数计算
        for col in range(N):
            # 构造单位控制点向量
            c = np.zeros(N)
            c[col] = 1.0
            basis = BSpline(knots, c, degree)
            Phi[row, col] = basis(u)

    return Phi


def fit_quintic_bspline(points: np.ndarray) -> tuple[BSpline, np.ndarray, np.ndarray]:
    """
    拟合五次B样条曲线 (Eq.1-7)。

    Args:
        points: (N, 3) 离散刀尖位置

    Returns:
        spline: scipy BSpline 对象
        knots: 节点向量
        u_bar: 参数值
    """
    degree = 5

    # Step 1: 向心参数化 (Eq.4)
    u_bar = centripetal_parameterization(points)

    # Step 2: 计算节点向量 (Eq.5)
    knots = compute_knot_vector(u_bar, degree)

    # Step 3: 构造基函数矩阵 (Eq.6)
    Phi = bspline_basis_matrix(u_bar, knots, degree)

    # Step 4: 求解控制点 (Eq.7)
    # Γ = Φ^(-1) Ψ
    control_points = np.linalg.solve(Phi, points)

    # 创建 BSpline 对象
    spline = BSpline(knots, control_points, degree)

    return spline, knots, u_bar


class FeedCorrectionPolynomial:
    """
    9阶进给校正多项式 (Eq.11-22)。

    将弧长 l 映射到样条参数 u，同时满足 C³ 边界条件。
    """

    def __init__(self, coeffs: np.ndarray, total_length: float):
        """
        Args:
            coeffs: (10,) 归一化多项式系数 [a_9, a_8, ..., a_0]
            total_length: 总弧长 S
        """
        self.coeffs = coeffs
        self.S = total_length

    def __call__(self, l: float | np.ndarray) -> float | np.ndarray:
        """计算 u(l)"""
        sigma = np.asarray(l) / self.S
        # 使用 Horner 法计算多项式
        u = np.zeros_like(sigma)
        for a in self.coeffs:
            u = u * sigma + a
        return u

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


def compute_boundary_derivatives(spline: BSpline, S: float) -> tuple[np.ndarray, np.ndarray]:
    """
    计算边界处的导数值 (Eq.16)。

    Args:
        spline: 位置 B 样条
        S: 总弧长

    Returns:
        derivs_0: [u_l(0), u_ll(0), u_lll(0)] * S^{1,2,3}
        derivs_1: [u_l(1), u_ll(1), u_lll(1)] * S^{1,2,3}
    """

    def f(u):
        """弧长导数 ||P'(u)||"""
        deriv = spline.derivative()(u)
        return np.linalg.norm(deriv)

    def f_prime(u, h=1e-7):
        """f'(u) 数值微分"""
        return (f(u + h) - f(u - h)) / (2 * h)

    def f_double_prime(u, h=1e-5):
        """f''(u) 数值微分"""
        return (f(u + h) - 2 * f(u) + f(u - h)) / (h**2)

    # 边界处的值
    derivs_0 = np.zeros(3)
    derivs_1 = np.zeros(3)

    for i, u in enumerate([0.0, 1.0]):
        # 避免精确边界点的数值问题
        u_eval = max(1e-8, min(u, 1 - 1e-8))

        f_val = f(u_eval)
        fp_val = f_prime(u_eval)
        fpp_val = f_double_prime(u_eval)

        # u_l = 1/f (Eq.16)
        u_l = 1.0 / f_val

        # u_ll = -f'/f³ (Eq.16)
        u_ll = -fp_val / (f_val**3)

        # u_lll = (3f'² - f''f) / f⁵ (Eq.16)
        u_lll = (3 * fp_val**2 - fpp_val * f_val) / (f_val**5)

        if i == 0:
            derivs_0 = np.array([u_l * S, u_ll * S**2, u_lll * S**3])
        else:
            derivs_1 = np.array([u_l * S, u_ll * S**2, u_lll * S**3])

    return derivs_0, derivs_1


def fit_feed_correction_polynomial(
    arc_lengths: np.ndarray,
    u_values: np.ndarray,
    spline: BSpline,
) -> FeedCorrectionPolynomial:
    """
    拟合9阶进给校正多项式 (Eq.11-22)。

    使用约束最小二乘法，满足 C³ 边界条件。

    Args:
        arc_lengths: (M,) 累积弧长 [0, l_1, ..., l_M=S]
        u_values: (M,) 对应的参数值 [0, u_1, ..., u_M=1]
        spline: 位置 B 样条

    Returns:
        FeedCorrectionPolynomial 对象
    """
    S = arc_lengths[-1]
    sigma = arc_lengths / S  # 归一化弧长

    # 构造 Vandermonde 矩阵 Φ (Eq.13)
    M = len(sigma)
    powers = np.arange(9, -1, -1)
    Phi = sigma[:, np.newaxis] ** powers  # (M, 10)

    # 计算边界导数 (Eq.16)
    derivs_0, derivs_1 = compute_boundary_derivatives(spline, S)

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

    # 右侧向量 η
    eta = np.array(
        [
            0,  # u(0) = 0
            derivs_0[0],  # u'(0)*S
            derivs_0[1],  # u''(0)*S²
            derivs_0[2],  # u'''(0)*S³
            1,  # u(1) = 1
            derivs_1[0],  # u'(1)*S
            derivs_1[1],  # u''(1)*S²
            derivs_1[2],  # u'''(1)*S³
        ]
    )

    # 构造 KKT 系统 (Eq.22)
    # [Φ'Φ  Ω']   [α]   [Φ'u*]
    # [Ω    0 ] * [Λ] = [η   ]
    PhiTPhi = Phi.T @ Phi
    PhiTu = Phi.T @ u_values

    n_constraints = 8
    n_coeffs = 10

    KKT = np.zeros((n_coeffs + n_constraints, n_coeffs + n_constraints))
    KKT[:n_coeffs, :n_coeffs] = PhiTPhi
    KKT[:n_coeffs, n_coeffs:] = Omega.T
    KKT[n_coeffs:, :n_coeffs] = Omega

    rhs = np.zeros(n_coeffs + n_constraints)
    rhs[:n_coeffs] = PhiTu
    rhs[n_coeffs:] = eta

    # 求解
    try:
        solution = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        solution = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    coeffs = solution[:n_coeffs]

    return FeedCorrectionPolynomial(coeffs, S)


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
        """拟合五次B样条和进给校正多项式。"""
        # Step 1: 拟合五次 B 样条
        self.spline, self.knots, self.u_bar = fit_quintic_bspline(self.points)

        # Step 2: 计算弧长表
        self._compute_arc_length_table()

        # Step 3: 自适应拟合进给校正多项式
        self._fit_feed_correction_adaptive()

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

        for start, end, length in intervals:
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

    def get_u_from_length(self, l: float) -> float:
        """根据弧长获取样条参数 u。"""
        l = np.clip(l, 0, self.length)

        for l_start, l_end, poly in self.feed_corrections:
            if l_start <= l <= l_end:
                return poly(l - l_start)

        # 如果没找到，使用最后一个
        _, _, poly = self.feed_corrections[-1]
        return poly(l - self.feed_corrections[-1][0])

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
        批量在弧长处评估位置。

        Args:
            l_values: (M,) 弧长数组

        Returns:
            (M, 3) 位置数组
        """
        return np.array([self.evaluate(l) for l in l_values])


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
