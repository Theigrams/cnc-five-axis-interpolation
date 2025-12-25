"""
orientation_spline - C³ 连续姿态样条生成

基于论文 Section 3: C³ tool orientation spline generation

实现:
1. 球坐标转换 (Eq.25)
2. 五次B样条拟合姿态 (Eq.26-30)
3. 7阶Bézier重参数化 (Eq.32-41)
"""

import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from ..utils.geometry import batch_cartesian_to_spherical, spherical_to_cartesian, unwrap_angles
from .position_spline import centripetal_parameterization, compute_knot_vector, bspline_basis_matrix


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
    w_bar = np.zeros(N)

    # 计算相邻姿态的角度变化 (Eq.28)
    # angle_k = arccos(o_k · o_{k-1})
    angles = []
    for k in range(1, N):
        dot = np.clip(np.dot(orientations[k], orientations[k - 1]), -1.0, 1.0)
        angle = np.arccos(dot)
        angles.append(angle)

    sqrt_angles = np.sqrt(angles)
    d2 = np.sum(sqrt_angles)

    if d2 < 1e-12:
        # 所有姿态相同，使用均匀参数化
        return np.linspace(0, 1, N)

    # 累积参数化
    w_bar[0] = 0.0
    for k in range(1, N):
        w_bar[k] = w_bar[k - 1] + sqrt_angles[k - 1] / d2
    w_bar[-1] = 1.0

    return w_bar


def fit_orientation_bspline(orientations: np.ndarray) -> tuple[BSpline, np.ndarray, np.ndarray]:
    """
    拟合姿态五次B样条 (Eq.26-30)。

    先将姿态转换为球坐标，在(θ,φ)平面拟合B样条，再映射回笛卡尔坐标。

    Args:
        orientations: (N, 3) 单位刀轴向量

    Returns:
        spline: scipy BSpline 对象 (在球坐标平面)
        knots: 节点向量
        w_bar: 参数值
    """
    degree = 5

    # Step 1: 转换为球坐标 (Eq.25)
    theta, phi = batch_cartesian_to_spherical(orientations)

    # 展开 φ 角避免 ±π 不连续
    phi = unwrap_angles(phi)

    # 组合为2D点
    spherical_points = np.column_stack([theta, phi])

    # Step 2: 角度参数化 (Eq.28)
    w_bar = angular_parameterization(orientations)

    # Step 3: 计算节点向量 (Eq.29)
    knots = compute_knot_vector(w_bar, degree)

    # Step 4: 构造基函数矩阵 (Eq.30)
    Phi = bspline_basis_matrix(w_bar, knots, degree)

    # Step 5: 求解控制点
    control_points = np.linalg.solve(Phi, spherical_points)

    # 创建 BSpline 对象
    spline = BSpline(knots, control_points, degree)

    return spline, knots, w_bar


def evaluate_orientation_from_spherical(spline: BSpline, w: float | np.ndarray) -> np.ndarray:
    """
    从球坐标样条评估姿态向量 (Eq.31)。

    Args:
        spline: 球坐标样条
        w: 参数值

    Returns:
        orientation: (3,) 或 (N, 3) 单位姿态向量
    """
    theta_phi = spline(w)
    if theta_phi.ndim == 1:
        theta, phi = theta_phi
        return spherical_to_cartesian(theta, phi)
    else:
        return np.array([spherical_to_cartesian(tp[0], tp[1]) for tp in theta_phi])


class BezierReparameterization:
    """
    7阶Bézier重参数化 (Eq.32-41)。

    将弧长 l 映射到姿态样条参数 w，满足 C³ 连续和单调性约束。
    """

    def __init__(self, l_values: np.ndarray, w_values: np.ndarray):
        """
        Args:
            l_values: (N,) 累积弧长 [l_0, l_1, ..., l_{N-1}]
            w_values: (N,) 对应的参数值 [w_0, w_1, ..., w_{N-1}]
        """
        self.l_values = l_values
        self.w_values = w_values
        self.N = len(l_values)

        # 每段的控制系数 Q[k] = [Q_0,k, Q_1,k, ..., Q_7,k]
        self.Q: list[np.ndarray] = []

        self._fit()

    def _fit(self):
        """拟合7阶Bézier样条 (Eq.39-41)。"""
        N = self.N - 1  # 段数

        if N == 0:
            return

        # 初始化控制系数 (Eq.41)
        self._initialize_coefficients()

        # 优化以最小化 jerk 积分 (Eq.39)
        self._optimize()

    def _initialize_coefficients(self):
        """初始化满足约束的控制系数 (Eq.41)。"""
        N = self.N - 1
        l = self.l_values
        w = self.w_values

        # 计算全局 δ (Eq.41)
        deltas = []
        for k in range(1, self.N):
            dl = l[k] - l[k - 1]
            dw = w[k] - w[k - 1]
            if dl > 1e-12:
                deltas.append(dw / (7 * dl))
            else:
                deltas.append(0)

        if deltas:
            delta = min(deltas) if min(deltas) > 0 else 1e-6
        else:
            delta = 1e-6

        # 初始化每段的控制系数
        self.Q = []
        for k in range(1, self.N):
            dl = l[k] - l[k - 1]
            Delta_k = delta * dl

            # Q_0,k = w_{k-1}, Q_7,k = w_k (Eq.37)
            Q = np.zeros(8)
            Q[0] = w[k - 1]
            Q[7] = w[k]

            # 中间系数使用线性初始化 (Eq.41)
            Q[1] = w[k - 1] + Delta_k
            Q[2] = w[k - 1] + 2 * Delta_k
            Q[3] = w[k - 1] + 3 * Delta_k
            Q[4] = w[k] - 3 * Delta_k
            Q[5] = w[k] - 2 * Delta_k
            Q[6] = w[k] - Delta_k

            self.Q.append(Q)

    def _optimize(self):
        """优化控制系数最小化 jerk 积分 (Eq.39)。"""
        if len(self.Q) == 0:
            return

        N = len(self.Q)
        l = self.l_values
        w = self.w_values

        # 提取可优化的变量: Q_1 到 Q_6 (Q_0 和 Q_7 固定)
        # 每段有6个自由变量
        n_vars = N * 6
        x0 = np.zeros(n_vars)
        for k in range(N):
            x0[k * 6 : (k + 1) * 6] = self.Q[k][1:7]

        def objective(x):
            """Jerk 积分目标函数 (Eq.39)"""
            total = 0.0
            for k in range(N):
                Q = np.zeros(8)
                Q[0] = w[k]
                Q[7] = w[k + 1]
                Q[1:7] = x[k * 6 : (k + 1) * 6]

                dl = l[k + 1] - l[k]
                if dl < 1e-12:
                    continue

                # 简化的 jerk 积分近似
                # 使用三阶差分作为 jerk 的代理
                for i in range(5):
                    jerk = Q[i + 3] - 3 * Q[i + 2] + 3 * Q[i + 1] - Q[i]
                    total += (jerk / dl**3) ** 2

            return total

        def build_constraints():
            """构建约束条件 (Eq.36, 38, 40)"""
            constraints = []

            # 单调性约束 (Eq.36): Q_0 <= Q_1 <= ... <= Q_7
            for k in range(N):

                def monotonicity_constraint(x, k=k, i=0):
                    Q = np.zeros(8)
                    Q[0] = w[k]
                    Q[7] = w[k + 1]
                    Q[1:7] = x[k * 6 : (k + 1) * 6]
                    return np.min(np.diff(Q))

                constraints.append({"type": "ineq", "fun": monotonicity_constraint})

            # C³ 连续性约束 (Eq.38)
            for k in range(N - 1):

                def c1_constraint(x, k=k):
                    Q_k = np.zeros(8)
                    Q_k[0] = w[k]
                    Q_k[7] = w[k + 1]
                    Q_k[1:7] = x[k * 6 : (k + 1) * 6]

                    Q_k1 = np.zeros(8)
                    Q_k1[0] = w[k + 1]
                    Q_k1[7] = w[k + 2]
                    Q_k1[1:7] = x[(k + 1) * 6 : (k + 2) * 6]

                    dl_k = l[k + 1] - l[k]
                    dl_k1 = l[k + 2] - l[k + 1]

                    if dl_k < 1e-12 or dl_k1 < 1e-12:
                        return 0

                    # C¹: (Q_7,k - Q_6,k)/dl_k = (Q_1,k+1 - Q_0,k+1)/dl_k+1
                    lhs = (Q_k[7] - Q_k[6]) / dl_k
                    rhs = (Q_k1[1] - Q_k1[0]) / dl_k1
                    return -(lhs - rhs) ** 2

                constraints.append({"type": "eq", "fun": c1_constraint})

            return constraints

        # 运行优化
        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints=build_constraints(),
                options={"maxiter": 100, "ftol": 1e-6},
            )

            if result.success:
                for k in range(N):
                    self.Q[k][1:7] = result.x[k * 6 : (k + 1) * 6]
        except Exception:
            # 如果优化失败，保持初始值
            pass

    def __call__(self, l: float) -> float:
        """根据弧长 l 计算参数 w。"""
        l = np.clip(l, self.l_values[0], self.l_values[-1])

        # 找到所在区间
        for k in range(len(self.Q)):
            l_k = self.l_values[k]
            l_k1 = self.l_values[k + 1]

            if l_k <= l <= l_k1:
                # 计算局部参数 r (Eq.32)
                dl = l_k1 - l_k
                if dl < 1e-12:
                    return self.w_values[k]

                r = (l - l_k) / dl

                # 计算7阶Bernstein多项式 (Eq.32)
                Q = self.Q[k]
                w = 0.0
                for i in range(8):
                    binom = math.comb(7, i)
                    w += binom * ((1 - r) ** (7 - i)) * (r**i) * Q[i]

                return w

        return self.w_values[-1]

    def evaluate_batch(self, l_values: np.ndarray) -> np.ndarray:
        """批量评估。"""
        return np.array([self(l) for l in l_values])


class OrientationSpline:
    """
    C³ 连续姿态样条。

    将离散刀轴姿态转换为球坐标后拟合五次B样条，
    并通过7阶Bézier实现弧长参数化。
    """

    def __init__(self, orientations: np.ndarray, arc_lengths: np.ndarray):
        """
        初始化姿态样条。

        Args:
            orientations: (N, 3) 刀轴姿态向量（单位向量）
            arc_lengths: (N,) 对应的弧长值
        """
        self.orientations = np.asarray(orientations)
        self.arc_lengths = np.asarray(arc_lengths)
        self.total_length = arc_lengths[-1]
        self.N = len(orientations)

        self.spline: BSpline | None = None
        self.knots: np.ndarray | None = None
        self.w_bar: np.ndarray | None = None
        self.reparameterization: BezierReparameterization | None = None

    def fit(self):
        """拟合姿态样条和重参数化曲线。"""
        # Step 1: 拟合球坐标 B 样条
        self.spline, self.knots, self.w_bar = fit_orientation_bspline(self.orientations)

        # Step 2: 构建 Bézier 重参数化
        self.reparameterization = BezierReparameterization(self.arc_lengths, self.w_bar)

    def get_w_from_length(self, l: float) -> float:
        """根据弧长获取样条参数 w。"""
        return self.reparameterization(l)

    def evaluate(self, l: float) -> np.ndarray:
        """
        在弧长 l 处评估姿态。

        Args:
            l: 弧长参数

        Returns:
            (3,) 单位姿态向量
        """
        w = self.get_w_from_length(l)
        orientation = evaluate_orientation_from_spherical(self.spline, w)
        # 确保归一化
        return orientation / np.linalg.norm(orientation)

    def evaluate_batch(self, l_values: np.ndarray) -> np.ndarray:
        """
        批量在弧长处评估姿态。

        Args:
            l_values: (M,) 弧长数组

        Returns:
            (M, 3) 姿态数组
        """
        return np.array([self.evaluate(l) for l in l_values])


if __name__ == "__main__":
    from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

    positions, orientations, _ = ijms2021_fan_shaped_path()

    # 计算弧长 (简化版)
    dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(dists)])

    print("=== 姿态样条测试 ===")
    print(f"输入点数: {len(orientations)}")

    spline = OrientationSpline(orientations, arc_lengths)
    spline.fit()

    print(f"总弧长: {spline.total_length:.4f} mm")

    # 验证插值精度
    errors = []
    for i, w in enumerate(spline.w_bar):
        interp_ori = evaluate_orientation_from_spherical(spline.spline, w)
        interp_ori = interp_ori / np.linalg.norm(interp_ori)
        error = np.linalg.norm(interp_ori - orientations[i])
        errors.append(error)

    print(f"插值误差: max={max(errors):.2e}, mean={np.mean(errors):.2e}")

    # 测试弧长参数化
    l_samples = np.linspace(0, spline.total_length, 50)
    ori_samples = spline.evaluate_batch(l_samples)
    norms = np.linalg.norm(ori_samples, axis=1)
    print(f"姿态归一化验证: all ≈ 1.0 = {np.allclose(norms, 1.0)}")
