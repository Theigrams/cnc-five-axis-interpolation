"""
algorithm - 五轴平滑轨迹生成主算法

基于论文: Yuen, Zhang, Altintas (2013)
"Smooth trajectory generation for five-axis machine tools"

该模块实现 FiveAxisPath 类，整合位置样条和姿态样条生成 C³ 连续的五轴轨迹。
"""

import numpy as np

from .core.kinematics import inverse_kinematics_ac, batch_inverse_kinematics_ac
from .core.orientation_spline import OrientationSpline
from .core.position_spline import PositionSpline


class FiveAxisPath:
    """
    五轴平滑轨迹生成器。

    通过解耦位置和姿态的插补，实现 C³ 连续的五轴轨迹生成。
    基于论文 Yuen et al. (2013) 的方法。

    Attributes:
        positions: (N, 3) 刀尖位置点
        orientations: (N, 3) 刀轴姿态向量
        position_spline: 位置样条对象
        orientation_spline: 姿态样条对象
        length: 轨迹总弧长
    """

    def __init__(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        mse_tolerance: float = 1e-6,
        L_ac_z: float = 70.0,
        L_Tya_z: float = 150.0,
    ):
        """
        初始化五轴轨迹生成器。

        Args:
            positions: (N, 3) 刀尖位置数据
            orientations: (N, 3) 刀轴姿态数据（单位向量）
            mse_tolerance: 进给校正多项式的MSE容差
            L_ac_z: A轴到C轴的Z方向偏移 (mm)
            L_Tya_z: 主轴到A轴的Z方向偏移 (mm)
        """
        self.positions = np.asarray(positions)
        self.orientations = np.asarray(orientations)
        self.mse_tolerance = mse_tolerance
        self.L_ac_z = L_ac_z
        self.L_Tya_z = L_Tya_z
        self.N = len(positions)

        # 确保姿态归一化
        norms = np.linalg.norm(self.orientations, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.orientations = self.orientations / norms

        # 样条对象
        self.position_spline: PositionSpline | None = None
        self.orientation_spline: OrientationSpline | None = None
        self.length: float = 0.0

        # 数据点对应的弧长
        self._arc_lengths_at_points: np.ndarray | None = None

    def fit(self):
        """
        拟合位置和姿态样条。返回 self 以支持链式调用。

        执行:
        1. 拟合五次 B 样条位置曲线
        2. 计算弧长参数化
        3. 拟合姿态样条和 Bézier 重参数化
        """
        # Step 1: 拟合位置样条
        self.position_spline = PositionSpline(self.positions, self.mse_tolerance)
        self.position_spline.fit()
        self.length = self.position_spline.length

        # Step 2: 计算数据点对应的弧长
        self._compute_arc_lengths_at_points()

        # Step 3: 拟合姿态样条
        self.orientation_spline = OrientationSpline(
            self.orientations, self._arc_lengths_at_points
        )
        self.orientation_spline.fit()

        return self

    def _compute_arc_lengths_at_points(self):
        """计算每个数据点对应的弧长。"""
        # 使用位置样条的参数值计算弧长
        arc_lengths = []
        for u in self.position_spline.u_bar:
            # 找到对应的弧长
            # 通过数值搜索找到 l 使得 get_u_from_length(l) ≈ u
            l = self._find_arc_length_for_u(u)
            arc_lengths.append(l)
        self._arc_lengths_at_points = np.array(arc_lengths)

    def _find_arc_length_for_u(self, target_u: float) -> float:
        """二分搜索找到对应参数 u 的弧长 l。"""
        if target_u <= 0:
            return 0.0
        if target_u >= 1:
            return self.length

        # 二分搜索
        l_low, l_high = 0.0, self.length
        for _ in range(50):  # 最多50次迭代
            l_mid = (l_low + l_high) / 2
            u_mid = self.position_spline.get_u_from_length(l_mid)
            if u_mid < target_u:
                l_low = l_mid
            else:
                l_high = l_mid
            if abs(u_mid - target_u) < 1e-12:
                break

        return (l_low + l_high) / 2

    def evaluate(self, l: float) -> tuple[np.ndarray, np.ndarray]:
        """
        在弧长 l 处评估位置和姿态。

        Args:
            l: 弧长参数

        Returns:
            position: (3,) 刀尖位置
            orientation: (3,) 刀轴姿态
        """
        position = self.position_spline.evaluate(l)
        orientation = self.orientation_spline.evaluate(l)
        return position, orientation

    def evaluate_machine_coords(self, l: float) -> tuple[np.ndarray, float, float]:
        """
        在弧长 l 处评估机床坐标。

        应用 A-C 配置逆运动学变换 (Eq.42)。

        Args:
            l: 弧长参数

        Returns:
            XYZ: (3,) 机床线性轴坐标
            A: A轴角度 (rad)
            C: C轴角度 (rad)
        """
        position, orientation = self.evaluate(l)
        return inverse_kinematics_ac(position, orientation, self.L_ac_z, self.L_Tya_z)

    def evaluate_batch(self, l_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        批量在弧长处评估位置和姿态。

        Args:
            l_values: (M,) 弧长数组

        Returns:
            positions: (M, 3) 位置数组
            orientations: (M, 3) 姿态数组
        """
        positions = self.position_spline.evaluate_batch(l_values)
        orientations = self.orientation_spline.evaluate_batch(l_values)
        return positions, orientations

    def evaluate_machine_coords_batch(
        self, l_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量在弧长处评估机床坐标。

        Args:
            l_values: (M,) 弧长数组

        Returns:
            XYZ: (M, 3) 机床线性轴坐标
            A: (M,) A轴角度 (rad)
            C: (M,) C轴角度 (rad)
        """
        positions, orientations = self.evaluate_batch(l_values)
        return batch_inverse_kinematics_ac(
            positions, orientations, self.L_ac_z, self.L_Tya_z
        )

    def sample_uniform(self, num_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        沿弧长均匀采样。

        Args:
            num_points: 采样点数

        Returns:
            l_values: (M,) 弧长值
            positions: (M, 3) 位置
            orientations: (M, 3) 姿态
        """
        l_values = np.linspace(0, self.length, num_points)
        positions, orientations = self.evaluate_batch(l_values)
        return l_values, positions, orientations

    def __repr__(self) -> str:
        status = "fitted" if self.position_spline is not None else "not fitted"
        return f"FiveAxisPath(N={self.N}, length={self.length:.2f}mm, {status})"


if __name__ == "__main__":
    from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

    positions, orientations, constraints = ijms2021_fan_shaped_path()

    print("=== 五轴轨迹生成测试 ===")
    print(f"输入点数: {len(positions)}")

    path = FiveAxisPath(positions, orientations, mse_tolerance=1e-6)
    path.fit()

    print(f"轨迹总弧长: {path.length:.4f} mm")
    print(f"位置样条段数: {len(path.position_spline.feed_corrections)}")

    # 测试评估
    l_test = path.length / 2
    pos, ori = path.evaluate(l_test)
    print(f"\n在 l={l_test:.2f}mm 处:")
    print(f"  位置: {pos}")
    print(f"  姿态: {ori}")

    # 测试机床坐标
    XYZ, A, C = path.evaluate_machine_coords(l_test)
    print(f"  机床坐标: X={XYZ[0]:.2f}, Y={XYZ[1]:.2f}, Z={XYZ[2]:.2f}")
    print(f"  旋转轴: A={np.degrees(A):.2f}°, C={np.degrees(C):.2f}°")

    # 均匀采样测试
    l_samples, pos_samples, ori_samples = path.sample_uniform(100)
    print(f"\n均匀采样: {len(l_samples)} 点")
    print(f"姿态归一化: {np.allclose(np.linalg.norm(ori_samples, axis=1), 1.0)}")
