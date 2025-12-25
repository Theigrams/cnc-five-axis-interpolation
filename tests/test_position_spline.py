"""
position_spline 模块单元测试
"""

import numpy as np
import pytest

from cnc_five_axis_interpolation.core.position_spline import PositionSpline
from cnc_five_axis_interpolation.core.bspline import (
    centripetal_parameterization,
    compute_knot_vector,
    fit_quintic_bspline,
)


class TestCentripetalParameterization:
    """向心参数化测试"""

    def test_boundary_values(self):
        """测试边界值为 0 和 1"""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        u_bar = centripetal_parameterization(points)
        assert u_bar[0] == 0.0
        assert u_bar[-1] == 1.0

    def test_monotonically_increasing(self):
        """测试参数值单调递增"""
        points = np.random.rand(10, 3) * 10
        u_bar = centripetal_parameterization(points)
        assert np.all(np.diff(u_bar) >= 0)

    def test_coincident_points(self):
        """测试重合点使用均匀参数化"""
        points = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        u_bar = centripetal_parameterization(points)
        np.testing.assert_allclose(u_bar, [0, 0.5, 1])


class TestKnotVector:
    """节点向量测试"""

    def test_knot_vector_length(self):
        """测试节点向量长度"""
        u_bar = np.linspace(0, 1, 10)
        degree = 5
        knots = compute_knot_vector(u_bar, degree)
        # 长度应为 N + degree + 1 = 10 + 5 + 1 = 16
        assert len(knots) == len(u_bar) + degree + 1

    def test_boundary_knots(self):
        """测试边界节点重复"""
        u_bar = np.linspace(0, 1, 10)
        degree = 5
        knots = compute_knot_vector(u_bar, degree)
        # 前 degree+1 个应为 0
        np.testing.assert_allclose(knots[: degree + 1], 0)
        # 后 degree+1 个应为 1
        np.testing.assert_allclose(knots[-(degree + 1) :], 1)


class TestQuinticBSpline:
    """五次B样条拟合测试"""

    def test_interpolation_accuracy(self):
        """测试样条穿过所有数据点"""
        points = np.array(
            [
                [0, 0, 0],
                [1, 2, 0],
                [3, 3, 1],
                [5, 2, 2],
                [6, 0, 1],
                [7, -1, 0],
            ],
            dtype=float,
        )
        spline, knots, u_bar = fit_quintic_bspline(points)

        for i, u in enumerate(u_bar):
            interp_point = spline(u)
            np.testing.assert_allclose(interp_point, points[i], atol=1e-10)

    def test_spline_degree(self):
        """测试样条阶数为5"""
        points = np.random.rand(10, 3)
        spline, _, _ = fit_quintic_bspline(points)
        assert spline.k == 5


class TestPositionSpline:
    """位置样条类测试"""

    @pytest.fixture
    def simple_path(self):
        """简单测试路径"""
        t = np.linspace(0, 2 * np.pi, 15)
        points = np.column_stack([np.cos(t), np.sin(t), t / (2 * np.pi)])
        return points

    def test_fit_and_length(self, simple_path):
        """测试拟合和弧长计算"""
        spline = PositionSpline(simple_path)
        spline.fit()

        assert spline.length > 0
        assert spline.spline is not None
        assert len(spline.feed_corrections) > 0

    def test_evaluate_at_boundaries(self, simple_path):
        """测试边界点评估"""
        spline = PositionSpline(simple_path)
        spline.fit()

        # l=0 应接近第一个点
        pos_start = spline.evaluate(0)
        np.testing.assert_allclose(pos_start, simple_path[0], atol=1e-6)

        # l=length 应接近最后一个点
        pos_end = spline.evaluate(spline.length)
        np.testing.assert_allclose(pos_end, simple_path[-1], atol=1e-6)

    def test_interpolation_passes_through_points(self, simple_path):
        """测试样条穿过所有数据点"""
        spline = PositionSpline(simple_path)
        spline.fit()

        for i, u in enumerate(spline.u_bar):
            interp_point = spline.spline(u)
            np.testing.assert_allclose(interp_point, simple_path[i], atol=1e-10)

    def test_batch_evaluate(self, simple_path):
        """测试批量评估"""
        spline = PositionSpline(simple_path)
        spline.fit()

        l_values = np.linspace(0, spline.length, 20)
        positions = spline.evaluate_batch(l_values)

        assert positions.shape == (20, 3)


class TestEdgeCases:
    """边界情况测试"""

    def test_minimum_points(self):
        """测试最少点数（6点，五次B样条最小要求）"""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [2, 1, 0],
            [3, 1, 1], [4, 0, 1], [5, 0, 0]
        ], dtype=float)
        spline = PositionSpline(points)
        spline.fit()
        assert spline.length > 0
        assert spline.spline is not None

    def test_straight_line(self):
        """测试直线路径"""
        points = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        spline = PositionSpline(points)
        spline.fit()
        # 直线弧长应接近端点距离
        expected_length = 9.0
        np.testing.assert_allclose(spline.length, expected_length, rtol=1e-3)

    def test_arc_length_clipping(self):
        """测试弧长裁剪（超出范围）"""
        points = np.array([
            [0, 0, 0], [1, 1, 0], [2, 0, 0],
            [3, 1, 0], [4, 0, 0], [5, 1, 0]
        ], dtype=float)
        spline = PositionSpline(points)
        spline.fit()

        # 负弧长应返回起点
        pos_neg = spline.evaluate(-10)
        np.testing.assert_allclose(pos_neg, points[0], atol=1e-6)

        # 超长弧长应返回终点
        pos_over = spline.evaluate(spline.length + 100)
        np.testing.assert_allclose(pos_over, points[-1], atol=1e-6)

    def test_batch_with_mixed_lengths(self):
        """测试混合弧长批量评估"""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [2, 1, 0],
            [3, 1, 1], [4, 0, 1], [5, 0, 0]
        ], dtype=float)
        spline = PositionSpline(points)
        spline.fit()

        # 包含边界和超出范围的值
        l_values = np.array([-1, 0, spline.length / 2, spline.length, spline.length + 1])
        positions = spline.evaluate_batch(l_values)

        assert positions.shape == (5, 3)
        np.testing.assert_allclose(positions[0], positions[1], atol=1e-6)  # -1 和 0 相同
        np.testing.assert_allclose(positions[3], positions[4], atol=1e-6)  # length 和 length+1 相同


class TestWithRealData:
    """使用真实数据集测试"""

    def test_ijms2021_data(self):
        """测试 IJMS2021 数据"""
        from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

        positions, _, _ = ijms2021_fan_shaped_path()
        spline = PositionSpline(positions, mse_tolerance=1e-6)
        spline.fit()

        # 验证弧长合理
        assert spline.length > 0

        # 验证插值精度
        for i, u in enumerate(spline.u_bar):
            interp_point = spline.spline(u)
            error = np.linalg.norm(interp_point - positions[i])
            assert error < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
