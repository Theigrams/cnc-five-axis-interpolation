"""
orientation_spline 模块单元测试
"""

import numpy as np
import pytest

from cnc_five_axis_interpolation.core.orientation_spline import (
    OrientationSpline,
    angular_parameterization,
    evaluate_orientation_from_spherical,
    fit_orientation_bspline,
    BezierReparameterization,
)


class TestAngularParameterization:
    """角度参数化测试"""

    def test_boundary_values(self):
        """测试边界值为 0 和 1"""
        orientations = np.array([
            [0, 0, 1],
            [0.1, 0, 0.995],
            [0.2, 0, 0.98],
            [0.3, 0, 0.954],
        ], dtype=float)
        # 归一化
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        w_bar = angular_parameterization(orientations)
        assert w_bar[0] == 0.0
        assert w_bar[-1] == 1.0

    def test_monotonically_increasing(self):
        """测试参数值单调递增"""
        orientations = np.array([
            [0, 0, 1],
            [0.1, 0.1, 0.99],
            [0.2, 0.2, 0.96],
            [0.3, 0.3, 0.91],
            [0.4, 0.4, 0.82],
        ], dtype=float)
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        w_bar = angular_parameterization(orientations)
        assert np.all(np.diff(w_bar) >= 0)

    def test_constant_orientations(self):
        """测试相同姿态使用均匀参数化"""
        orientations = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
        w_bar = angular_parameterization(orientations)
        np.testing.assert_allclose(w_bar, [0, 0.5, 1])


class TestOrientationBSpline:
    """姿态B样条拟合测试"""

    def test_interpolation_accuracy(self):
        """测试样条穿过所有数据点"""
        # 创建平滑变化的姿态
        t = np.linspace(0, np.pi / 4, 8)
        orientations = np.column_stack([
            np.sin(t) * np.cos(t),
            np.sin(t) * np.sin(t),
            np.cos(t)
        ])

        spline, knots, w_bar = fit_orientation_bspline(orientations)

        for i, w in enumerate(w_bar):
            interp_ori = evaluate_orientation_from_spherical(spline, w)
            interp_ori = interp_ori / np.linalg.norm(interp_ori)
            np.testing.assert_allclose(interp_ori, orientations[i], atol=1e-10)


class TestBezierReparameterization:
    """Bézier重参数化测试"""

    def test_boundary_values(self):
        """测试边界点正确"""
        l_values = np.array([0, 10, 25, 50])
        w_values = np.array([0, 0.2, 0.5, 1.0])

        reparam = BezierReparameterization(l_values, w_values)

        # 端点应匹配
        assert np.isclose(reparam(0), 0, atol=1e-6)
        assert np.isclose(reparam(50), 1.0, atol=1e-6)

    def test_monotonicity(self):
        """测试单调性"""
        l_values = np.array([0, 10, 25, 50])
        w_values = np.array([0, 0.2, 0.5, 1.0])

        reparam = BezierReparameterization(l_values, w_values)

        l_samples = np.linspace(0, 50, 100)
        w_samples = reparam.evaluate_batch(l_samples)

        # 应该单调递增
        assert np.all(np.diff(w_samples) >= -1e-10)


class TestOrientationSpline:
    """姿态样条类测试"""

    @pytest.fixture
    def simple_orientations(self):
        """简单测试姿态"""
        t = np.linspace(0, np.pi / 3, 10)
        orientations = np.column_stack([
            np.sin(t),
            np.zeros_like(t),
            np.cos(t)
        ])
        arc_lengths = np.linspace(0, 100, 10)
        return orientations, arc_lengths

    def test_fit(self, simple_orientations):
        """测试拟合"""
        orientations, arc_lengths = simple_orientations

        spline = OrientationSpline(orientations, arc_lengths)
        spline.fit()

        assert spline.spline is not None
        assert spline.reparameterization is not None

    def test_evaluate_at_boundaries(self, simple_orientations):
        """测试边界点评估"""
        orientations, arc_lengths = simple_orientations

        spline = OrientationSpline(orientations, arc_lengths)
        spline.fit()

        # l=0 应接近第一个姿态
        ori_start = spline.evaluate(0)
        np.testing.assert_allclose(ori_start, orientations[0], atol=1e-3)

        # l=max 应接近最后一个姿态
        ori_end = spline.evaluate(arc_lengths[-1])
        np.testing.assert_allclose(ori_end, orientations[-1], atol=1e-3)

    def test_orientations_normalized(self, simple_orientations):
        """测试输出姿态归一化"""
        orientations, arc_lengths = simple_orientations

        spline = OrientationSpline(orientations, arc_lengths)
        spline.fit()

        l_samples = np.linspace(0, arc_lengths[-1], 50)
        ori_samples = spline.evaluate_batch(l_samples)

        norms = np.linalg.norm(ori_samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestWithRealData:
    """使用真实数据集测试"""

    def test_ijms2021_data(self):
        """测试 IJMS2021 数据"""
        from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

        positions, orientations, _ = ijms2021_fan_shaped_path()

        # 计算弧长
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        arc_lengths = np.concatenate([[0], np.cumsum(dists)])

        spline = OrientationSpline(orientations, arc_lengths)
        spline.fit()

        # 验证插值精度
        for i, w in enumerate(spline.w_bar):
            interp_ori = evaluate_orientation_from_spherical(spline.spline, w)
            interp_ori = interp_ori / np.linalg.norm(interp_ori)
            error = np.linalg.norm(interp_ori - orientations[i])
            assert error < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
