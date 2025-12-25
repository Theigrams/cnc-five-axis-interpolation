"""
utils 模块单元测试
"""

import numpy as np
import pytest

from cnc_five_axis_interpolation.utils.geometry import (
    normalize,
    cartesian_to_spherical,
    spherical_to_cartesian,
    batch_cartesian_to_spherical,
    batch_spherical_to_cartesian,
    unwrap_angles,
)
from cnc_five_axis_interpolation.utils.integrals import (
    adaptive_simpson,
    arc_length_integral,
)


class TestGeometry:
    """几何工具函数测试"""

    def test_normalize_single_vector(self):
        """测试单向量归一化"""
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        assert np.isclose(np.linalg.norm(result), 1.0)
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0])

    def test_normalize_batch(self):
        """测试批量向量归一化"""
        vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        result = normalize(vectors)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0])

    def test_spherical_cartesian_roundtrip(self):
        """测试球坐标与笛卡尔坐标往返转换"""
        # 测试多个方向
        test_vectors = [
            np.array([1.0, 0.0, 0.0]),  # x轴
            np.array([0.0, 1.0, 0.0]),  # y轴
            np.array([0.0, 0.0, 1.0]),  # z轴
            normalize(np.array([1.0, 1.0, 1.0])),  # 对角线
            normalize(np.array([1.0, -1.0, 0.5])),  # 任意方向
        ]

        for v in test_vectors:
            theta, phi = cartesian_to_spherical(v)
            v_back = spherical_to_cartesian(theta, phi)
            np.testing.assert_allclose(v, v_back, atol=1e-10)

    def test_batch_spherical_conversion(self):
        """测试批量球坐标转换"""
        orientations = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        theta, phi = batch_cartesian_to_spherical(orientations)
        result = batch_spherical_to_cartesian(theta, phi)
        np.testing.assert_allclose(result, orientations, atol=1e-10)

    def test_theta_range(self):
        """测试极角范围 [0, π]"""
        # z轴正方向: θ = 0
        _, _ = cartesian_to_spherical(np.array([0, 0, 1]))
        theta_z_pos, _ = cartesian_to_spherical(np.array([0, 0, 1]))
        assert np.isclose(theta_z_pos, 0)

        # z轴负方向: θ = π
        theta_z_neg, _ = cartesian_to_spherical(np.array([0, 0, -1]))
        assert np.isclose(theta_z_neg, np.pi)

        # xy平面: θ = π/2
        theta_xy, _ = cartesian_to_spherical(np.array([1, 0, 0]))
        assert np.isclose(theta_xy, np.pi / 2)

    def test_unwrap_angles(self):
        """测试角度展开"""
        # 模拟跨越 ±π 边界的角度序列
        phi = np.array([2.9, 3.1, -3.1, -2.9])  # 跨越 π
        unwrapped = unwrap_angles(phi)

        # 展开后应该是连续的
        diffs = np.diff(unwrapped)
        assert np.all(np.abs(diffs) < np.pi)


class TestIntegrals:
    """数值积分测试"""

    def test_adaptive_simpson_exp(self):
        """测试 e^x 积分"""
        result = adaptive_simpson(np.exp, 0, 1)
        exact = np.e - 1
        assert np.isclose(result, exact, rtol=1e-6)

    def test_adaptive_simpson_polynomial(self):
        """测试多项式积分"""
        # ∫x^2 dx from 0 to 1 = 1/3

        def f(x):
            return x**2

        result = adaptive_simpson(f, 0, 1)
        assert np.isclose(result, 1 / 3, rtol=1e-6)

    def test_arc_length_circle(self):
        """测试圆弧长度"""

        def circle_derivative(t):
            # 四分之一圆: x = cos(πt/2), y = sin(πt/2)
            return np.array([-np.pi / 2 * np.sin(np.pi * t / 2), np.pi / 2 * np.cos(np.pi * t / 2)])

        arc_len = arc_length_integral(circle_derivative, 0, 1)
        exact = np.pi / 2
        assert np.isclose(arc_len, exact, rtol=1e-6)

    def test_arc_length_line(self):
        """测试直线长度"""

        def line_derivative(t):
            # 从 (0,0,0) 到 (1,1,1) 的直线
            return np.array([1.0, 1.0, 1.0])

        arc_len = arc_length_integral(line_derivative, 0, 1)
        exact = np.sqrt(3)
        assert np.isclose(arc_len, exact, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
