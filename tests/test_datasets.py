"""
datasets 模块单元测试
"""

import numpy as np
import pytest

from cnc_five_axis_interpolation.datasets import (
    IJMS2021Constraints,
    JCDE2022Constraints,
    ijms2021_fan_shaped_path,
    jcde2022_dual_bspline_path,
)


class TestIJMS2021:
    """IJMS 2021 数据集测试"""

    def test_ijms2021_returns_correct_shapes(self):
        """测试返回数据形状正确"""
        positions, orientations, constraints = ijms2021_fan_shaped_path()
        assert positions.ndim == 2
        assert orientations.ndim == 2
        assert positions.shape[1] == 3
        assert orientations.shape[1] == 3
        assert len(positions) == len(orientations)

    def test_ijms2021_orientations_normalized(self):
        """测试刀轴矢量已归一化"""
        _, orientations, _ = ijms2021_fan_shaped_path()
        norms = np.linalg.norm(orientations, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_ijms2021_constraints_type(self):
        """测试约束参数类型正确"""
        _, _, constraints = ijms2021_fan_shaped_path()
        assert isinstance(constraints, IJMS2021Constraints)
        assert constraints.interpolation_period_s > 0
        assert constraints.chord_error_limit_mm > 0

    def test_ijms2021_data_not_empty(self):
        """测试数据非空"""
        positions, orientations, _ = ijms2021_fan_shaped_path()
        assert len(positions) > 0
        assert len(orientations) > 0


class TestJCDE2022:
    """JCDE 2022 数据集测试"""

    def test_jcde2022_returns_correct_shapes(self):
        """测试返回数据形状正确"""
        positions, orientations, constraints = jcde2022_dual_bspline_path(100)
        assert positions.shape == (100, 3)
        assert orientations.shape == (100, 3)

    def test_jcde2022_orientations_normalized(self):
        """测试刀轴矢量已归一化"""
        _, orientations, _ = jcde2022_dual_bspline_path(50)
        norms = np.linalg.norm(orientations, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_jcde2022_constraints_type(self):
        """测试约束参数类型正确"""
        _, _, constraints = jcde2022_dual_bspline_path()
        assert isinstance(constraints, JCDE2022Constraints)
        assert constraints.interpolation_period_s > 0
        assert constraints.chord_error_limit_mm > 0

    def test_jcde2022_different_sample_counts(self):
        """测试不同采样数量"""
        for n in [10, 50, 200]:
            positions, orientations, _ = jcde2022_dual_bspline_path(n)
            assert positions.shape == (n, 3)
            assert orientations.shape == (n, 3)

    def test_jcde2022_curve_continuity(self):
        """测试曲线连续性（相邻点距离有界）"""
        positions, _, _ = jcde2022_dual_bspline_path(1000)
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        # 相邻点距离应该较小且变化平缓
        assert np.max(distances) < 1.0  # 最大步长 < 1mm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
