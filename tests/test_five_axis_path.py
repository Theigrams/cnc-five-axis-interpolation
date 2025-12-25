"""
algorithm (FiveAxisPath) 模块单元测试
"""

import numpy as np
import pytest

from cnc_five_axis_interpolation import FiveAxisPath


class TestFiveAxisPath:
    """FiveAxisPath 主类测试"""

    @pytest.fixture
    def simple_path_data(self):
        """简单测试路径数据"""
        t = np.linspace(0, np.pi, 10)
        positions = np.column_stack([
            10 * np.cos(t),
            10 * np.sin(t),
            t
        ])
        orientations = np.column_stack([
            np.sin(t / 2),
            np.zeros_like(t),
            np.cos(t / 2)
        ])
        return positions, orientations

    def test_initialization(self, simple_path_data):
        """测试初始化"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)

        assert path.N == len(positions)
        assert path.position_spline is None  # 未拟合
        assert path.length == 0.0

    def test_fit(self, simple_path_data):
        """测试拟合"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)
        path.fit()

        assert path.position_spline is not None
        assert path.orientation_spline is not None
        assert path.length > 0

    def test_evaluate_at_boundaries(self, simple_path_data):
        """测试边界点评估"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)
        path.fit()

        # l=0 应接近第一个点
        pos_start, ori_start = path.evaluate(0)
        np.testing.assert_allclose(pos_start, positions[0], atol=1e-3)

        # l=length 应接近最后一个点
        pos_end, ori_end = path.evaluate(path.length)
        np.testing.assert_allclose(pos_end, positions[-1], atol=1e-3)

    def test_orientations_normalized(self, simple_path_data):
        """测试输出姿态归一化"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)
        path.fit()

        l_samples = np.linspace(0, path.length, 50)
        _, ori_samples = path.evaluate_batch(l_samples)

        norms = np.linalg.norm(ori_samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_machine_coords(self, simple_path_data):
        """测试机床坐标评估"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)
        path.fit()

        XYZ, A, C = path.evaluate_machine_coords(path.length / 2)

        assert XYZ.shape == (3,)
        assert 0 <= A <= np.pi
        assert isinstance(C, (float, np.floating))

    def test_batch_machine_coords(self, simple_path_data):
        """测试批量机床坐标评估"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)
        path.fit()

        l_values = np.linspace(0, path.length, 20)
        XYZ, A, C = path.evaluate_machine_coords_batch(l_values)

        assert XYZ.shape == (20, 3)
        assert A.shape == (20,)
        assert C.shape == (20,)

    def test_sample_uniform(self, simple_path_data):
        """测试均匀采样"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)
        path.fit()

        l_vals, pos_samples, ori_samples = path.sample_uniform(50)

        assert len(l_vals) == 50
        assert pos_samples.shape == (50, 3)
        assert ori_samples.shape == (50, 3)

        # 弧长应均匀分布
        np.testing.assert_allclose(l_vals, np.linspace(0, path.length, 50))

    def test_repr(self, simple_path_data):
        """测试字符串表示"""
        positions, orientations = simple_path_data
        path = FiveAxisPath(positions, orientations)

        assert "not fitted" in repr(path)

        path.fit()
        assert "fitted" in repr(path)


class TestWithRealData:
    """使用真实数据集测试"""

    def test_ijms2021_data(self):
        """测试 IJMS2021 数据"""
        from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

        positions, orientations, _ = ijms2021_fan_shaped_path()
        path = FiveAxisPath(positions, orientations)
        path.fit()

        # 验证弧长合理
        assert path.length > 0

        # 验证插值精度 - 位置
        for i, u in enumerate(path.position_spline.u_bar):
            interp_pos = path.position_spline.spline(u)
            error = np.linalg.norm(interp_pos - positions[i])
            assert error < 1e-10

        # 验证均匀采样
        l_vals, pos_samples, ori_samples = path.sample_uniform(100)
        assert len(l_vals) == 100

    def test_jcde2022_data(self):
        """测试 JCDE2022 数据"""
        from cnc_five_axis_interpolation.datasets import jcde2022_dual_bspline_path

        positions, orientations, _ = jcde2022_dual_bspline_path(50)
        path = FiveAxisPath(positions, orientations)
        path.fit()

        # 验证弧长合理
        assert path.length > 0

        # 验证机床坐标有效
        XYZ, A, C = path.evaluate_machine_coords_batch(np.linspace(0, path.length, 20))
        assert not np.any(np.isnan(XYZ))
        assert not np.any(np.isnan(A))
        assert not np.any(np.isnan(C))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
