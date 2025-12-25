"""
kinematics 模块单元测试
"""

import numpy as np
import pytest

from cnc_five_axis_interpolation.core.kinematics import (
    inverse_kinematics_ac,
    batch_inverse_kinematics_ac,
)


class TestInverseKinematicsAC:
    """A-C 配置逆运动学测试"""

    def test_z_axis_orientation(self):
        """测试刀轴沿 z 轴正方向 (A=0)"""
        P = np.array([10.0, 20.0, 30.0])
        O = np.array([0.0, 0.0, 1.0])

        XYZ, A, C = inverse_kinematics_ac(P, O)

        # A 应为 0 (刀轴沿 z 轴)
        assert np.isclose(A, 0.0, atol=1e-10)

    def test_tilted_45_degrees(self):
        """测试刀轴倾斜 45 度"""
        P = np.array([0.0, 0.0, 0.0])
        # 在 YZ 平面倾斜 45 度
        O = np.array([0.0, np.sin(np.radians(45)), np.cos(np.radians(45))])

        XYZ, A, C = inverse_kinematics_ac(P, O)

        # A 应为 45 度
        assert np.isclose(np.degrees(A), 45.0, atol=1e-6)

    def test_angle_ranges(self):
        """测试角度范围"""
        P = np.array([0.0, 0.0, 0.0])

        # 多个测试方向
        test_cases = [
            ([0, 0, 1], 0),  # A = 0°
            ([0, 1, 0], 90),  # A = 90°
            ([1, 0, 0], 90),  # A = 90°
            ([0, 0, -1], 180),  # A = 180°
        ]

        for orientation, expected_A_deg in test_cases:
            O = np.array(orientation, dtype=float)
            O = O / np.linalg.norm(O)
            XYZ, A, C = inverse_kinematics_ac(P, O)
            assert np.isclose(np.degrees(A), expected_A_deg, atol=1e-6), \
                f"Failed for orientation {orientation}: expected A={expected_A_deg}°, got {np.degrees(A):.2f}°"

    def test_output_shape(self):
        """测试输出形状"""
        P = np.array([1.0, 2.0, 3.0])
        O = np.array([0.1, 0.2, 0.97])
        O = O / np.linalg.norm(O)

        XYZ, A, C = inverse_kinematics_ac(P, O)

        assert XYZ.shape == (3,)
        assert isinstance(A, (float, np.floating))
        assert isinstance(C, (float, np.floating))

    def test_with_offsets(self):
        """测试带偏移参数"""
        P = np.array([0.0, 0.0, 0.0])
        O = np.array([0.0, 0.0, 1.0])

        L_ac_z = 100.0
        L_Tya_z = 200.0

        XYZ, A, C = inverse_kinematics_ac(P, O, L_ac_z=L_ac_z, L_Tya_z=L_Tya_z)

        # 当 A=0 时，Z = cos(0)*P_z + cos(0)*L_ac_z + L_Tya_z = 0 + 100 + 200 = 300
        assert np.isclose(XYZ[2], L_ac_z + L_Tya_z, atol=1e-10)


class TestBatchInverseKinematics:
    """批量逆运动学测试"""

    def test_batch_consistency(self):
        """测试批量与单点结果一致"""
        positions = np.array([
            [0, 0, 0],
            [10, 20, 30],
            [5, 5, 5],
        ], dtype=float)

        orientations = np.array([
            [0, 0, 1],
            [0.1, 0.2, 0.97],
            [0.3, 0.4, 0.866],
        ], dtype=float)
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        XYZ_batch, A_batch, C_batch = batch_inverse_kinematics_ac(positions, orientations)

        for i in range(len(positions)):
            XYZ_single, A_single, C_single = inverse_kinematics_ac(
                positions[i], orientations[i]
            )
            np.testing.assert_allclose(XYZ_batch[i], XYZ_single, atol=1e-10)
            assert np.isclose(A_batch[i], A_single, atol=1e-10)
            assert np.isclose(C_batch[i], C_single, atol=1e-10)

    def test_batch_output_shape(self):
        """测试批量输出形状"""
        N = 50
        positions = np.random.rand(N, 3) * 100
        orientations = np.random.rand(N, 3)
        orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

        XYZ, A, C = batch_inverse_kinematics_ac(positions, orientations)

        assert XYZ.shape == (N, 3)
        assert A.shape == (N,)
        assert C.shape == (N,)


class TestWithRealData:
    """使用真实数据集测试"""

    def test_ijms2021_data(self):
        """测试 IJMS2021 数据"""
        from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

        positions, orientations, _ = ijms2021_fan_shaped_path()

        XYZ, A, C = batch_inverse_kinematics_ac(positions, orientations)

        # 验证输出有效
        assert not np.any(np.isnan(XYZ))
        assert not np.any(np.isnan(A))
        assert not np.any(np.isnan(C))

        # A 角度应在 [0, π] 范围内
        assert np.all(A >= 0)
        assert np.all(A <= np.pi)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
