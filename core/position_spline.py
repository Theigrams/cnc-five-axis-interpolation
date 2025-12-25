"""
position_spline - C³ 连续位置样条生成

基于论文 Section 2: C³ tool tip position spline generation

实现:
1. 五次B样条拟合刀尖位置 (Eq.1-7)
2. 9阶多项式进给校正 (Eq.11-22)
3. 自适应细分 (Eq.23)
"""

import numpy as np


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

        # TODO: 实现样条拟合
        self.spline = None
        self.feed_correction = None
        self.length = 0.0

    def fit(self):
        """拟合五次B样条和进给校正多项式。"""
        raise NotImplementedError

    def evaluate(self, l: float) -> np.ndarray:
        """在弧长 l 处评估位置。"""
        raise NotImplementedError
