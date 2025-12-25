# cnc_five_axis_interpolation

基于 Yuen et al. (2013) 的 C³ 连续五轴轨迹生成算法实现。

## 概述

本包实现了论文 "Smooth trajectory generation for five-axis machine tools" 中的核心算法，包括：

- **C³ 连续位置样条**：五次 B 样条 + 9 阶进给校正多项式
- **C³ 连续姿态样条**：球坐标 B 样条 + 7 阶 Bézier 重参数化
- **A-C 配置逆运动学**：工件坐标到机床坐标的变换

## 快速开始

```python
from cnc_five_axis_interpolation import FiveAxisPath
from cnc_five_axis_interpolation.datasets import ijms2021_fan_shaped_path

# 加载数据
positions, orientations, _ = ijms2021_fan_shaped_path()

# 拟合轨迹
path = FiveAxisPath(positions, orientations).fit()

# 在任意弧长处评估
pos, ori = path.evaluate(50.0)  # l = 50mm

# 获取机床坐标
XYZ, A, C = path.evaluate_machine_coords(50.0, L_ac_z=70.0, L_Tya_z=150.0)
```

## 文档结构

### 教程 (Notebooks)

实用导向，快速上手 + 原理概述：

| Notebook | 内容 |
|----------|------|
| [01_position_spline](notebooks/01_position_spline.ipynb) | 位置样条使用与原理 |
| [02_orientation_spline](notebooks/02_orientation_spline.ipynb) | 姿态样条使用与原理 |
| [03_inverse_kinematics](notebooks/03_inverse_kinematics.ipynb) | 逆运动学使用与原理 |
| [04_end_to_end_example](notebooks/04_end_to_end_example.ipynb) | 完整五轴轨迹生成流程 |

### 数学推导 (Theory)

理论导向，详细公式推导：

| 文档 | 覆盖公式 |
|------|----------|
| [bspline_interpolation](theory/bspline_interpolation.md) | Eq.1-7 |
| [arc_length_parameterization](theory/arc_length_parameterization.md) | Eq.8-10 |
| [feed_correction_polynomial](theory/feed_correction_polynomial.md) | Eq.11-23 |
| [spherical_orientation](theory/spherical_orientation.md) | Eq.25-31 |
| [bezier_reparameterization](theory/bezier_reparameterization.md) | Eq.32-41 |
| [ac_kinematics](theory/ac_kinematics.md) | Eq.42 |

## 为什么需要 C³ 连续性？

| 连续性 | 物理意义 | 影响 |
|--------|----------|------|
| C⁰ | 位置连续 | 轨迹无跳变 |
| C¹ | 速度连续 | 无速度突变 |
| C² | 加速度连续 | 无加速度突变 |
| **C³** | **Jerk 连续** | **减少机床振动，提高加工质量** |

## 算法流程

```
离散数据点 (P, O)
       ↓
五次 B 样条拟合 (Eq.1-7)
       ↓
弧长计算 (Eq.8-10)
       ↓
进给校正/Bézier 重参数化 (Eq.11-41)
       ↓
C³ 连续弧长参数化轨迹
       ↓
逆运动学 (Eq.42)
       ↓
机床五轴运动 (X, Y, Z, A, C)
```

## 本地构建文档

```bash
# 安装依赖
pip install mkdocs mkdocs-material mkdocs-jupyter

# 本地预览
cd cnc_five_axis_interpolation/docs
mkdocs serve

# 构建静态站点
mkdocs build
```

## 参考文献

Yuen, A., Zhang, K., & Altintas, Y. (2013). Smooth trajectory generation for five-axis machine tools. *International Journal of Machine Tools and Manufacture*, 71, 11-19.
