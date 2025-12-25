# 球坐标姿态推导 (Eq.25-31)

本文档详细推导球坐标表示刀轴姿态的数学原理。

## 1. 为什么用球坐标

### 1.1 单位向量约束

刀轴姿态 $\mathbf{O} = (O_i, O_j, O_k)^T$ 是单位向量，满足：

$$\|\mathbf{O}\| = \sqrt{O_i^2 + O_j^2 + O_k^2} = 1$$

如果直接在笛卡尔坐标下拟合 B 样条，插值结果可能不满足单位长度约束。

### 1.2 球坐标的优势

球坐标 $(\theta, \phi)$ 自动满足单位长度约束：

$$\mathbf{O} = \begin{pmatrix} \sin\theta \cos\phi \\ \sin\theta \sin\phi \\ \cos\theta \end{pmatrix}$$

其中：

- $\theta \in [0, \pi]$：极角（与 Z 轴夹角）
- $\phi \in (-\pi, \pi]$：方位角（在 XY 平面的投影与 X 轴夹角）

## 2. 笛卡尔 ↔ 球坐标转换 (Eq.25)

### 2.1 笛卡尔 → 球坐标

$$\theta = \arccos(O_k) \tag{Eq.25a}$$

$$\phi = \text{atan2}(O_j, O_i) \tag{Eq.25b}$$

### 2.2 球坐标 → 笛卡尔

$$O_i = \sin\theta \cos\phi$$

$$O_j = \sin\theta \sin\phi$$

$$O_k = \cos\theta$$

### 2.3 角度展开

$\phi$ 可能在 $\pm\pi$ 处跳变。需要角度展开（unwrap）保证连续性：

```python
def unwrap_angles(phi):
    """展开角度，消除 ±π 跳变"""
    diff = np.diff(phi)
    # 跳变超过 π 时修正
    diff[diff > np.pi] -= 2 * np.pi
    diff[diff < -np.pi] += 2 * np.pi
    return np.concatenate([[phi[0]], phi[0] + np.cumsum(diff)])
```

## 3. 角度参数化 (Eq.28)

### 3.1 姿态变化角

相邻姿态间的变化角定义为：

$$\Delta\alpha_k = \arccos(\mathbf{O}_k \cdot \mathbf{O}_{k-1})$$

这是两个单位向量间的夹角。

### 3.2 角度参数化公式 (Eq.28)

$$\bar{w}_0 = 0$$

$$\bar{w}_k = \bar{w}_{k-1} + \frac{\Delta\alpha_k}{\sum_{i=1}^{N-1} \Delta\alpha_i}, \quad k = 1, \ldots, N-1$$

$$\bar{w}_{N-1} = 1$$

### 3.3 与向心参数化的对比

| 参数化方法 | 适用对象 | 度量 |
|-----------|---------|------|
| 向心参数化 | 位置 | 点间欧氏距离的平方根 |
| **角度参数化** | **姿态** | **姿态间变化角** |

角度参数化更适合球面上的曲线。

## 4. (θ, φ) 平面 B 样条拟合 (Eq.29-30)

### 4.1 拟合过程

1. 将所有姿态转换为球坐标 $\{(\theta_k, \phi_k)\}$
2. 对 $\phi$ 进行角度展开
3. 将 $(\theta, \phi)$ 作为二维点在参数 $w$ 下拟合五次 B 样条

### 4.2 球坐标样条

$$\begin{pmatrix} \theta(w) \\ \phi(w) \end{pmatrix} = \sum_{j=0}^{N-1} N_{j,5}(w) \begin{pmatrix} \Theta_j \\ \Phi_j \end{pmatrix} \tag{Eq.29}$$

其中 $(\Theta_j, \Phi_j)$ 是球坐标平面上的控制点。

### 4.3 插值条件 (Eq.30)

$$\theta(\bar{w}_k) = \theta_k, \quad \phi(\bar{w}_k) = \phi_k, \quad k = 0, \ldots, N-1 \tag{Eq.30}$$

## 5. 球坐标逆映射 (Eq.31)

评估姿态时，从球坐标样条得到 $(\theta, \phi)$，再转换回笛卡尔坐标：

$$\mathbf{O}(w) = \begin{pmatrix} \sin\theta(w) \cos\phi(w) \\ \sin\theta(w) \sin\phi(w) \\ \cos\theta(w) \end{pmatrix} \tag{Eq.31}$$

结果自动是单位向量（无需归一化）。

## 6. 代码实现

```python
def batch_cartesian_to_spherical(orientations):
    """批量笛卡尔转球坐标 (Eq.25)"""
    theta = np.arccos(np.clip(orientations[:, 2], -1, 1))
    phi = np.arctan2(orientations[:, 1], orientations[:, 0])
    return theta, phi


def spherical_to_cartesian(theta, phi):
    """球坐标转笛卡尔 (Eq.31)"""
    sin_theta = np.sin(theta)
    return np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        np.cos(theta)
    ])


def angular_parameterization(orientations):
    """角度参数化 (Eq.28)"""
    N = len(orientations)

    # 计算相邻姿态间的变化角
    dots = np.sum(orientations[:-1] * orientations[1:], axis=1)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)

    total = angles.sum()
    if total < 1e-12:
        return np.linspace(0, 1, N)

    w_bar = np.zeros(N)
    w_bar[1:] = np.cumsum(angles) / total
    w_bar[-1] = 1.0

    return w_bar


def fit_orientation_bspline(orientations):
    """拟合姿态 B 样条 (Eq.29-30)"""
    # 转换为球坐标
    theta, phi = batch_cartesian_to_spherical(orientations)
    phi = unwrap_angles(phi)
    spherical_points = np.column_stack([theta, phi])

    # 角度参数化
    w_bar = angular_parameterization(orientations)

    # 拟合 B 样条
    spline, knots = fit_interpolating_bspline(spherical_points, w_bar, degree=5)

    return spline, knots, w_bar
```

## 7. 特殊情况处理

### 7.1 极点问题

当 $\theta \approx 0$（姿态接近 Z 轴）时，$\phi$ 不稳定。实际中需要检测并处理。

### 7.2 $\phi$ 跳变

$\phi$ 在 $\pm\pi$ 处跳变，必须使用角度展开。否则 B 样条拟合会产生错误的插值。

## 总结

球坐标表示通过将单位向量约束内置于坐标系统中，避免了归一化问题。角度参数化基于姿态变化角分配参数，比向心参数化更适合姿态插值。在 $(\theta, \phi)$ 平面拟合 B 样条后，逆映射回笛卡尔坐标即可得到单位姿态向量。
