# 进给校正多项式推导 (Eq.11-23)

本文档详细推导 9 阶进给校正多项式的数学原理，这是实现 C³ 连续弧长参数化的核心。

## 1. 问题定义

### 1.1 弧长参数化目标

给定 B 样条曲线 $\mathbf{P}(u)$，我们需要找到映射 $u(l)$，使得：

$$\mathbf{P}(u(l))$$

是弧长 $l$ 的函数，且满足 C³ 连续性。

### 1.2 为什么需要 C³ 连续

| 连续性等级 | 物理意义 | 影响 |
|-----------|----------|------|
| C⁰ | 位置连续 | 轨迹无跳变 |
| C¹ | 速度连续 | 无速度突变 |
| C² | 加速度连续 | 无加速度突变 |
| **C³** | **Jerk 连续** | **减少机床振动** |

## 2. 9 阶多项式形式 (Eq.11-13)

### 2.1 归一化形式

在区间 $[0, S]$ 上，定义归一化变量 $\sigma = l/S$，9 阶多项式为：

$$u(\sigma) = \sum_{i=0}^{9} a_i \sigma^i \tag{Eq.11}$$

其中 $\sigma \in [0, 1]$，$u \in [u_{start}, u_{end}]$。

### 2.2 矩阵形式 (Eq.12-13)

设采样点 $\{(\sigma_k, u_k)\}_{k=0}^{M-1}$，构造 Vandermonde 矩阵：

$$\mathbf{\Phi} = \begin{bmatrix}
\sigma_0^9 & \sigma_0^8 & \cdots & \sigma_0 & 1 \\
\sigma_1^9 & \sigma_1^8 & \cdots & \sigma_1 & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\sigma_{M-1}^9 & \sigma_{M-1}^8 & \cdots & \sigma_{M-1} & 1
\end{bmatrix} \tag{Eq.12}$$

最小二乘目标：

$$\min_{\mathbf{a}} \|\mathbf{\Phi} \mathbf{a} - \mathbf{u}\|^2 \tag{Eq.13}$$

## 3. C³ 边界条件 (Eq.14-18)

### 3.1 导数关系推导

设 $f(u) = \|\mathbf{P}'(u)\|$，由弧长微分 $dl = f(u) du$，得：

$$\frac{du}{dl} = \frac{1}{f(u)} \tag{Eq.14}$$

对 $l$ 求导：

$$\frac{d^2u}{dl^2} = -\frac{f'(u)}{f(u)^3} \tag{Eq.15}$$

$$\frac{d^3u}{dl^3} = \frac{3f'(u)^2 - f''(u)f(u)}{f(u)^5} \tag{Eq.16}$$

### 3.2 边界约束

在 $l = 0$（$\sigma = 0$）和 $l = S$（$\sigma = 1$）处，多项式必须满足：

| 约束 | $\sigma = 0$ | $\sigma = 1$ |
|------|--------------|--------------|
| 值 | $u(0) = u_{start}$ | $u(1) = u_{end}$ |
| 一阶导 | $u'(0) = u_l(0) \cdot S$ | $u'(1) = u_l(1) \cdot S$ |
| 二阶导 | $u''(0) = u_{ll}(0) \cdot S^2$ | $u''(1) = u_{ll}(1) \cdot S^2$ |
| 三阶导 | $u'''(0) = u_{lll}(0) \cdot S^3$ | $u'''(1) = u_{lll}(1) \cdot S^3$ |

其中 $u_l, u_{ll}, u_{lll}$ 由 (Eq.14-16) 计算。

### 3.3 约束矩阵构造 (Eq.17-18)

8 个约束可写成矩阵形式：

$$\mathbf{\Omega} \mathbf{a} = \mathbf{\eta} \tag{Eq.17}$$

其中 $\mathbf{\Omega}$ 是 $8 \times 10$ 约束矩阵：

$$\mathbf{\Omega} = \begin{bmatrix}
0 & 0 & \cdots & 0 & 1 & \leftarrow u(0) \\
0 & 0 & \cdots & 1 & 0 & \leftarrow u'(0) \\
0 & 0 & \cdots & 2 & 0 & 0 & \leftarrow u''(0) \\
0 & \cdots & 6 & 0 & 0 & 0 & \leftarrow u'''(0) \\
1 & 1 & \cdots & 1 & 1 & \leftarrow u(1) \\
9 & 8 & \cdots & 1 & 0 & \leftarrow u'(1) \\
72 & 56 & \cdots & 2 & 0 & \leftarrow u''(1) \\
504 & 336 & \cdots & 6 & 0 & \leftarrow u'''(1)
\end{bmatrix} \tag{Eq.18}$$

## 4. 约束最小二乘 (Eq.19-22)

### 4.1 KKT 系统

约束最小二乘问题：

$$\min_{\mathbf{a}} \|\mathbf{\Phi} \mathbf{a} - \mathbf{u}\|^2 \quad \text{s.t.} \quad \mathbf{\Omega} \mathbf{a} = \mathbf{\eta} \tag{Eq.19}$$

引入拉格朗日乘子 $\mathbf{\lambda}$，KKT 条件为：

$$\begin{bmatrix}
\mathbf{\Phi}^T \mathbf{\Phi} & \mathbf{\Omega}^T \\
\mathbf{\Omega} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\mathbf{a} \\
\mathbf{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{\Phi}^T \mathbf{u} \\
\mathbf{\eta}
\end{bmatrix} \tag{Eq.20-22}$$

### 4.2 求解

求解 $18 \times 18$ 线性系统得到系数 $\mathbf{a}$。

## 5. 自适应细分 (Eq.23)

### 5.1 MSE 误差准则

拟合后计算均方误差：

$$\text{MSE} = \frac{1}{M} \sum_{k=0}^{M-1} (u(\sigma_k) - u_k)^2 \tag{Eq.23}$$

### 5.2 细分策略

```
fit_segment(start, end):
    arc_lengths = arc_table[start:end+1]
    u_values = u_table[start:end+1]

    poly = fit_polynomial(arc_lengths, u_values)
    mse = compute_mse(poly, arc_lengths, u_values)

    if mse < tolerance:
        return [poly]
    else:
        mid = (start + end) // 2
        return fit_segment(start, mid) + fit_segment(mid, end)
```

### 5.3 分段策略的优势

- 曲率平缓区域：单段多项式足够
- 曲率剧烈区域：自动细分保证精度
- 整体保持 C³ 连续（边界条件约束）

## 6. 代码实现

```python
def compute_derivatives_at_u(spline, u, S):
    """计算边界导数 (Eq.14-16)"""
    spline_d1 = spline.derivative(1)
    spline_d2 = spline.derivative(2)
    spline_d3 = spline.derivative(3)

    P1 = spline_d1(u)
    P2 = spline_d2(u)
    P3 = spline_d3(u)

    f = np.linalg.norm(P1)
    if f < 1e-12:
        return np.array([S, 0.0, 0.0])

    fp = np.dot(P1, P2) / f
    fpp = (np.dot(P2, P2) + np.dot(P1, P3) - fp**2) / f

    u_l = 1.0 / f
    u_ll = -fp / f**3
    u_lll = (3*fp**2 - fpp*f) / f**5

    return np.array([u_l * S, u_ll * S**2, u_lll * S**3])

def fit_feed_correction_polynomial(arc_lengths, u_values, spline):
    """拟合 9 阶进给校正多项式 (Eq.11-22)"""
    S = arc_lengths[-1]
    sigma = arc_lengths / S

    # Vandermonde 矩阵
    powers = np.arange(9, -1, -1)
    Phi = sigma[:, np.newaxis] ** powers

    # 边界导数
    derivs_0 = compute_derivatives_at_u(spline, u_values[0], S)
    derivs_1 = compute_derivatives_at_u(spline, u_values[-1], S)

    # 约束矩阵
    Omega = np.zeros((8, 10))
    Omega[0, 9] = 1  # u(0)
    Omega[1, 8] = 1  # u'(0)
    Omega[2, 7] = 2  # u''(0)
    Omega[3, 6] = 6  # u'''(0)
    Omega[4, :] = 1  # u(1)
    Omega[5, :] = np.arange(9, -1, -1)  # u'(1)
    Omega[6, :] = [72, 56, 42, 30, 20, 12, 6, 2, 0, 0]  # u''(1)
    Omega[7, :] = [504, 336, 210, 120, 60, 24, 6, 0, 0, 0]  # u'''(1)

    # 右侧向量
    eta = np.array([0, derivs_0[0], derivs_0[1], derivs_0[2],
                    1, derivs_1[0], derivs_1[1], derivs_1[2]])

    # KKT 系统
    KKT = np.zeros((18, 18))
    KKT[:10, :10] = Phi.T @ Phi
    KKT[:10, 10:] = Omega.T
    KKT[10:, :10] = Omega

    rhs = np.zeros(18)
    rhs[:10] = Phi.T @ u_values
    rhs[10:] = eta

    solution = np.linalg.solve(KKT, rhs)
    coeffs = solution[:10]

    return coeffs
```

## 总结

9 阶进给校正多项式通过约束最小二乘法，在满足 C³ 边界条件的前提下拟合 $u(l)$ 映射。自适应细分策略确保在曲率变化剧烈处也能保持高精度。这是实现 C³ 连续弧长参数化的核心算法。
