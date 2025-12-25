# 弧长参数化推导 (Eq.8-10)

本文档详细推导弧长计算和自适应 Simpson 积分的数学原理。

## 1. 弧长积分定义 (Eq.8)

给定参数曲线 $\mathbf{P}(u)$，从参数 $0$ 到 $u$ 的弧长定义为：

$$s(u) = \int_0^u \|\mathbf{P}'(\tau)\| d\tau \tag{Eq.8}$$

其中 $\mathbf{P}'(u) = \frac{d\mathbf{P}}{du}$ 是曲线的切向量。

### 1.1 被积函数

弧长微分 $ds = \|\mathbf{P}'(u)\| du$，被积函数为：

$$f(u) = \|\mathbf{P}'(u)\| = \sqrt{P_x'(u)^2 + P_y'(u)^2 + P_z'(u)^2}$$

对于五次 B 样条，$\mathbf{P}'(u)$ 是四次多项式，可以通过样条导数直接计算。

## 2. 自适应 Simpson 积分 (Eq.9-10)

### 2.1 Simpson 法则

Simpson 法则使用抛物线近似被积函数：

$$\int_a^b f(x) dx \approx \frac{b-a}{6} \left[ f(a) + 4f\left(\frac{a+b}{2}\right) + f(b) \right]$$

### 2.2 自适应细分策略

自适应 Simpson 积分通过递归细分提高精度 (Eq.9-10)：

1. 计算整个区间的 Simpson 近似 $S_1$
2. 将区间二分，计算两个子区间的 Simpson 近似之和 $S_2$
3. 估计误差 $E = |S_2 - S_1| / 15$
4. 如果 $E < \epsilon$，接受 $S_2$；否则递归细分

```
adaptive_simpson(f, a, b, ε):
    c = (a + b) / 2
    S_whole = simpson(f, a, b)
    S_left = simpson(f, a, c)
    S_right = simpson(f, c, b)
    S_sum = S_left + S_right

    if |S_sum - S_whole| / 15 < ε:
        return S_sum
    else:
        return adaptive_simpson(f, a, c, ε/2) + adaptive_simpson(f, c, b, ε/2)
```

### 2.3 误差估计

Richardson 外推给出误差估计：

$$E \approx \frac{|S_2 - S_1|}{15}$$

这是因为 Simpson 法则的误差阶为 $O(h^5)$，二分后误差减少为 $1/32$。

## 3. 弧长表构建

### 3.1 目的

弧长表存储 $(u_i, s_i)$ 对，用于后续的弧长参数化映射。

### 3.2 构建过程

在自适应积分过程中，记录每个细分区间的端点和弧长：

```python
def compute_arc_length_table(spline, tol=1e-9):
    """计算弧长表"""
    spline_deriv = spline.derivative()

    def integrand(u):
        return np.linalg.norm(spline_deriv(u))

    intervals = []  # 存储 (u_start, u_end, length)
    total_length = adaptive_simpson(integrand, 0, 1, tol, intervals)

    # 构建累积弧长表
    u_values = [0.0]
    arc_lengths = [0.0]
    cumulative = 0.0

    for _, u_end, length in intervals:
        cumulative += length
        u_values.append(u_end)
        arc_lengths.append(cumulative)

    return np.array(u_values), np.array(arc_lengths), total_length
```

### 3.3 表的性质

- $u_0 = 0$, $s_0 = 0$
- $u_n = 1$, $s_n = S$（总弧长）
- $(u_i, s_i)$ 单调递增
- 区间分布自适应于曲率变化

## 4. 弧长查找

给定弧长 $l$，需要找到对应的参数 $u$，使得 $s(u) = l$。

### 4.1 二分查找

1. 在弧长表中找到 $s_{i-1} \leq l < s_i$ 的区间
2. 在区间 $[u_{i-1}, u_i]$ 内使用牛顿法或二分法精确求解

### 4.2 为什么需要进给校正多项式

直接查表 + 插值的方法：
- 只能达到 C⁰ 或 C¹ 连续
- 无法保证 C³ 连续性

因此需要 9 阶进给校正多项式来实现 C³ 连续的 $u(l)$ 映射。

## 5. 代码实现

```python
def adaptive_simpson(f, a, b, tol, intervals=None):
    """自适应 Simpson 积分"""
    c = (a + b) / 2
    h = b - a

    fa, fc, fb = f(a), f(c), f(b)
    S_whole = h / 6 * (fa + 4*fc + fb)

    d = (a + c) / 2
    e = (c + b) / 2
    fd, fe = f(d), f(e)

    S_left = (c - a) / 6 * (fa + 4*fd + fc)
    S_right = (b - c) / 6 * (fc + 4*fe + fb)
    S_sum = S_left + S_right

    error = abs(S_sum - S_whole) / 15

    if error < tol:
        if intervals is not None:
            intervals.append((a, b, S_sum))
        return S_sum
    else:
        left = adaptive_simpson(f, a, c, tol/2, intervals)
        right = adaptive_simpson(f, c, b, tol/2, intervals)
        return left + right
```

## 总结

弧长计算通过自适应 Simpson 积分实现高精度数值积分。积分过程中构建的弧长表为后续的弧长参数化提供基础数据。然而，直接使用弧长表无法保证 C³ 连续性，需要进给校正多项式来实现平滑映射。
