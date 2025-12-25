# B样条插值数学推导 (Eq.1-7)

本文档详细推导五次 B 样条插值的数学原理。

## 1. B样条基础

### 1.1 基函数递归定义

$p$ 次 B 样条基函数 $N_{i,p}(u)$ 由 Cox-de Boor 递归公式定义：

$$N_{i,0}(u) = \begin{cases} 1 & u_i \leq u < u_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$N_{i,p}(u) = \frac{u - u_i}{u_{i+p} - u_i} N_{i,p-1}(u) + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(u)$$

其中 $\{u_i\}$ 是节点向量。

### 1.2 B样条曲线

给定 $n+1$ 个控制点 $\{\mathbf{Q}_0, \mathbf{Q}_1, \ldots, \mathbf{Q}_n\}$，$p$ 次 B 样条曲线定义为：

$$\mathbf{P}(u) = \sum_{i=0}^{n} N_{i,p}(u) \mathbf{Q}_i$$

## 2. 五次 B 样条插值 (Eq.1-3)

### 2.1 插值问题

给定 $N$ 个数据点 $\{\mathbf{P}_0, \mathbf{P}_1, \ldots, \mathbf{P}_{N-1}\}$ 和对应的参数值 $\{\bar{u}_0, \bar{u}_1, \ldots, \bar{u}_{N-1}\}$，求控制点 $\{\mathbf{Q}_j\}$ 使得：

$$\mathbf{P}(\bar{u}_k) = \mathbf{P}_k, \quad k = 0, 1, \ldots, N-1 \tag{Eq.1}$$

### 2.2 线性系统构造

将插值条件写成矩阵形式 (Eq.2)：

$$\begin{bmatrix} N_{0,5}(\bar{u}_0) & N_{1,5}(\bar{u}_0) & \cdots & N_{N-1,5}(\bar{u}_0) \\ N_{0,5}(\bar{u}_1) & N_{1,5}(\bar{u}_1) & \cdots & N_{N-1,5}(\bar{u}_1) \\ \vdots & \vdots & \ddots & \vdots \\ N_{0,5}(\bar{u}_{N-1}) & N_{1,5}(\bar{u}_{N-1}) & \cdots & N_{N-1,5}(\bar{u}_{N-1}) \end{bmatrix} \begin{bmatrix} \mathbf{Q}_0 \\ \mathbf{Q}_1 \\ \vdots \\ \mathbf{Q}_{N-1} \end{bmatrix} = \begin{bmatrix} \mathbf{P}_0 \\ \mathbf{P}_1 \\ \vdots \\ \mathbf{P}_{N-1} \end{bmatrix}$$

简写为：

$$\mathbf{N} \cdot \mathbf{Q} = \mathbf{P} \tag{Eq.3}$$

### 2.3 求解控制点

求解线性系统得到控制点：

$$\mathbf{Q} = \mathbf{N}^{-1} \cdot \mathbf{P}$$

实际实现中使用 LU 分解或其他数值方法求解。

## 3. 向心参数化 (Eq.4-7)

参数化方法决定了参数值 $\bar{u}_k$ 如何分配给数据点。

### 3.1 向心参数化公式

向心参数化 (centripetal parameterization) 定义为 (Eq.4-6)：

$$\bar{u}_0 = 0 \tag{Eq.4}$$

$$\bar{u}_k = \bar{u}_{k-1} + \frac{\sqrt{|\mathbf{P}_k - \mathbf{P}_{k-1}|}}{\sum_{i=1}^{N-1} \sqrt{|\mathbf{P}_i - \mathbf{P}_{i-1}|}}, \quad k = 1, \ldots, N-1 \tag{Eq.5}$$

$$\bar{u}_{N-1} = 1 \tag{Eq.6}$$

### 3.2 为什么用向心参数化

| 参数化方法 | 公式 | 优点 | 缺点 |
|-----------|------|------|------|
| 均匀 | $\bar{u}_k = k/(N-1)$ | 简单 | 忽略点间距离 |
| 弦长 | $\propto \|\mathbf{P}_k - \mathbf{P}_{k-1}\|$ | 考虑距离 | 尖点处可能出问题 |
| **向心** | $\propto \sqrt{\|\mathbf{P}_k - \mathbf{P}_{k-1}\|}$ | 避免尖点问题 | 稍复杂 |

向心参数化在曲率变化剧烈处表现更好，是五轴轨迹生成的推荐选择。

## 4. 节点向量构造 (Eq.7)

### 4.1 开放均匀节点向量

对于 $N$ 个数据点和 $p$ 次样条，节点向量长度为 $N + p + 1$。

开放均匀节点向量定义为：

$$u_j = \begin{cases}
0 & j = 0, 1, \ldots, p \\
\frac{1}{N-p} \sum_{i=j-p}^{j-1} \bar{u}_i & j = p+1, \ldots, N-1 \\
1 & j = N, N+1, \ldots, N+p
\end{cases} \tag{Eq.7}$$

### 4.2 节点向量性质

- 首尾各有 $p+1$ 个重复节点（保证曲线穿过端点）
- 内部节点基于参数值平均
- 总长度 = $N + p + 1 = N + 6$（对于五次样条）

## 5. 代码实现

```python
def centripetal_parameterization(points: np.ndarray) -> np.ndarray:
    """向心参数化 (Eq.4-6)"""
    N = len(points)
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)

    # 使用平方根
    sqrt_dists = np.sqrt(dists)
    total = sqrt_dists.sum()

    if total < 1e-12:
        # 退化情况：所有点重合
        return np.linspace(0, 1, N)

    u_bar = np.zeros(N)
    u_bar[1:] = np.cumsum(sqrt_dists) / total
    u_bar[-1] = 1.0

    return u_bar


def compute_knot_vector(u_bar: np.ndarray, degree: int = 5) -> np.ndarray:
    """构造开放均匀节点向量 (Eq.7)"""
    N = len(u_bar)
    n_knots = N + degree + 1
    knots = np.zeros(n_knots)

    # 首尾重复节点
    knots[:degree+1] = 0
    knots[-(degree+1):] = 1

    # 内部节点：参数值平均
    for j in range(degree + 1, N):
        knots[j] = np.mean(u_bar[j-degree:j])

    return knots
```

## 总结

五次 B 样条插值通过向心参数化和开放均匀节点向量，将离散数据点拟合为 C⁴ 连续的曲线。这是实现 C³ 连续弧长参数化的基础。
