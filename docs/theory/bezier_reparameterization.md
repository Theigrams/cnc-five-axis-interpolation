# Bézier 重参数化推导 (Eq.32-41)

本文档详细推导 7 阶 Bézier 重参数化的数学原理，用于实现 C³ 连续的弧长参数化姿态样条。

## 1. 问题定义

### 1.1 目标

构建映射 $w(l)$，将弧长 $l$ 映射到姿态样条参数 $w$，满足：

- C³ 连续性
- 单调性（$w$ 随 $l$ 单调递增）
- 端点插值

### 1.2 为什么用 Bézier 而非多项式

与位置样条的 9 阶多项式不同，姿态重参数化使用 7 阶 Bézier 曲线，原因：

- Bézier 曲线天然支持分段表示
- 控制点直观，便于施加单调性约束
- 凸包性质有利于优化

## 2. 7 阶 Bernstein 多项式 (Eq.32-33)

### 2.1 Bernstein 基函数

7 阶 Bernstein 基函数定义为：

$$B_{i,7}(r) = \binom{7}{i} (1-r)^{7-i} r^i, \quad i = 0, 1, \ldots, 7 \tag{Eq.32}$$

其中 $r \in [0, 1]$ 是区间内的归一化参数。

### 2.2 Bézier 曲线形式

每段 $[l_k, l_{k+1}]$ 上的重参数化函数为：

$$w(r) = \sum_{i=0}^{7} B_{i,7}(r) Q_{i,k} \tag{Eq.33}$$

其中 $Q_{i,k}$ 是第 $k$ 段的控制系数。

### 2.3 局部参数

归一化参数 $r$ 与弧长 $l$ 的关系：

$$r = \frac{l - l_k}{l_{k+1} - l_k}$$

## 3. 端点约束 (Eq.37)

### 3.1 值约束

Bézier 曲线端点等于首尾控制点：

$$w(0) = Q_{0,k} = w_k \tag{Eq.37a}$$

$$w(1) = Q_{7,k} = w_{k+1} \tag{Eq.37b}$$

### 3.2 自由变量

每段有 8 个控制点，端点固定后剩余 6 个自由变量：$Q_1, Q_2, \ldots, Q_6$。

## 4. 单调性约束 (Eq.36)

### 4.1 Bézier 曲线单调性条件

若控制点单调递增，则 Bézier 曲线单调：

$$Q_{0,k} \leq Q_{1,k} \leq Q_{2,k} \leq \cdots \leq Q_{7,k} \tag{Eq.36}$$

### 4.2 约束形式

对于优化问题，单调性约束表示为不等式约束：

$$Q_{i+1,k} - Q_{i,k} \geq 0, \quad i = 0, 1, \ldots, 6$$

## 5. C³ 连续性约束 (Eq.38)

### 5.1 导数与控制点差分的关系

Bézier 曲线在端点的导数：

$$w'(0) = 7 \cdot \frac{Q_1 - Q_0}{\Delta l}$$

$$w''(0) = 42 \cdot \frac{Q_2 - 2Q_1 + Q_0}{\Delta l^2}$$

$$w'''(0) = 210 \cdot \frac{Q_3 - 3Q_2 + 3Q_1 - Q_0}{\Delta l^3}$$

右端点类似。

### 5.2 段间连续性 (Eq.38)

第 $k$ 段末尾与第 $k+1$ 段开头的导数必须相等：

**C¹ 连续**：
$$\frac{Q_{7,k} - Q_{6,k}}{\Delta l_k} = \frac{Q_{1,k+1} - Q_{0,k+1}}{\Delta l_{k+1}} \tag{Eq.38a}$$

**C² 连续**：
$$\frac{Q_{7,k} - 2Q_{6,k} + Q_{5,k}}{\Delta l_k^2} = \frac{Q_{2,k+1} - 2Q_{1,k+1} + Q_{0,k+1}}{\Delta l_{k+1}^2} \tag{Eq.38b}$$

**C³ 连续**：
$$\frac{Q_{7,k} - 3Q_{6,k} + 3Q_{5,k} - Q_{4,k}}{\Delta l_k^3} = \frac{Q_{3,k+1} - 3Q_{2,k+1} + 3Q_{1,k+1} - Q_{0,k+1}}{\Delta l_{k+1}^3} \tag{Eq.38c}$$

## 6. Jerk 最小化目标 (Eq.39)

### 6.1 目标函数

最小化 jerk 积分：

$$\min \int_0^L |w'''(l)|^2 dl \tag{Eq.39}$$

### 6.2 离散近似

使用控制点三阶差分作为 jerk 的代理：

$$J \approx \sum_k \sum_{i=0}^{4} \left( \frac{Q_{i+3,k} - 3Q_{i+2,k} + 3Q_{i+1,k} - Q_{i,k}}{\Delta l_k^3} \right)^2$$

## 7. 优化求解 (Eq.40-41)

### 7.1 初始值设定 (Eq.41)

为保证单调性，初始化使用均匀分布：

$$\delta = \min_k \frac{w_{k+1} - w_k}{7 \cdot \Delta l_k}$$

$$Q_{i,k} = w_k + i \cdot \delta \cdot \Delta l_k, \quad i = 1, \ldots, 6 \tag{Eq.41}$$

### 7.2 优化问题 (Eq.40)

$$
\begin{aligned}
\min_{\{Q_{i,k}\}} &\quad J(\{Q_{i,k}\})\\
\text{s.t.} &\quad \text{单调性约束 (Eq.36)}\\
&\quad \text{C³ 连续性约束 (Eq.38)}
\end{aligned}
$$

### 7.3 SLSQP 求解

使用 Sequential Least Squares Programming 求解：

```python
from scipy.optimize import minimize

result = minimize(
    objective,           # Jerk 积分
    x0,                  # 初始控制点
    method='SLSQP',
    constraints=[
        {'type': 'ineq', 'fun': monotonicity_constraint},
        {'type': 'eq', 'fun': c1_constraint},
        {'type': 'eq', 'fun': c2_constraint},
        {'type': 'eq', 'fun': c3_constraint},
    ],
    options={'maxiter': 100}
)
```

## 8. 代码实现

```python
class BezierReparameterization:
    def __init__(self, l_values, w_values):
        self.l_values = l_values
        self.w_values = w_values
        self.Q = []  # 每段的控制系数

        self._initialize_coefficients()
        self._optimize()

    def _initialize_coefficients(self):
        """初始化控制系数 (Eq.41)"""
        l, w = self.l_values, self.w_values

        # 计算全局 δ
        deltas = [(w[k+1] - w[k]) / (7 * (l[k+1] - l[k]))
                  for k in range(len(l) - 1)
                  if l[k+1] - l[k] > 1e-12]
        delta = min(deltas) if deltas else 1e-6

        # 初始化每段
        for k in range(len(l) - 1):
            dl = l[k+1] - l[k]
            Q = np.zeros(8)
            Q[0] = w[k]
            Q[7] = w[k+1]
            for i in range(1, 7):
                Q[i] = w[k] + i * delta * dl
            self.Q.append(Q)

    def __call__(self, l):
        """评估 w(l)"""
        l = np.clip(l, self.l_values[0], self.l_values[-1])

        for k in range(len(self.Q)):
            if self.l_values[k] <= l <= self.l_values[k+1]:
                dl = self.l_values[k+1] - self.l_values[k]
                r = (l - self.l_values[k]) / dl

                # Bernstein 多项式
                Q = self.Q[k]
                w = sum(comb(7, i) * (1-r)**(7-i) * r**i * Q[i]
                        for i in range(8))
                return w

        return self.w_values[-1]
```

## 9. 算法特点

| 特性 | 说明 |
|------|------|
| 连续性 | C³ 连续，jerk 平滑 |
| 单调性 | 控制点约束保证 |
| 灵活性 | 分段表示，自适应精度 |
| 鲁棒性 | 凸优化，收敛稳定 |

## 总结

7 阶 Bézier 重参数化通过控制点优化，在满足 C³ 连续性和单调性约束的前提下最小化 jerk。与位置样条的 9 阶多项式不同，Bézier 形式更适合分段优化和约束处理。这是实现姿态样条弧长参数化的关键技术。
