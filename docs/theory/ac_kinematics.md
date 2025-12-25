# A-C 逆运动学推导 (Eq.42)

本文档详细推导 A-C 配置五轴机床的逆运动学变换。

## 1. A-C 配置机床几何

### 1.1 坐标系定义

- **工件坐标系 (WCS)**：固定在工件上，刀尖位置 $\mathbf{P}$ 和刀轴姿态 $\mathbf{O}$ 在此坐标系下定义
- **机床坐标系 (MCS)**：固定在机床上，控制器指令 $(X, Y, Z, A, C)$ 在此坐标系下

### 1.2 旋转轴布局

A-C 配置的特点：

- **A 轴**：绕 X 轴旋转，控制刀具倾斜角
- **C 轴**：绕 Z 轴旋转，控制工作台旋转

```
        Z (主轴方向)
        ↑
        │    刀具
        │   ╱
        │  ╱ A轴旋转
        │ ╱
        ├───────→ Y
       ╱
      ╱
     X
    ↙
       ← C轴旋转 (工作台)
```

### 1.3 偏移参数

- $L_{ac,z}$：A-C 轴交点到工作台表面的 Z 方向偏移
- $L_{Tya,z}$：刀具长度补偿（刀尖到主轴参考点的距离）

## 2. 旋转轴角度计算 (Eq.42)

### 2.1 刀轴姿态与旋转轴关系

刀轴姿态 $\mathbf{O} = (O_i, O_j, O_k)^T$ 是单位向量，表示刀具轴线方向。

在 A-C 配置下：

- A 角是刀轴与 Z 轴的夹角
- C 角是刀轴在 XY 平面投影与 Y 轴的夹角

### 2.2 角度计算公式 (Eq.42)

$$A = \arccos(O_k) \tag{Eq.42a}$$

$$C = \text{atan2}(O_i, O_j) \tag{Eq.42b}$$

### 2.3 公式推导

**A 角推导**：

刀轴与 Z 轴的夹角满足：

$$\cos(A) = \mathbf{O} \cdot \mathbf{e}_z = O_k$$

因此 $A = \arccos(O_k)$，范围 $[0, \pi]$。

**C 角推导**：

刀轴在 XY 平面的投影为 $(O_i, O_j)$，与 Y 轴正方向的夹角：

$$C = \text{atan2}(O_i, O_j)$$

注意：这里是 $\text{atan2}(O_i, O_j)$ 而非 $\text{atan2}(O_j, O_i)$，因为 C 轴绕 Z 轴旋转，正方向从 Y 轴到 X 轴。

## 3. 旋转矩阵

### 3.1 A 轴旋转矩阵

绕 X 轴旋转 A 角：

$$R_A = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos A & -\sin A \\
0 & \sin A & \cos A
\end{bmatrix}$$

### 3.2 C 轴旋转矩阵

绕 Z 轴旋转 C 角：

$$R_C = \begin{bmatrix}
\cos C & -\sin C & 0 \\
\sin C & \cos C & 0 \\
0 & 0 & 1
\end{bmatrix}$$

### 3.3 复合旋转

从工件坐标系到机床坐标系的旋转：

$$R = R_A \cdot R_C$$

逆变换（机床到工件）：

$$R^{-1} = R_C^{-1} \cdot R_A^{-1} = R_C^T \cdot R_A^T$$

## 4. 线性轴变换

### 4.1 变换公式

机床线性轴位置由以下公式计算：

$$\mathbf{XYZ} = R^{-1} \cdot (\mathbf{P} + L_{Tya,z} \cdot \mathbf{O}) - \begin{pmatrix} 0 \\ 0 \\ L_{ac,z} \end{pmatrix}$$

### 4.2 公式解释

1. $\mathbf{P} + L_{Tya,z} \cdot \mathbf{O}$：刀尖位置加上刀具长度补偿，得到主轴参考点位置
2. $R^{-1} \cdot (\cdots)$：从工件坐标系变换到旋转后的机床坐标系
3. 减去 $L_{ac,z}$：补偿 A-C 轴交点到工作台的偏移

### 4.3 展开形式

$$X = (\cos C)(P_x + L_{Tya,z} O_i) + (\sin C)(P_y + L_{Tya,z} O_j)$$

$$Y = (-\sin C \cos A)(P_x + L_{Tya,z} O_i) + (\cos C \cos A)(P_y + L_{Tya,z} O_j) + (\sin A)(P_z + L_{Tya,z} O_k)$$

$$Z = (\sin C \sin A)(P_x + L_{Tya,z} O_i) + (-\cos C \sin A)(P_y + L_{Tya,z} O_j) + (\cos A)(P_z + L_{Tya,z} O_k) - L_{ac,z}$$

## 5. 奇异性分析

### 5.1 奇异点条件

当 $A = 0$（刀轴与 Z 轴平行）时：
- $O_i = O_j = 0$, $O_k = 1$
- $C = \text{atan2}(0, 0)$ 未定义

### 5.2 奇异点处理

实际应用中的处理方法：
1. **保持 C 不变**：使用上一时刻的 C 值
2. **小角度近似**：当 $A < \epsilon$ 时，$C$ 设为 0 或保持不变
3. **轨迹规划避开**：在轨迹规划阶段避开奇异配置

```python
def handle_singularity(A, C, A_prev, C_prev, threshold=1e-6):
    """处理奇异点"""
    if A < threshold:
        return A, C_prev  # 保持 C 不变
    return A, C
```

## 6. 代码实现

```python
def inverse_kinematics_ac(P, O, L_ac_z=0, L_Tya_z=0):
    """A-C 配置逆运动学 (Eq.42)"""
    # 旋转轴角度
    A = np.arccos(np.clip(O[2], -1, 1))
    C = np.arctan2(O[0], O[1])

    # 旋转矩阵
    cos_A, sin_A = np.cos(A), np.sin(A)
    cos_C, sin_C = np.cos(C), np.sin(C)

    R_A_inv = np.array([
        [1, 0, 0],
        [0, cos_A, sin_A],
        [0, -sin_A, cos_A]
    ])
    R_C_inv = np.array([
        [cos_C, sin_C, 0],
        [-sin_C, cos_C, 0],
        [0, 0, 1]
    ])
    R_inv = R_C_inv @ R_A_inv

    # 刀具补偿
    P_comp = P + L_Tya_z * O

    # 线性轴位置
    XYZ = R_inv @ P_comp - np.array([0, 0, L_ac_z])

    return XYZ, A, C

def batch_inverse_kinematics_ac(positions, orientations, L_ac_z=0, L_Tya_z=0):
    """批量逆运动学"""
    N = len(positions)
    XYZ = np.zeros((N, 3))
    A = np.zeros(N)
    C = np.zeros(N)

    for i in range(N):
        XYZ[i], A[i], C[i] = inverse_kinematics_ac(
            positions[i], orientations[i], L_ac_z, L_Tya_z
        )

    return XYZ, A, C
```

## 7. 验证

### 7.1 单位向量检验

$$O_i^2 + O_j^2 + O_k^2 = \sin^2 A \cos^2 C + \sin^2 A \sin^2 C + \cos^2 A = 1 \checkmark$$

### 7.2 逆变换验证

从 $(A, C)$ 恢复姿态向量：

$$\mathbf{O} = \begin{pmatrix} \sin A \sin C \\ \sin A \cos C \\ \cos A \end{pmatrix}$$

应与原始姿态一致。

## 总结

A-C 配置逆运动学将工件坐标系下的刀尖位置和刀轴姿态转换为机床的五轴运动指令。旋转轴角度直接由姿态向量的球坐标分量计算，线性轴位置需考虑旋转变换和刀具偏移补偿。需注意 $A = 0$ 处的奇异性问题。
