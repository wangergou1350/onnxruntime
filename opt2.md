
#### **6. EliminateNop (无操作消除)**

**数学定义**:
无操作（No-operation）节点定义为不执行任何实际计算的节点：
$$\text{Nop}(x) = x \quad \text{(无副作用的恒等变换)}$$

**理论原理**:
某些算子在特定条件下退化为恒等操作，可以安全移除：
- **Pad操作**: 所有padding值为0时
- **Slice操作**: 切片范围覆盖整个张量时
- **Reshape操作**: 新形状与原形状相同时

**检测条件**:
```python
def is_nop_operation(node):
    if node.op_type == "Pad":
        # 检查所有padding值是否为0
        pads = get_attribute(node, "pads")
        return all(pad == 0 for pad in pads)

    elif node.op_type == "Slice":
        # 检查是否为完整切片
        starts, ends, axes = get_slice_params(node)
        input_shape = get_input_shape(node)
        return is_full_slice(starts, ends, axes, input_shape)

    elif node.op_type == "Reshape":
        # 检查形状是否实际改变
        input_shape = get_input_shape(node)
        output_shape = get_output_shape(node)
        return shapes_equivalent(input_shape, output_shape)

    return False
```

**优化示例**:
```
优化前: Input[1,3,224,224] → Pad(pads=[0,0,0,0,0,0,0,0]) → Output[1,3,224,224]
优化后: Input[1,3,224,224] ──────────────────────────────→ Output[1,3,224,224]
```

---

#### **7. EliminateMaxPool (MaxPool消除)**

**数学定义**:
MaxPool操作定义为：
$$\text{MaxPool}(X, \text{kernel\_size}, \text{stride}) = \max_{k \in \text{kernel}} X[\text{window}_k]$$

**消除条件**: 当kernel_size = 1且stride = 1时，MaxPool退化为恒等操作：
$$\text{MaxPool}(X, 1, 1) = \max(X[i]) = X[i] \quad \text{(单元素最大值)}$$

**理论原理**:
1x1池化核只有一个元素，最大值就是该元素本身
$$\max\{x\} = x$$

**优化前后对比**:
```
优化前: Input[B,C,H,W] → MaxPool(kernel=1, stride=1) → Output[B,C,H,W]
优化后: Input[B,C,H,W] ─────────────────────────────→ Output[B,C,H,W]
```

**内存和计算节省**:
- **计算复杂度**: 从O(N)降为O(1)
- **内存访问**: 消除池化操作的内存开销

---

#### **8. EliminateConcat (Concat消除)**

**数学定义**:
Concat操作定义为张量拼接：
$$\text{Concat}([X_1, X_2, \ldots, X_n], \text{axis}) = [X_1; X_2; \ldots; X_n]_{\text{axis}}$$

**消除条件**: 当只有单个输入时，Concat退化为恒等操作：
$$\text{Concat}([X], \text{axis}) = X$$

**理论原理**:
单输入拼接的数学性质：
$$[X]_{\text{axis}} = X \quad \text{(单元素拼接)}$$

**常见场景**:
```python
# 动态图优化后可能产生单输入Concat
def dynamic_concat(inputs):
    if condition:
        return concat([tensor1, tensor2], axis=0)
    else:
        return concat([tensor1], axis=0)  # ← 可优化为直接返回tensor1
```

**优化示例**:
```
优化前: Input[2,3,4] → Concat(inputs=[Input], axis=0) → Output[2,3,4]
优化后: Input[2,3,4] ─────────────────────────────────→ Output[2,3,4]
```

---

#### **9. DivMulFusion (除法乘法融合)**

**数学定义**: 除法和乘法的等价变换：
$$x \div a = x \times a^{-1}, \quad \text{其中} \quad a \neq 0$$

**理论原理**:
1. **计算复杂度**: 除法运算比乘法运算更昂贵（约2-10倍延迟）
2. **数值稳定性**: 预计算 $a^{-1}$ 可以避免重复除法计算
3. **硬件友好**: 现代处理器对乘法有更好的向量化支持

**优化变换**:
```
优化前: y = x ÷ 2.0
       ┌───┐    ┌─────┐    ┌───┐
       │ x │───→│ Div │───→│ y │
       └───┘    │ 2.0 │    └───┘
                └─────┘

优化后: y = x × 0.5
       ┌───┐    ┌─────┐    ┌───┐
       │ x │───→│ Mul │───→│ y │
       └───┘    │ 0.5 │    └───┘
                └─────┘
```

**性能对比**:
```
操作类型    │ 延迟 (cycles) │ 吞吐量 (ops/cycle) │
───────────┼──────────────┼──────────────────┤
除法 (FP32) │     14       │       0.5        │
乘法 (FP32) │      4       │       2.0        │
提升比率    │    3.5x      │       4.0x       │
```

**应用条件**:
```python
def can_apply_div_to_mul(divisor):
    # 检查除数是否为非零常量
    if not is_constant(divisor) or divisor == 0:
        return False

    # 检查倒数是否可精确表示
    reciprocal = 1.0 / divisor
    if not is_representable_exactly(reciprocal):
        return False

    return True
```

**数值精度考虑**:
- 当 $a$ 是2的幂次时，$a^{-1}$ 可以精确表示
- 对于其他值，需要评估精度损失是否可接受

---

#### **10. FuseReluClip (ReLU-Clip融合)**

**数学定义**:
- **ReLU**: $\text{ReLU}(x) = \max(0, x)$
- **Clip**: $\text{Clip}(x, \text{min}, \text{max}) = \max(\text{min}, \min(\text{max}, x))$
- **ReLU6**: $\text{ReLU6}(x) = \min(6, \max(0, x)) = \text{Clip}(\text{ReLU}(x), 0, 6)$

**理论原理**: ReLU6是移动设备上广泛使用的激活函数，结合了ReLU的稀疏性和Clip的有界性：
$$\text{ReLU6}(x) = \min(6, \max(0, x)) = \begin{cases}
0 & \text{if } x \leq 0 \\
x & \text{if } 0 < x < 6 \\
6 & \text{if } x \geq 6
\end{cases}$$

**激活函数对比图**:
```
     f(x)
      ↑
    6 ┤     ┌─────────  ReLU6
      │    ╱
    4 ┤   ╱
      │  ╱
    2 ┤ ╱               ReLU
      │╱                 ↗
    0 ┼──────────────────→ x
     -2 0  2  4  6  8

ReLU:   f(x) = max(0, x)        无上界
ReLU6:  f(x) = min(6,max(0,x))  有界[0,6]
```

**融合优势**:
1. **内存减少**: 消除中间张量
2. **计算合并**: 单次kernel调用
3. **数值稳定**: 避免大值传播

**优化前后对比**:
```
优化前 (两次kernel调用):
Input → ReLU kernel → Intermediate → Clip kernel → Output
  x   →  max(0,x)   →      y      → min(6,max(0,y)) → z

优化后 (单次kernel调用):
Input ──────── ReLU6 kernel ──────→ Output
  x   ──────── min(6,max(0,x)) ────→   z
```

**CUDA kernel示例**:
```cpp
// 优化前 - 两个分离的kernel
__global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void clip_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fminf(6.0f, input[idx]);
}

// 优化后 - 融合kernel
__global__ void relu6_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fminf(6.0f, fmaxf(0.0f, input[idx]));
}
```

**检测模式**:
```python
def detect_relu_clip_pattern(graph):
    for node in graph.nodes:
        if (node.op_type == 'Relu' and
            has_single_consumer(node) and
            consumer.op_type == 'Clip' and
            consumer.min_val == 0.0 and
            consumer.max_val == 6.0):
            return True
    return False
```

---

#### **11. GemmSumFusion (GEMM求和融合)**

**数学定义**:
将GEMM操作 $Y = \alpha \cdot A \cdot B + \beta \cdot C$ 与后续的求和操作融合：
$$\text{Sum}(\text{GEMM}(A, B, C)) = \sum_{i,j} (\alpha \cdot A \cdot B + \beta \cdot C)_{i,j}$$

**理论原理**: 利用求和操作的线性性质：
$$\sum_{i,j} (\alpha \cdot (A \cdot B)_{i,j} + \beta \cdot C_{i,j}) = \alpha \sum_{i,j} (A \cdot B)_{i,j} + \beta \sum_{i,j} C_{i,j}$$

**优化前计算图**:
```
Input_A[m,k]  Input_B[k,n]  Bias[m,n]
    \             |             /
     \            |            /
      ──→ GEMM(A,B,C) ──→ [m,n] ──→ Sum ──→ [1] (标量)
```

**优化后计算图**:
```
Input_A[m,k]  Input_B[k,n]  Bias[m,n]
    \             |             /
     \            |            /
      ──→ GemmSum(A,B,C) ──────────→ [1] (标量)
```

**实现优化**:
- **内存访问**: 避免存储中间的完整矩阵结果
- **缓存效率**: 在GEMM计算过程中直接累加求和
- **数值稳定性**: 使用Kahan求和算法提高精度

**伪代码实现**:
```cpp
float gemm_sum_fused(Matrix A, Matrix B, Matrix C, float alpha, float beta) {
    float total_sum = 0.0;
    float compensation = 0.0;  // Kahan求和补偿

    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            float dot_product = 0.0;
            for (int k = 0; k < A.cols; k++) {
                dot_product += A[i][k] * B[k][j];
            }
            float element = alpha * dot_product + beta * C[i][j];

            // Kahan求和算法
            float y = element - compensation;
            float t = total_sum + y;
            compensation = (t - total_sum) - y;
            total_sum = t;
        }
    }
    return total_sum;
}
```

**性能优势**:
- **内存节省**: $O(m \times n)$ → $O(1)$ 输出存储
- **计算效率**: 减少一次完整矩阵遍历
- **带宽优化**: 避免写回中间结果到内存

---

#### **12. GemmTransposeFusion (GEMM转置融合)**

**数学定义**:
将输入转置操作与GEMM融合，利用矩阵乘法的转置性质：
$$(A^T \cdot B)^T = B^T \cdot A$$
$$A \cdot B^T = (B \cdot A^T)^T$$

**理论原理**:
对于计算 $Y = \text{Transpose}(A) \cdot B$，可以重写为：
$$Y = A^T \cdot B = (B^T \cdot A)^T$$

**转置融合策略**:

1. **左矩阵转置**: $C = A^T \cdot B$
   ```
   优化前: A[m,k] → Transpose → A^T[k,m] → GEMM(A^T,B) → C[k,n]
   优化后: A[m,k], B[k,n] → GemmT(A,B,transA=true) → C[k,n]
   ```

2. **右矩阵转置**: $C = A \cdot B^T$
   ```
   优化前: B[n,k] → Transpose → B^T[k,n] → GEMM(A,B^T) → C[m,n]
   优化后: A[m,k], B[n,k] → GemmT(A,B,transB=true) → C[m,n]
   ```

**内存访问模式优化**:

**连续内存访问**:
```
标准GEMM: A[i,k] * B[k,j] (B按列访问，缓存不友好)
转置融合: A[i,k] * B^T[k,j] = A[i,k] * B[j,k] (B按行访问，缓存友好)
```

**图示对比**:
```
优化前 - 显式转置:
┌──────────┐    ┌───────────┐    ┌─────────┐    ┌─────────┐
│ A[m,k]   │───→│Transpose  │───→│ A^T[k,m]│───→│ GEMM    │───→ C[k,n]
└──────────┘    │Operation  │    └─────────┘    │A^T * B  │
                └───────────┘                   └─────────┘
┌──────────┐                                          ↑
│ B[k,n]   │─────────────────────────────────────────┘
└──────────┘

优化后 - 融合转置:
┌──────────┐    ┌─────────────────────┐
│ A[m,k]   │───→│ GEMM with Transpose │───→ C[k,n]
└──────────┘    │ (transA = true)     │
┌──────────┐    │                     │
│ B[k,n]   │───→│                     │
└──────────┘    └─────────────────────┘
```

**BLAS调用优化**:
```cpp
// 优化前 - 两次BLAS调用
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
           k, n, m, 1.0, A, k, B, n, 0.0, temp, n);  // Transpose
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           k, n, m, 1.0, temp, n, B, n, 0.0, C, n);   // GEMM

// 优化后 - 一次BLAS调用
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
           k, n, m, 1.0, A, k, B, n, 0.0, C, n);      // 融合调用
```

**性能优势**:
- **内存减少**: 消除转置中间结果 $O(m \times k)$ 存储
- **缓存优化**: 改善内存访问模式，提高缓存命中率
- **FLOPS效率**: 减少转置操作的 $O(m \times k)$ 数据移动

---

#### **13. NotWhereFusion (Not-Where融合)**

**数学定义**:
将逻辑非操作与条件选择操作融合：
$$\text{Where}(\neg \text{condition}, x, y) = \text{Where}(\text{condition}, y, x)$$

**理论原理**:
利用条件选择的对偶性质，消除显式的逻辑非操作：
$$\text{Not}(c) \circ \text{Where}(c, a, b) = \text{Where}(c, b, a)$$

**逻辑等价变换**:
```
原始表达式: Where(Not(condition), true_value, false_value)
等价表达式: Where(condition, false_value, true_value)
```

**真值表验证**:
| condition | Not(condition) | Where(Not(condition), x, y) | Where(condition, y, x) |
|-----------|----------------|------------------------------|------------------------|
| True      | False          | y                            | y                      |
| False     | True           | x                            | x                      |

**计算图变换**:
```
优化前:
┌───────────┐    ┌─────────┐    ┌─────────────────────┐
│ condition │───→│   Not   │───→│ Where(Not(cond),   │───→ Output
└───────────┘    └─────────┘    │       x, y)        │
┌───────────┐                   │                    │
│     x     │──────────────────→│                    │
└───────────┘                   │                    │
┌───────────┐                   │                    │
│     y     │──────────────────→│                    │
└───────────┘                   └─────────────────────┘

优化后:
┌───────────┐    ┌─────────────────────┐
│ condition │───→│ Where(cond, y, x)   │───→ Output
└───────────┘    │                     │
┌───────────┐    │                     │
│     x     │───→│                     │
└───────────┘    │                     │
┌───────────┐    │                     │
│     y     │───→│                     │
└───────────┘    └─────────────────────┘
```

**实现细节**:
```python
def detect_not_where_pattern(graph):
    for node in graph.nodes:
        if (node.op_type == "Where" and
            node.inputs[0].producer.op_type == "Not"):
            # 检测到 Where(Not(condition), x, y) 模式
            condition = node.inputs[0].producer.inputs[0]
            x = node.inputs[1]
            y = node.inputs[2]

            # 替换为 Where(condition, y, x)
            new_where = create_where_node(condition, y, x)
            replace_node(node, new_where)

            # 如果Not节点没有其他用途，删除它
            if not_node.outputs[0].consumers_count == 0:
                remove_node(not_node)
```

**性能优势**:
- **计算减少**: 消除逻辑非操作 (约减少10-20%计算量)
- **内存优化**: 减少一个中间张量存储
- **指令优化**: 现代CPU的条件选择指令直接支持参数交换

**适用场景**:
- 条件激活函数实现
- 掩码操作优化
- 分支预测优化的条件计算

---

#### **14. ConvAddFusion (Conv+Add融合)**

**原理**: 将卷积和加法融合，通常用于bias添加。

**优化前**:
```
Input ──Conv(weights)─→ temp ──Add(bias)─→ Output
```

**优化后**:
```
Input ──Conv(weights, bias)─→ Output
```

**实现**: 将Add的bias参数合并到Conv的bias参数中。

---

#### **15. ConvMulFusion (Conv+Mul融合)**

**原理**: 将卷积与逐元素乘法融合。

---

### **B. GraphTransformer复杂优化器 (20个)**

#### **1. ConstantFolding (常量折叠)**

**数学定义**: 对于操作 $f(c_1, c_2, \ldots, c_n)$，其中 $c_i$ 都是编译时常量，常量折叠计算：
$$\text{ConstantFold}(f(c_1, \ldots, c_n)) = f(c_1, \ldots, c_n) = c_{\text{result}}$$

**理论原理**: 在编译时预计算所有操作数都是常量的表达式，避免运行时重复计算。

**优化示例**:
```
优化前计算图:
Const(2.0) ──┐
             ├─→ Add ──→ Output
Const(3.0) ──┘

优化后计算图:
Const(5.0) ────────────→ Output
```

**复杂示例**:
```
优化前:
Const(2)   Const(3)
   \         /
    \   Mul /      Const(1)
     \  |  /         /
      \ | /         /
       Add ────────/
        |
     Output

步骤分解:
1. Mul(2,3) = 6     ← 编译时计算
2. Add(6,1) = 7     ← 编译时计算

优化后:
Const(7) ──→ Output
```

**算法复杂度**:
- **时间复杂度**: $O(n \cdot d)$，其中 $n$ 是节点数，$d$ 是常量表达式的平均深度
- **空间复杂度**: $O(n)$ 用于存储常量传播信息

**伪代码实现**:
```python
def constant_folding(graph):
    changed = True
    while changed:
        changed = False
        for node in topological_order(graph):
            if all_inputs_are_constants(node):
                # 执行编译时计算
                result = evaluate_at_compile_time(node)
                replace_node_with_constant(node, result)
                changed = True
    return graph
```

**应用条件**:
- 所有输入操作数必须是编译时已知的常量
- 操作必须是确定性的（无随机性、无副作用）
- 计算结果必须在目标数据类型范围内

---

#### **2. CommonSubexpressionElimination (公共子表达式消除)**

**数学定义**: 对于两个表达式 $E_1$ 和 $E_2$，如果它们在语义上等价且计算相同的值：
$$E_1 \equiv E_2 \Rightarrow \text{CSE}(E_1, E_2) = \{E_1\} \text{ or } \{E_2\}$$

**理论原理**: 基于值编号（Value Numbering）算法，为每个子表达式分配唯一标识符，相同标识符的表达式可以合并。

**支配关系**: 表达式 $E_1$ 支配 $E_2$ 当且仅当从入口到 $E_2$ 的所有路径都经过 $E_1$：
$$E_1 \text{ dominates } E_2 \Leftrightarrow \forall \text{path}(entry \to E_2): E_1 \in \text{path}$$

**算法步骤**:
1. **构建表达式哈希表**: $H(op, input_1, input_2, \ldots) \rightarrow \text{value\_number}$
2. **支配性分析**: 确保替换的安全性
3. **表达式替换**: 将后续相同表达式替换为首次计算结果

**优化前**:
```
计算图:
     Input(x)
    /        \
   |          |
Square(x)  Square(x)  ← 重复计算 x²
   |          |
Result_1   Result_2

哈希表:
Square(x) → value_1
Square(x) → value_1  ← 相同哈希值
```

**优化后**:
```
计算图:
     Input(x)
        |
    Square(x)    ← 只计算一次
    /       \
Result_1  Result_2
```

**复杂示例 - 子图CSE**:
```
优化前:
Input_A  Input_B      Input_A  Input_B
   \       /             \       /
    \     /               \     /
     Add_1                 Add_2  ← 相同的子图
      |                     |
    ReLU_1                ReLU_2
      |                     |
   Output_1              Output_2

优化后:
Input_A  Input_B
   \       /
    \     /
     Add      ← 合并相同计算
      |
     ReLU
    /    \
Output_1  Output_2
```

**Value Numbering算法**:
```python
class ValueNumbering:
    def __init__(self):
        self.value_map = {}  # (op, inputs) → value_number
        self.expression_map = {}  # value_number → first_node

    def get_value_number(self, node):
        # 创建表达式签名
        signature = (node.op_type, tuple(sorted(node.inputs)))

        if signature in self.value_map:
            return self.value_map[signature]
        else:
            new_value = len(self.value_map)
            self.value_map[signature] = new_value
            self.expression_map[new_value] = node
            return new_value
```

**CSE收益分析**:
- **计算减少**: $C_{\text{after}} = C_{\text{before}} - \sum_{i} (n_i - 1) \cdot cost_i$
  - 其中 $n_i$ 是第 $i$ 个重复表达式的出现次数
- **内存减少**: 消除重复的中间张量存储

---

#### **3. DoubleQDQPairsRemover (双重量化对消除)**

**数学定义**: 量化-反量化操作对：
$$\text{QDQ}(x) = \text{Dequantize}(\text{Quantize}(x, s, z), s, z)$$

其中量化函数：
$$\text{Quantize}(x, s, z) = \text{round}\left(\frac{x}{s}\right) + z$$

反量化函数：
$$\text{Dequantize}(q, s, z) = s \cdot (q - z)$$

**理论原理**: 对于相同的量化参数，连续的QDQ操作在数学上近似恒等变换：
$$\text{Dequantize}(\text{Quantize}(x, s, z), s, z) \approx x + \epsilon$$

其中 $\epsilon$ 是量化误差，满足 $|\epsilon| \leq \frac{s}{2}$。

**量化误差分析**:
```
原始值:     x = 3.7
量化参数:   scale = 0.1, zero_point = 128

量化过程:
1. x_normalized = 3.7 / 0.1 = 37
2. x_quantized = round(37) + 128 = 165

反量化过程:
1. x_dequantized = 0.1 * (165 - 128) = 3.7

误差: |3.7 - 3.7| = 0  ← 此例中误差为0
```

**连续QDQ问题**:
```
原始: x = 3.75
第一次QDQ: x₁ = QDQ(3.75) = 3.7  (误差: -0.05)
第二次QDQ: x₂ = QDQ(3.7)  = 3.7  (累积误差: -0.05)
```

**优化策略**:
```
模式1 - 连续相同参数QDQ:
Input → Q(s₁,z₁) → DQ(s₁,z₁) → Q(s₁,z₁) → DQ(s₁,z₁) → Output
简化为:
Input → Q(s₁,z₁) → DQ(s₁,z₁) → Output

模式2 - 往返QDQ:
Input(fp32) → Q(s,z) → DQ(s,z) → Output(fp32)
简化为:
Input(fp32) ──────────────────→ Output(fp32)
```

**检测算法**:
```python
def detect_redundant_qdq(graph):
    redundant_patterns = []

    for node in graph.nodes:
        if node.op_type == 'DequantizeLinear':
            # 查找前驱量化节点
            quant_node = find_predecessor(node, 'QuantizeLinear')
            if quant_node and same_scale_zero_point(quant_node, node):
                # 检查是否有中间消费者
                if not has_intermediate_consumers(quant_node, node):
                    redundant_patterns.append((quant_node, node))

    return redundant_patterns
```

---

#### **4. ConstantSharing (常量共享)**

**数学定义**: 对于常量集合 $C = \{c_1, c_2, \ldots, c_n\}$，其中 $c_i = c_j$ 对某些 $i \neq j$：
$$\text{ConstantSharing}(C) = \{c_{\text{unique}} : c_{\text{unique}} \in \text{unique}(C)\}$$

**理论原理**: 基于常量值的相等性检测，将多个相同的常量节点合并为单个共享节点。

**内存节省计算**:
对于 $k$ 个相同的常量节点，每个大小为 $s$ 字节：
- **优化前内存**: $M_{\text{before}} = k \times s$
- **优化后内存**: $M_{\text{after}} = s$
- **节省比例**: $\eta = \frac{k-1}{k} \times 100\%$

**哈希检测算法**:
```python
import hashlib

def compute_constant_hash(constant_node):
    """计算常量节点的哈希值"""
    # 连接数据类型、形状和值
    data = constant_node.dtype.encode()
    data += str(constant_node.shape).encode()
    data += constant_node.value.tobytes()

    return hashlib.sha256(data).hexdigest()

def constant_sharing_optimization(graph):
    constant_map = {}  # hash → first_constant_node

    for node in graph.nodes:
        if node.op_type == 'Constant':
            hash_value = compute_constant_hash(node)

            if hash_value in constant_map:
                # 找到重复常量，替换所有使用
                original_node = constant_map[hash_value]
                redirect_consumers(node, original_node)
                remove_node(graph, node)
            else:
                constant_map[hash_value] = node
```

**图示优化过程**:
```
优化前:
Const_A(1.0) ────→ Conv_1
Const_B(1.0) ────→ Conv_2
Const_C(1.0) ────→ Conv_3
Const_D(2.0) ────→ Conv_4

步骤1: 哈希计算
hash(Const_A) = "abc123"
hash(Const_B) = "abc123"  ← 与A相同
hash(Const_C) = "abc123"  ← 与A相同
hash(Const_D) = "def456"

步骤2: 重定向消费者
    Const_A(1.0)
   /     |     \
Conv_1 Conv_2 Conv_3
Const_D(2.0) ────→ Conv_4

优化后:
    Const_A(1.0)
   /     |     \
Conv_1 Conv_2 Conv_3
Const_D(2.0) ────→ Conv_4

内存节省: 3个常量节点 → 2个常量节点 (节省33%)
```

**高级优化 - 近似常量合并**:
对于浮点常量，可以考虑在允许误差范围内的近似相等：
```python
def approximately_equal(a, b, tolerance=1e-6):
    return abs(a - b) < tolerance

# 扩展哈希函数支持容差
def fuzzy_constant_hash(value, tolerance=1e-6):
    # 将值舍入到指定精度
    rounded_value = round(value / tolerance) * tolerance
    return hash(rounded_value)
```

**性能收益**:
1. **内存使用**: 线性减少重复常量数量
2. **缓存效率**: 提高指令和数据缓存命中率
3. **加载时间**: 减少模型文件大小和加载时间

---

#### **4. ConstantFolding (常量折叠)**

**原理**: 在编译时计算所有仅依赖常量的表达式，减少运行时计算。

**优化前**:
```
  Const(2.0)    Const(3.0)
      |             |
      └─────Add─────┘
               |
           result(?)
```

**优化后**:
```
      Const(5.0)
           |
      result(5.0)
```

**代码示例**:
```python
# 优化前的ONNX图
Add(Constant(2.0), Constant(3.0)) → output

# 优化后
Constant(5.0) → output
```

**收益**: 减少运行时计算，降低延迟和内存使用。

---

#### **5. MatMulAddFusion (MatMul+Add融合)**

**数学定义**:
将矩阵乘法与矩阵加法融合为通用矩阵乘法(GEMM)操作：
$$\text{MatMul}(A, B) + C = \text{GEMM}(A, B, C, \alpha=1, \beta=1)$$

其中GEMM的完整定义为：
$$\text{GEMM}(A, B, C, \alpha, \beta) = \alpha \cdot A \cdot B + \beta \cdot C$$

**理论原理**:
利用线性代数的分配律，将两个独立的操作合并为一个优化的BLAS调用：
$$\underbrace{A \cdot B}_{\text{MatMul}} + \underbrace{C}_{\text{bias}} = \underbrace{A \cdot B + C}_{\text{GEMM}}$$

**维度兼容性分析**:
设 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，则：
- $\text{MatMul}(A, B) \in \mathbb{R}^{m \times n}$
- $C$ 必须是广播兼容的形状，如：
  - $C \in \mathbb{R}^{m \times n}$ (矩阵加法)
  - $C \in \mathbb{R}^{1 \times n}$ (行向量广播)
  - $C \in \mathbb{R}^{m \times 1}$ (列向量广播)
  - $C \in \mathbb{R}^{1 \times 1}$ (标量广播)

**计算图变换**:
```
优化前:
┌─────────┐    ┌──────────────────┐    ┌─────────────────┐
│ A[m,k]  │───→│                  │───→│ temp_result     │
└─────────┘    │ MatMul(A, B)     │    │ [m,n]          │───┐
┌─────────┐    │                  │    └─────────────────┘   │
│ B[k,n]  │───→│                  │                          │    ┌─────────────────┐
└─────────┘    └──────────────────┘                          │───→│ Add(temp, C)    │───→ Output[m,n]
┌─────────┐                                                  │    └─────────────────┘
│ C[m,n]  │─────────────────────────────────────────────────┘
└─────────┘

优化后:
┌─────────┐    ┌─────────────────────────────────┐
│ A[m,k]  │───→│                                 │
└─────────┘    │ GEMM(A, B, C)                  │───→ Output[m,n]
┌─────────┐    │ = A×B + C                      │
│ B[k,n]  │───→│                                 │
└─────────┘    │                                 │
┌─────────┐    │                                 │
│ C[m,n]  │───→│                                 │
└─────────┘    └─────────────────────────────────┘
```

**BLAS优化实现**:
```cpp
// 优化前 - 两次BLAS调用
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           m, n, k, 1.0, A, k, B, n, 0.0, temp, n);    // A×B
cblas_saxpy(m*n, 1.0, C, 1, temp, 1);                  // temp + C

// 优化后 - 一次BLAS调用
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           m, n, k, 1.0, A, k, B, n, 1.0, C, n);       // A×B + C
```

**性能分析**:

**理论FLOPS计算**:
- **MatMul**: $2mnk$ FLOPs (每个元素需要k次乘法和k-1次加法)
- **Add**: $mn$ FLOPs (逐元素加法)
- **总计**: $2mnk + mn$ FLOPs

**内存访问模式**:
```
优化前内存访问:
1. 读取 A: mk 元素
2. 读取 B: kn 元素
3. 写入 temp: mn 元素 (中间结果)
4. 读取 temp: mn 元素
5. 读取 C: mn 元素
6. 写入 output: mn 元素
总访问量: mk + kn + 4mn

优化后内存访问:
1. 读取 A: mk 元素
2. 读取 B: kn 元素
3. 读取/写入 C: 2mn 元素 (in-place操作)
总访问量: mk + kn + 2mn
```

**缓存效率提升**:
- **消除中间存储**: 减少$mn$的临时矩阵存储
- **内存带宽**: 减少约$2mn$的内存传输
- **缓存复用**: C矩阵在GEMM内核中重复使用

**应用场景**:
- **全连接层**: $y = Wx + b$ (权重矩阵乘法 + bias加法)
- **Attention机制**: $\text{Attention} = QK^T + \text{mask}$
- **线性变换**: 神经网络中的仿射变换

**数值稳定性考虑**:
使用FMA(Fused Multiply-Add)指令时，保持更高的中间精度：
$$\text{FMA}(a, b, c) = a \times b + c \quad \text{(单次舍入)}$$

相比分离操作的两次舍入，FMA提供更好的数值精度。

---

#### **6. ReshapeFusion (Reshape融合)**

**数学定义**:
Reshape操作定义为张量维度重排函数：
$$\text{Reshape}(X, \text{shape}) : \mathbb{R}^{d_1 \times d_2 \times \ldots \times d_n} \rightarrow \mathbb{R}^{s_1 \times s_2 \times \ldots \times s_m}$$

其中必须满足元素总数守恒：
$$\prod_{i=1}^{n} d_i = \prod_{j=1}^{m} s_j$$

**理论原理**:
连续的Reshape操作具有复合性质，可以通过函数复合简化：
$$\text{Reshape}(\text{Reshape}(X, \text{shape}_1), \text{shape}_2) = \text{Reshape}(X, \text{shape}_2)$$

前提是中间形状 $\text{shape}_1$ 在数据布局上与最终形状 $\text{shape}_2$ 兼容。

**维度变换示例**:
```
原始张量: X ∈ ℝ^(1×3×224×224)  [批量×通道×高×宽]
        ┌─────────────────────────┐
        │ [1, 3, 224, 224]        │  总元素: 1×3×224×224 = 150,528
        └─────────────────────────┘

第一次Reshape: → ℝ^(1×3×50176)
        ┌─────────────────────────┐
        │ [1, 3, 50176]           │  总元素: 1×3×50176 = 150,528 ✓
        └─────────────────────────┘

第二次Reshape: → ℝ^(1×50176)
        ┌─────────────────────────┐
        │ [1, 150528]             │  总元素: 1×150528 = 150,528 ✓
        └─────────────────────────┘

直接融合: X ∈ ℝ^(1×3×224×224) → ℝ^(1×150528)
        ┌─────────────────────────┐
        │ [1, 3, 224, 224]        │ ──→ │ [1, 150528]             │
        └─────────────────────────┘     └─────────────────────────┘
```

**计算图优化**:
```
优化前 - 两次Reshape操作:
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│Input        │───→│  Reshape1    │───→│  Temp        │───→│  Reshape2   │───→ Output
│[1,3,224,224]│    │[1,3,50176]   │    │[1,3,50176]   │    │[1,150528]   │    [1,150528]
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘

优化后 - 单次Reshape操作:
┌─────────────┐    ┌──────────────────────────────────────────────────┐    ┌─────────────┐
│Input        │───→│              Reshape_fused                       │───→│Output       │
│[1,3,224,224]│    │            [1,3,224,224] → [1,150528]           │    │[1,150528]   │
└─────────────┘    └──────────────────────────────────────────────────┘    └─────────────┘
```

**内存布局分析**:

**行优先存储 (C-style)**:
```
原始4D张量 [1,3,224,224] 的内存布局:
Index: [b,c,h,w] → MemoryOffset = b×(3×224×224) + c×(224×224) + h×224 + w

融合后2D张量 [1,150528] 的内存布局:
Index: [b,f] → MemoryOffset = b×150528 + f

关键: 物理内存布局保持不变，只是逻辑视图改变
```

**融合条件检查**:
```python
def can_fuse_reshapes(shape1, shape2, final_shape):
    """检查是否可以安全融合连续的Reshape操作"""

    # 1. 检查元素总数守恒
    if (prod(shape1) != prod(shape2) or
        prod(shape2) != prod(final_shape)):
        return False

    # 2. 检查内存布局兼容性
    if not is_contiguous_memory_layout(shape1, shape2, final_shape):
        return False

    # 3. 检查动态维度处理
    if has_dynamic_dimensions(shape1, shape2, final_shape):
        return requires_runtime_check()

    return True

def is_contiguous_memory_layout(shape1, shape2, final_shape):
    """检查内存布局是否连续兼容"""
    return (is_c_contiguous(shape1, shape2) and
            is_c_contiguous(shape2, final_shape))
```

**性能优化效果**:

**操作复杂度**:
- **Reshape复杂度**: $O(1)$ (仅元数据操作，无数据拷贝)
- **融合收益**: 减少中间张量创建和元数据操作

**内存访问模式**:
```
优化前内存访问:
1. 读取原始张量元数据
2. 创建中间张量元数据  ← 额外开销
3. 读取中间张量元数据  ← 额外开销
4. 创建最终张量元数据

优化后内存访问:
1. 读取原始张量元数据
2. 创建最终张量元数据    ← 直接跳转
```

**特殊情况处理**:

**动态形状融合**:
```python
# 支持包含-1的动态形状
Input: [batch, 3, 224, 224]
Reshape1: [batch, 3, -1]      # -1 自动推断为 50176
Reshape2: [batch, -1]         # -1 自动推断为 150528

融合结果: [batch, 3, 224, 224] → [batch, 150528]
```

**非连续内存处理**:
如果存在transpose等操作影响内存布局，融合可能需要额外检查：
```python
# 需要谨慎处理的情况
Input[1,3,224,224] → Transpose[1,224,224,3] → Reshape[1,-1]
# 此时不能简单融合，需要考虑transpose的内存重排
```

**典型应用场景**:
- **CNN到FC层转换**: 特征图展平为全连接层输入
- **Transformer输入处理**: 多维序列数据重组
- **批处理优化**: 动态批量大小的维度调整

---

#### **7. FreeDimensionOverrideTransformer (自由维度覆盖)**

**数学定义**:
自由维度是指在模型定义中未指定具体数值的维度，通常表示为动态或符号维度：
$$\text{FreeDim} : \mathbb{N}^* \cup \{\text{dynamic}, \text{symbolic}\}$$

**理论原理**:
在深度学习推理中，某些维度（如batch size、sequence length）在编译时未知，需要在运行时确定。FreeDimensionOverride允许在图优化阶段为这些维度提供具体值，以启用更多优化机会。

**维度类型分类**:
```
1. 固定维度 (Fixed):     [224, 224, 3]           ← 编译时已知
2. 自由维度 (Free):      [batch, seq_len, 768]   ← 运行时确定
3. 符号维度 (Symbolic):  [N, M, K]              ← 代数表示
4. 动态维度 (Dynamic):   [-1, None, '?']        ← 框架特定标记
```

**覆盖机制**:
```
原始模型定义:
Input: [batch_size, sequence_length, hidden_dim]
       [   ?    ,       ?         ,     768    ]

运行时信息注入:
batch_size = 32
sequence_length = 512

覆盖后维度:
Input: [32, 512, 768]  ← 现在可以进行静态优化
```

**计算图变换示例**:
```
优化前 - 动态维度限制优化:
┌─────────────────┐    ┌─────────────────────────────┐
│Input            │───→│  MatMul                     │
│[batch, seq, 768]│    │  (无法预分配内存)           │
│[  ?,   ?, 768] │     │  (无法展开循环)             │
└─────────────────┘    └─────────────────────────────┘

优化后 - 固定维度启用优化:
┌─────────────────┐    ┌─────────────────────────────┐
│Input            │───→│  MatMul_optimized           │
│[32, 512, 768]   │    │  (预分配: 32×512×768)       │
│(override应用)   │    │  (循环展开、向量化)         │
└─────────────────┘    └─────────────────────────────┘
```

**应用场景分析**:

**1. 批处理优化**:
```python
# 原始动态批处理
def dynamic_inference(input_batch):  # input: [?, 224, 224, 3]
    # 无法预优化，每次都需要动态分配
    return model(input_batch)

# 维度覆盖优化
def fixed_batch_inference(input_batch):  # input: [32, 224, 224, 3]
    # 可以预分配内存、展开循环、应用更多优化
    return optimized_model(input_batch)
```

**2. 序列长度优化**:
```python
# NLP模型中的序列长度优化
Original:  [batch, seq_len, vocab_size] where seq_len = ?
Override:  [batch, 512, vocab_size]     # 固定最大序列长度

启用优化:
- 预分配attention矩阵 [batch, 512, 512]
- 静态展开transformer层
- 优化内存池分配
```

**实现算法**:
```python
class FreeDimensionOverrideTransformer:
    def __init__(self, dimension_overrides):
        """
        dimension_overrides: Dict[str, int]
        例如: {"batch_size": 32, "seq_length": 512}
        """
        self.overrides = dimension_overrides

    def transform(self, graph):
        for node in graph.nodes:
            for i, input_shape in enumerate(node.input_shapes):
                new_shape = self.apply_overrides(input_shape)
                if new_shape != input_shape:
                    node.input_shapes[i] = new_shape
                    self.propagate_shape_change(graph, node, i, new_shape)

    def apply_overrides(self, shape):
        new_shape = []
        for dim in shape:
            if isinstance(dim, str) and dim in self.overrides:
                new_shape.append(self.overrides[dim])
            elif dim == -1 and "dynamic_dim" in self.overrides:
                new_shape.append(self.overrides["dynamic_dim"])
            else:
                new_shape.append(dim)
        return new_shape
```

**优化收益分析**:

**内存分配优化**:
```
动态分配 (优化前):
- 每次推理都需要查询输入形状
- 运行时内存分配: malloc/free开销
- 无法使用内存池优化

静态分配 (优化后):
- 编译时确定内存布局
- 预分配内存池: 零分配开销
- 可以应用内存对齐优化
```

**计算优化启用**:
```
可启用的优化类型:
1. 循环展开 (Loop Unrolling)
2. 向量化 (Vectorization)
3. 常量折叠 (Constant Folding)
4. 内存预取 (Memory Prefetching)
5. 指令级并行 (ILP)
```

**风险和注意事项**:

**1. 维度不匹配风险**:
```python
# 覆盖设定: batch_size = 32
# 实际输入: batch_size = 64  ← 运行时错误!

解决方案:
- 添加运行时检查
- 使用最大维度覆盖
- 提供回退机制
```

**2. 内存溢出风险**:
```python
# 过大的维度覆盖可能导致内存不足
# 需要根据硬件资源合理设置覆盖值

内存估算:
memory_required = batch_size × seq_length × hidden_dim × sizeof(float32)
if memory_required > available_memory:
    reduce_override_values()
```

**最佳实践**:
```python
# 推荐的维度覆盖策略
dimension_overrides = {
    "batch_size": min(target_batch_size, max_memory_batch_size),
    "seq_length": common_sequence_length,  # 根据数据集统计
    "image_size": standard_image_size,     # 标准输入尺寸
}

# 渐进式优化
def progressive_override(model):
    # 1. 先应用保守的覆盖值
    apply_conservative_overrides(model)

    # 2. 测量性能和内存使用
    profile_model(model)

    # 3. 逐步调整到最优值
    optimize_override_values(model)
```

---

#### **8. GeluFusion (GELU融合)**

**数学定义**:
GELU (Gaussian Error Linear Unit) 激活函数定义为：
$$\text{GELU}(x) = \frac{x}{2} \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

其中 $\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt$ 是误差函数。

**Tanh近似实现**:
由于erf函数计算复杂，通常使用tanh近似：
$$\text{GELU}(x) \approx \frac{x}{2} \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right)\right)$$

**理论推导**:
基于erf函数的泰勒展开式：
$$\text{erf}(z) \approx \tanh\left(\sqrt{\frac{2}{\pi}} \left(z + 0.044715z^3\right)\right)$$

**分解算子序列分析**:
```
1. x³ 计算:        Pow(x, 3)
2. 三次项系数:      Mul(x³, 0.044715)
3. 线性+三次项:     Add(x, 0.044715x³)
4. 缩放因子:        Mul(result, √(2/π))
5. Tanh激活:       Tanh(scaled_result)
6. 偏移:           Add(tanh_result, 1)
7. 归一化:         Mul(shifted_result, 0.5)
8. 原值缩放:       Mul(x, normalized_result)
```

**计算图优化对比**:
```
优化前 - 8个独立算子:
┌───────┐    ┌─────────┐    ┌─────────────┐    ┌─────────────┐
│   x   │───→│ Pow(3)  │───→│Mul(0.044715)│───→│  Add(x)     │
└───────┘    └─────────┘    └─────────────┘    └─────────────┘
                                                       │
┌─────────────┐    ┌─────────┐    ┌─────────────┐     │
│ Mul(√(2/π)) │←───│  Tanh   │←───│   Add(1)    │←────┘
└─────────────┘    └─────────┘    └─────────────┘
       │
┌─────────────┐    ┌─────────────┐
│  Mul(0.5)   │───→│   Mul(x)    │───→ Output
└─────────────┘    └─────────────┘

优化后 - 1个融合算子:
┌───────┐    ┌─────────────┐
│   x   │───→│    GELU     │───→ Output
└───────┘    └─────────────┘
```

**内存访问优化分析**:

**优化前内存模式**:
- 每个中间步骤产生临时张量
- 内存占用: $8 \times N$ (N为张量大小)
- 内存带宽: $16 \times N$ (8次读取 + 8次写入)

**优化后内存模式**:
- 单次读取输入，单次写入输出
- 内存占用: $2 \times N$ (输入+输出)
- 内存带宽: $2 \times N$ (1次读取 + 1次写入)

**数值精度对比**:

| 输入x | 精确GELU | Tanh近似 | 绝对误差 | 相对误差(%) |
|-------|----------|----------|----------|-------------|
| -3.0  | -0.0013  | -0.0011  | 0.0002   | 15.4        |
| -1.0  | -0.1587  | -0.1588  | 0.0001   | 0.06        |
| 0.0   | 0.0000   | 0.0000   | 0.0000   | 0.00        |
| 1.0   | 0.8413   | 0.8412   | 0.0001   | 0.01        |
| 3.0   | 2.9987   | 2.9989   | 0.0002   | 0.007       |

**融合kernel实现示例**:
```cpp
__global__ void gelu_fused_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];

        // 内联计算，避免中间存储
        float x_cubed = x * x * x;
        float tanh_input = sqrtf(2.0f / M_PI) * (x + 0.044715f * x_cubed);
        float tanh_result = tanhf(tanh_input);

        output[idx] = 0.5f * x * (1.0f + tanh_result);
    }
}
```

**性能提升分析**:
- **算子融合**: 8个算子 → 1个算子 (87.5%减少)
- **内存带宽**: 16N → 2N (87.5%减少)
- **Kernel启动**: 8次 → 1次 (GPU并行化更高效)
- **缓存命中**: 显著提升，避免反复读取中间结果

**应用场景**:
- **Transformer模型**: FFN层的激活函数
- **BERT/GPT**: 大规模语言模型的标准激活
- **Vision Transformer**: 计算机视觉Transformer模型

---

#### **9. LayerNormFusion (LayerNorm融合)**

**数学定义**:
Layer Normalization对输入张量 $X \in \mathbb{R}^{B \times S \times D}$ 的每个样本在特征维度上进行归一化：
$$\text{LN}(x_i) = \gamma \odot \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta$$

其中：
- $\mu_i = \frac{1}{D} \sum_{j=1}^{D} x_{i,j}$ (特征维度均值)
- $\sigma_i^2 = \frac{1}{D} \sum_{j=1}^{D} (x_{i,j} - \mu_i)^2$ (特征维度方差)
- $\gamma, \beta \in \mathbb{R}^D$ (可学习的缩放和偏移参数)
- $\epsilon$ (数值稳定性常数，通常为 $10^{-5}$)

**理论原理**:
LayerNorm通过归一化特征维度来稳定训练过程，数学基础是标准化变换：
$$Z = \frac{X - E[X]}{\sqrt{\text{Var}[X] + \epsilon}}$$

这确保每层输入的分布稳定，加速收敛并提高梯度流的稳定性。

**算子分解分析**:
```
1. 均值计算:     μ = ReduceMean(x, axis=-1, keepdims=True)
2. 偏差计算:     x_centered = Sub(x, μ)
3. 方差分子:     x_squared = Pow(x_centered, 2)
4. 方差计算:     σ² = ReduceMean(x_squared, axis=-1, keepdims=True)
5. 稳定性项:     σ²_stable = Add(σ², ε)
6. 标准差:       σ = Sqrt(σ²_stable)
7. 标准化:       x_norm = Div(x_centered, σ)
8. 缩放:         x_scaled = Mul(x_norm, γ)
9. 偏移:         output = Add(x_scaled, β)
```

**计算图优化对比**:
```
优化前 - 9个算子链式计算:
┌─────────┐    ┌──────────────┐    ┌─────────────┐
│ Input   │───→│ ReduceMean   │───→│    Sub      │
│ [B,S,D] │    │  (axis=-1)   │    │ (x - μ)     │
└─────────┘    └──────────────┘    └─────────────┘
                       │                    │
                       │              ┌─────────────┐
                       │              │   Pow(2)    │
                       │              └─────────────┘
                       │                    │
                       │              ┌─────────────┐    ┌─────────────┐
                       │              │ ReduceMean  │───→│  Add(ε)     │
                       │              │  (axis=-1)  │    └─────────────┘
                       │              └─────────────┘          │
                       │                                 ┌─────────────┐
                       │                                 │    Sqrt     │
                       │                                 └─────────────┘
                       │                                       │
                 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                 │   Mul(γ)    │←───│     Div     │←───│             │
                 └─────────────┘    └─────────────┘    └─────────────┘
                       │
                 ┌─────────────┐
                 │   Add(β)    │───→ Output
                 └─────────────┘

优化后 - 1个融合算子:
┌─────────┐    ┌─────────────────────┐
│ Input   │───→│   LayerNorm        │───→ Output
│ [B,S,D] │    │   (γ, β, ε)        │    [B,S,D]
└─────────┘    └─────────────────────┘
```

**数值稳定性优化**:

**Welford在线算法**用于稳定的方差计算：
```cpp
void welford_variance(float* data, int n, float* mean, float* variance) {
    float M = 0.0f, S = 0.0f;

    for (int i = 0; i < n; i++) {
        float delta = data[i] - M;
        M += delta / (i + 1);              // 在线均值更新
        float delta2 = data[i] - M;
        S += delta * delta2;               // 在线方差累积
    }

    *mean = M;
    *variance = S / n;                     // 无偏方差估计
}
```

**融合kernel实现**:
```cpp
__global__ void layer_norm_fused_kernel(
    float* input, float* output, float* gamma, float* beta,
    int batch_size, int seq_len, int hidden_dim, float epsilon) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len) {
        float* x = input + idx * hidden_dim;
        float* y = output + idx * hidden_dim;

        // 第一次遍历：计算均值
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            sum += x[i];
        }
        float mean = sum / hidden_dim;

        // 第二次遍历：计算方差
        float var_sum = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float diff = x[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / hidden_dim;
        float inv_std = rsqrtf(variance + epsilon);  // 快速逆平方根

        // 第三次遍历：归一化和仿射变换
        for (int i = 0; i < hidden_dim; i++) {
            y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}
```

**内存访问优化**:

**优化前模式**:
- 9次完整张量读写操作
- 内存访问量: $18 \times B \times S \times D$
- 大量中间张量存储

**优化后模式**:
- 单kernel内完成所有计算
- 内存访问量: $2 \times B \times S \times D$ (仅输入输出)
- 寄存器内计算，避免全局内存中间结果

**性能分析**:

| 模型维度 | 优化前时间(ms) | 优化后时间(ms) | 加速比 |
|----------|----------------|----------------|--------|
| [32,512,768] | 2.45 | 0.31 | 7.9x |
| [64,1024,1024] | 9.87 | 1.23 | 8.0x |
| [128,2048,2048] | 38.2 | 4.76 | 8.0x |

**梯度融合优化**:
训练时的反向传播也可以融合：
$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sigma} \left( \frac{\partial L}{\partial y_i} - \frac{1}{D}\sum_j \frac{\partial L}{\partial y_j} - \frac{x_i - \mu}{\sigma^2 D} \sum_j (x_j - \mu) \frac{\partial L}{\partial y_j} \right)$$

**应用场景**:
- **Transformer架构**: 每个子层后的归一化
- **BERT/GPT**: 自注意力和FFN后的LayerNorm
- **Vision Transformer**: 图像块处理的归一化层

**图解**:
```
原始LayerNorm计算步骤:
Input[B, S, D]
    ↓
1. mean = ReduceMean(input, axis=-1)      [B, S, 1]
    ↓
2. variance = ReduceMean((input-mean)², axis=-1)  [B, S, 1]
    ↓
3. std = sqrt(variance + epsilon)
    ↓
4. normalized = (input - mean) / std
    ↓
5. output = normalized * gamma + beta
    ↓
Output[B, S, D]

融合后一步完成:
Input[B, S, D] → LayerNorm_fused → Output[B, S, D]
```

**收益**: 减少70%的内存访问，提升40-60%性能。

---

#### **10. EliminateUnusedOutputs (未使用输出消除)**

**数学定义**: 对于图中的节点 $v$ 具有多个输出 $\{o_1, o_2, \ldots, o_k\}$，如果某些输出 $o_i$ 在图中没有被任何后续节点使用，则消除这些输出：
$$\text{EliminateUnused}(v) = v' \text{ with outputs } \{o_j : o_j \text{ is used}\}$$

**理论原理**:
- **死码消除**: 移除计算但不使用的中间结果
- **内存优化**: 减少不必要的内存分配
- **缓存友好**: 提高数据局部性

**优化条件**:
```cpp
// 检测未使用输出
bool has_unused_outputs = false;
for (int i = 0; i < node.OutputCount(); ++i) {
    if (node.GetOutputs()[i].GetUseCount() == 0) {
        has_unused_outputs = true;
        break;
    }
}
```

**性能提升**: 内存使用减少10-30%，缓存命中率提升

---

#### **11. EliminateSharedInitializer (共享初始化器消除)**

**数学定义**: 对于相同内容的初始化器 $W_1, W_2, \ldots, W_n$，合并为单个共享初始化器：
$$\text{Merge}(\{W_i : \text{content}(W_i) = \text{content}(W_j)\}) = W_{\text{shared}}$$

**理论原理**:
- **内存去重**: 避免存储重复的权重数据
- **加载优化**: 减少模型文件大小
- **缓存效率**: 提高权重访问的缓存命中率

**去重算法**:
```cpp
// 计算初始化器哈希值
auto hash_value = ComputeHash(initializer.data(), initializer.size());
if (hash_map.find(hash_value) != hash_map.end()) {
    // 找到重复，进行合并
    MergeInitializers(hash_map[hash_value], initializer);
}
```

**优化效果**: 模型大小减少5-20%，内存使用优化

---

#### **12. PropagateInputShapes (输入形状传播)**

**数学定义**: 对于计算图中的每个节点 $v_i$，基于输入形状 $S_{in}$ 推导输出形状 $S_{out}$：
$$S_{out} = \text{ShapeInference}(v_i, S_{in})$$

**形状推导规则**:
- **线性层**: $S_{out} = (N, \ldots, D_{out})$ where $D_{in} \times D_{out}$ weight matrix
- **卷积层**: $S_{out} = (N, C_{out}, H_{out}, W_{out})$
- **池化层**: $S_{out} = (N, C, \lfloor\frac{H + 2p - k}{s}\rfloor + 1, \lfloor\frac{W + 2p - k}{s}\rfloor + 1)$

**传播算法**:
```cpp
void PropagateShapes(Graph& graph) {
    for (auto& node : graph.Nodes()) {
        auto output_shapes = InferOutputShapes(node.InputShapes(), node.OpType());
        node.SetOutputShapes(output_shapes);
    }
}
```

**收益**: 提前优化内存分配，减少运行时形状推导开销

---

#### **13. IdentityElimination (Identity消除增强)**

**数学定义**: 扩展的Identity节点检测，包括隐式身份变换：
$$\text{DetectIdentity}(f) = \begin{cases}
\text{true} & \text{if } f(x) = x \\
\text{true} & \text{if } \text{Reshape}(x, \text{shape}(x)) \\
\text{true} & \text{if } \text{Transpose}(x, [0,1,2,...])
\end{cases}$$

**检测模式**:
1. **显式Identity**: `Identity(x) → x`
2. **冗余Reshape**: `Reshape(x, shape(x)) → x`
3. **平凡Transpose**: `Transpose(x, [0,1,2,...]) → x`
4. **空Cast**: `Cast(x, dtype(x)) → x`

**优化示例**:
```
# 原始图
x → Reshape(x, [2,3,4]) → Identity → Transpose([0,1,2]) → y

# 优化后
x → y  (所有中间节点被消除)
```

**性能影响**: 减少15-25%的无效算子调用

---

#### **14. MemoryOptimization (内存布局优化)**

**数学定义**: 优化张量内存布局以减少内存拷贝和提高缓存效率：
$$\text{Layout}(T) = \arg\min_{L} \sum_{op} \text{CopyCost}(T, L) + \text{AccessCost}(T, L)$$

**布局优化策略**:
1. **连续化**: 确保张量在内存中连续存储
2. **对齐优化**: 按照硬件要求对齐内存地址
3. **重排序**: 优化张量维度顺序以提高访问效率

**内存模式**:
```cpp
enum class MemoryPattern {
    NCHW,    // 通道优先 (卷积网络标准)
    NHWC,    // 空间优先 (移动设备友好)
    NC,      // 全连接层标准
    HWC      // 图像处理标准
};
```

**优化效果**: 内存带宽利用率提升20-40%

---

#### **15. SubgraphCluster (子图聚类)**

**数学定义**: 将相似或相关的操作聚类成子图以便进行批量优化：
$$\text{Cluster}(G) = \{C_1, C_2, \ldots, C_k\} \text{ where } \bigcup C_i = V(G)$$

**聚类算法**:
```cpp
struct ClusterCriteria {
    bool same_device;      // 相同执行设备
    bool data_dependency;  // 数据依赖性
    bool memory_pattern;   // 内存访问模式
    bool compute_intensity; // 计算密集度
};
```

**聚类策略**:
1. **算子类型聚类**: 相同类型算子组合
2. **数据流聚类**: 按数据依赖关系分组
3. **计算强度聚类**: 按计算复杂度分组
4. **内存访问聚类**: 按内存访问模式分组

**ASCII图示**:
```
原始图:
A → B → C → D
↓   ↓   ↓   ↓
E → F → G → H

聚类后:
[Cluster1: A,B,E,F] → [Cluster2: C,D,G,H]
```

**优化收益**: 减少30-50%的设备间通信，提升并行执行效率

---

#### **16. DeadCodeElimination (死码消除)**

**数学定义**: 识别并移除计算图中永远不会被执行或其结果永远不会被使用的节点：
$$\text{DeadCode}(v) = \{v \in V : \text{Reachable}(v, \text{outputs}) = \text{false}\}$$

**死码检测算法**:
```cpp
std::unordered_set<Node*> FindDeadNodes(const Graph& graph) {
    std::unordered_set<Node*> reachable;
    std::queue<Node*> worklist;

    // 从输出节点开始反向遍历
    for (auto& output : graph.GetOutputs()) {
        worklist.push(output.GetNode());
    }

    while (!worklist.empty()) {
        Node* node = worklist.front();
        worklist.pop();

        if (reachable.insert(node).second) {
            for (auto& input : node->GetInputs()) {
                worklist.push(input.GetNode());
            }
        }
    }

    // 返回不可达节点(死码)
    std::unordered_set<Node*> dead_nodes;
    for (auto& node : graph.Nodes()) {
        if (reachable.find(&node) == reachable.end()) {
            dead_nodes.insert(&node);
        }
    }
    return dead_nodes;
}
```

**优化效果**: 减少10-30%的计算开销，降低内存使用

---

#### **17. ValueInfo (值信息传播)**

**数学定义**: 在计算图中传播值的类型、形状和范围信息：
$$\text{ValueInfo}(v) = \{\text{dtype}, \text{shape}, \text{min\_val}, \text{max\_val}, \text{sparsity}\}$$

**信息传播规则**:
```cpp
struct ValueInfo {
    DataType dtype;
    std::vector<int64_t> shape;
    float min_value, max_value;
    float sparsity_ratio;
    bool is_constant;
};

ValueInfo PropagateValueInfo(const OpType& op,
                           const std::vector<ValueInfo>& inputs) {
    switch (op) {
        case OpType::Add:
            return {inputs[0].dtype,
                   inputs[0].shape,
                   inputs[0].min_value + inputs[1].min_value,
                   inputs[0].max_value + inputs[1].max_value};
        case OpType::Mul:
            return {inputs[0].dtype,
                   inputs[0].shape,
                   ComputeMinProduct(inputs[0], inputs[1]),
                   ComputeMaxProduct(inputs[0], inputs[1])};
        // ... 其他算子
    }
}
```

**应用场景**: 量化优化、内存分配、算子选择

---

#### **18. SliceElimination (Slice消除增强)**

**数学定义**: 检测和消除冗余的Slice操作：
$$\text{EliminateSlice}(x[\text{start}:\text{end}:\text{step}]) = \begin{cases}
x & \text{if start=0, end=shape, step=1} \\
\text{EmptyTensor} & \text{if start≥end} \\
\text{Optimized} & \text{otherwise}
\end{cases}$$

**优化模式**:
1. **完整切片**: `x[0:len(x):1] → x`
2. **空切片**: `x[5:3] → EmptyTensor`
3. **单元素切片**: `x[i:i+1] → x[i]`
4. **连续切片合并**: `x[a:b][c:d] → x[a+c:a+d]`

**切片合并算法**:
```cpp
bool CanMergeSlices(const SliceOp& slice1, const SliceOp& slice2) {
    return slice1.output_connects_to(slice2) &&
           slice1.axis == slice2.axis &&
           slice1.end == slice2.start;
}

SliceOp MergeSlices(const SliceOp& slice1, const SliceOp& slice2) {
    return SliceOp{slice1.start, slice2.end, slice1.step, slice1.axis};
}
```

**性能提升**: 减少20-35%的内存拷贝操作

---

#### **19. BroadcastElimination (广播消除)**

**数学定义**: 优化不必要的广播操作：
$$\text{Broadcast}(x, \text{shape}) = \begin{cases}
x & \text{if shape}(x) = \text{shape} \\
\text{OptimizedBroadcast}(x) & \text{if compatible}
\end{cases}$$

**广播优化策略**:
1. **形状匹配检测**: 检查是否需要实际广播
2. **维度复用**: 利用内存布局避免数据复制
3. **批量广播**: 合并多个广播操作

**优化示例**:
```python
# 原始代码
x = np.random.rand(1, 1, 256)
y = np.broadcast_to(x, (32, 128, 256))  # 显式广播

# 优化后 - 利用内存视图
x_view = x.view()  # 零拷贝视图
y = x_view.expand(32, 128, 256)  # 惰性广播
```

**内存节省**: 减少50-80%的广播相关内存使用

---

#### **20. TransposeOptimizer (转置优化器)**

**数学定义**: 优化和消除不必要的转置操作：
$$\text{Transpose}(x, \text{perm}) = \begin{cases}
x & \text{if perm} = [0,1,2,\ldots] \\
\text{Fused} & \text{if can\_fuse\_with\_next\_op}
\end{cases}$$

**转置优化策略**:
1. **恒等转置消除**: 移除不改变数据顺序的转置
2. **转置链合并**: 合并连续的转置操作
3. **算子融合**: 将转置融合到后续计算中

**转置链合并**:
```cpp
std::vector<int> ComposePermutations(const std::vector<int>& perm1,
                                   const std::vector<int>& perm2) {
    std::vector<int> result(perm1.size());
    for (size_t i = 0; i < perm1.size(); ++i) {
        result[i] = perm1[perm2[i]];
    }
    return result;
}

// 示例: Transpose([0,2,1]) → Transpose([1,0,2]) = Transpose([2,0,1])
```

**ASCII图示**:
```
优化前:
Input [B,H,W,C] → Transpose[0,3,1,2] → [B,C,H,W] → Conv → Transpose[0,2,3,1] → [B,H,W,C]

优化后:
Input [B,H,W,C] → Conv(layout_optimized) → [B,H,W,C]
```

**性能提升**:
- **内存带宽**: 减少60-90%的数据移动
- **计算效率**: 提升25-45%的算子执行速度
- **缓存命中**: 提高内存访问局部性

---

## Level 2: 扩展优化 (ORT_ENABLE_EXTENDED)

### **30个扩展优化算法**

Level 2在Level 1基础上新增了30个高级优化算法，总计65个优化器。主要包含复杂的算子融合、量化优化和特定硬件优化。

**源码定义**:
- RewriteRule: `onnxruntime/core/optimizer/graph_transformer_utils.cc:GenerateRewriteRules(TransformerLevel::Level2)`
- GraphTransformer: `onnxruntime/core/optimizer/graph_transformer_utils.cc:GenerateTransformers(TransformerLevel::Level2)`

---

#### **Level 2 新增RewriteRule (3个)**

**1. ClipQuantFusion** (`qdq_transformer/clip_quantizelinear.h`)
```cpp
rules.push_back(std::make_unique<ClipQuantFusion>());
```
- **作用**: 融合Clip和QuantizeLinear操作
- **原理**: 将Clip操作的范围限制集成到量化过程中

**2. ReluQuantFusion** (`qdq_transformer/relu_quantizelinear.h`)
```cpp
rules.push_back(std::make_unique<ReluQuantFusion>());
```
- **作用**: 融合ReLU和QuantizeLinear操作
- **原理**: 将ReLU的非负约束集成到量化中

**3. GemmTransposeFusion** (`gemm_transpose_fusion.h`)
```cpp
rules.push_back(std::make_unique<GemmTransposeFusion>());
```
- **作用**: Level 2版本的GEMM转置融合
- **原理**: 支持更复杂的转置+GEMM模式

---

#### **Level 2 新增GraphTransformer (27个)**

**量化优化类 (5个)**:

**1. QDQS8ToU8Transformer** (`qdq_transformer/qdq_s8_to_u8.h`)
```cpp
if (!qdq_is_int8_allowed) {
    transformers.emplace_back(std::make_unique<QDQS8ToU8Transformer>(avx2_precision_mode, cpu_ep));
}
```
- **作用**: S8到U8量化转换
- **原理**: 将有符号8位量化转换为无符号8位量化

**2. QDQSelectorActionTransformer** (`qdq_transformer/selectors_actions/qdq_selector_action_transformer.h`)
```cpp
transformers.emplace_back(std::make_unique<QDQSelectorActionTransformer>(qdq_is_int8_allowed, SatApplyContextVariant{}, qdq_matmulnbits_accuracy_level, intra_op_thread_pool));
```
- **作用**: QDQ选择器动作转换器
- **原理**: 基于选择器-动作模式的QDQ优化

**3. Avx2WeightS8ToU8Transformer** (`qdq_transformer/avx2_weight_s8_to_u8.h`)
```cpp
#ifdef MLAS_TARGET_AMD64_IX86
if (avx2_precision_mode) {
    transformers.emplace_back(std::make_unique<Avx2WeightS8ToU8Transformer>(cpu_ep));
}
#endif
```
- **作用**: AVX2权重S8到U8转换
- **原理**: 针对AVX2指令集的权重量化优化

**4. MatMulNBitsFusion** (`matmul_nbits_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<MatMulNBitsFusion>(cpu_ep));
```
- **作用**: MatMul N-bit量化融合
- **原理**: 支持4-bit/2-bit等低精度量化的MatMul

**5. QDQFinalCleanupTransformer** (`qdq_transformer/qdq_final_cleanup.h`)
```cpp
transformers.emplace_back(std::make_unique<QDQFinalCleanupTransformer>(enable_quant_qdq_cleanup));
```
- **作用**: QDQ最终清理转换器
- **原理**: 在所有其他变换后进行QDQ节点的最终清理

**运算符融合类 (15个)**:

**6. GemmActivationFusion** (`gemm_activation_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<GemmActivationFusion>(cpu_ep));
```
- **作用**: GEMM激活函数融合
- **原理**: 将GEMM后的激活函数合并为单个操作

**7. MatMulIntegerToFloatFusion** (`matmul_integer_to_float.h`)
```cpp
transformers.emplace_back(std::make_unique<MatMulIntegerToFloatFusion>(cpu_dml_acl_eps));
```
- **作用**: 整数到浮点MatMul融合
- **原理**: 优化混合精度的矩阵乘法

**8. DynamicQuantizeMatMulFusion** (`dynamic_quantize_matmul_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<DynamicQuantizeMatMulFusion>(cpu_acl_eps));
```
- **作用**: 动态量化MatMul融合
- **原理**: 运行时动态量化的MatMul优化

**9. ConvActivationFusion** (`conv_activation_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<ConvActivationFusion>(cpu_rocm_acl_armnn_js_webgpu_eps));
```
- **作用**: 卷积激活函数融合
- **原理**: 将卷积后的激活函数合并

**10. GeluFusion** (`gelu_fusion.h`) - Level 2版本
```cpp
transformers.emplace_back(std::make_unique<GeluFusion>(cpu_acl_cuda_dml_rocm_eps, level));
```
- **作用**: Level 2版本的GELU融合
- **原理**: 支持更多EP的GELU优化

**11. LayerNormFusion** (`layer_norm_fusion.h`) - Level 2版本
```cpp
transformers.emplace_back(std::make_unique<LayerNormFusion>(cpu_acl_cuda_dml_rocm_eps, level));
```
- **作用**: Level 2版本的LayerNorm融合
- **原理**: 支持更多EP和更复杂模式

**12. SimplifiedLayerNormFusion** (`layer_norm_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<SimplifiedLayerNormFusion>(cpu_cuda_rocm_eps));
```
- **作用**: 简化LayerNorm融合
- **原理**: 针对简化版LayerNorm的优化

**13. AttentionFusion** (`attention_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<AttentionFusion>(cpu_acl_cuda_dml_rocm_eps));
```
- **作用**: 注意力机制融合
- **原理**: 将Multi-Head Attention的多个操作融合

**14. EmbedLayerNormFusion** (`embed_layer_norm_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<EmbedLayerNormFusion>(cpu_acl_cuda_dml_rocm_eps));
```
- **作用**: 嵌入层归一化融合
- **原理**: 融合Embedding + LayerNorm模式

**15. GatherSliceToSplitFusion** (`gather_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<GatherSliceToSplitFusion>(cpu_cuda_rocm_eps));
```
- **作用**: Gather-Slice到Split融合
- **原理**: 将Gather+Slice模式转换为更高效的Split

**16. GatherToSliceFusion** (`gather_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<GatherToSliceFusion>(cpu_cuda_rocm_eps));
```
- **作用**: Gather到Slice融合
- **原理**: 将特定的Gather操作转换为Slice

**17. MatmulTransposeFusion** (`matmul_transpose_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<MatmulTransposeFusion>(cpu_cuda_dml_rocm_eps));
```
- **作用**: MatMul转置融合
- **原理**: 将转置操作参数化到MatMul中

**18. BiasGeluFusion** (`bias_gelu_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<BiasGeluFusion>(cpu_acl_cuda_dml_rocm_eps));
```
- **作用**: 偏置GELU融合
- **原理**: 融合Add(bias) + GELU模式

**19. SkipLayerNormFusion** (`skip_layer_norm_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<SkipLayerNormFusion>(cpu_acl_cuda_dml_rocm_eps));
```
- **作用**: 跳跃连接LayerNorm融合
- **原理**: 融合Add(skip) + LayerNorm模式

**20. FastGeluFusion** (`fast_gelu_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<FastGeluFusion>(cpu_cuda_dml_rocm_eps));
```
- **作用**: 快速GELU融合
- **原理**: 使用近似算法的高效GELU实现

**高级融合类 (7个)**:

**21. GroupQueryAttentionFusion** (`group_query_attention_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<GroupQueryAttentionFusion>(cuda_eps));
```
- **作用**: 分组查询注意力融合
- **原理**: 针对GQA架构的专门优化

**22. MatMulAddFusion** (`matmul_add_fusion.h`) - Level 2版本
```cpp
transformers.emplace_back(std::make_unique<MatMulAddFusion>(no_limit_empty_ep_list, false));
```
- **作用**: Level 2版本MatMul+Add融合
- **原理**: 清理注意力融合后剩余的MatMul+Add

**23. QuickGeluFusion** (`quick_gelu_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<QuickGeluFusion>(cpu_acl_cuda_dml_rocm_eps));
```
- **作用**: 快速GELU融合
- **原理**: 另一种GELU的高效实现

**24. GeluApproximation** (`gelu_approximation.h`) *可选*
```cpp
if (enable_gelu_approximation) {
    transformers.emplace_back(std::make_unique<GeluApproximation>(cpu_cuda_rocm_eps));
}
```
- **作用**: GELU近似优化
- **原理**: 用更快的近似函数替换精确的GELU

**25. BiasSoftmaxFusion** (`bias_softmax_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<BiasSoftmaxFusion>(cpu_cuda_rocm_eps));
```
- **作用**: 偏置Softmax融合
- **原理**: 融合Add(bias) + Softmax模式

**26. BiasDropoutFusion** (`bias_dropout_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<BiasDropoutFusion>(cuda_rocm_eps));
```
- **作用**: 偏置Dropout融合
- **原理**: 融合Add(bias) + Dropout模式

**27. MatMulScaleFusion** (`matmul_scale_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<MatMulScaleFusion>(cpu_acl_cuda_dml_rocm_eps));
```
- **作用**: MatMul缩放融合
- **原理**: 将MatMul后的缩放操作合并

**训练专用优化器 (3个，仅在训练时)**:

**28. BitmaskDropoutReplacement** (`orttraining/core/optimizer/bitmask_dropout_replacement.h`)
```cpp
#ifdef ENABLE_TRAINING
transformers.emplace_back(std::make_unique<BitmaskDropoutReplacement>(cuda_rocm_eps));
#endif
```
- **作用**: 位掩码Dropout替换
- **原理**: 用更高效的位掩码实现替换标准Dropout

**29. BiasSoftmaxDropoutFusion** (`orttraining/core/optimizer/bias_softmax_dropout_fusion.h`)
```cpp
#ifdef ENABLE_TRAINING
transformers.emplace_back(std::make_unique<BiasSoftmaxDropoutFusion>(cuda_rocm_eps));
#endif
```
- **作用**: 偏置Softmax Dropout融合
- **原理**: 融合Add(bias) + Softmax + Dropout模式

**30. SceLossGradBiasFusion** (`orttraining/core/optimizer/sce_loss_grad_bias_fusion.h`)
```cpp
#ifdef ENABLE_TRAINING
transformers.emplace_back(std::make_unique<SceLossGradBiasFusion>(cpu_cuda_rocm_eps));
#endif
```
- **作用**: SCE损失梯度偏置融合
- **原理**: 融合SoftmaxCrossEntropy的梯度和偏置计算

---

#### **训练专用Level 2优化器 (5个)**

**源码定义**: `orttraining/orttraining/core/optimizer/graph_transformer_utils.cc:GeneratePreTrainingTransformers(TransformerLevel::Level2)`

**训练专用RewriteRule (2个)**:

**1. ConcatReplacement** (`orttraining/core/optimizer/concat_replacement.h`)
```cpp
ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<ConcatReplacement>()));
```
- **作用**: Concat替换
- **原理**: 将Concat操作替换为训练友好的版本，优化内存使用

**2. TransposeReplacement** (`orttraining/core/optimizer/transpose_replacement.h`)
```cpp
ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<TransposeReplacement>()));
```
- **作用**: Transpose替换
- **原理**: 将Transpose操作替换为训练优化版本

**训练专用GraphTransformer (3个)**:

**1. 条件EmbedLayerNormFusion** (`core/optimizer/embed_layer_norm_fusion.h`)
```cpp
#ifdef USE_CUDA
transformers.emplace_back(std::make_unique<EmbedLayerNormFusion>(compatible_eps));
#endif
```
- **作用**: Embed层归一化融合（仅CUDA）
- **原理**: 在CUDA上融合Embedding和LayerNorm操作

**2. BiasSoftmaxFusion** (`core/optimizer/bias_softmax_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<BiasSoftmaxFusion>(compatible_eps));
```
- **作用**: Bias-Softmax融合
- **原理**: 融合偏置加法和Softmax计算

**3. MatMulIntegerToFloatFusion** (`core/optimizer/matmul_integer_to_float_fusion.h`)
```cpp
transformers.emplace_back(std::make_unique<MatMulIntegerToFloatFusion>(compatible_eps));
```
- **作用**: 整数MatMul到浮点融合
- **原理**: 训练中的混合精度优化

**其他训练Level 2优化**:

**CUDA特定优化**:
```cpp
#ifdef USE_CUDA
transformers.emplace_back(std::make_unique<AttentionFusion>(compatible_eps));
transformers.emplace_back(std::make_unique<SkipLayerNormFusion>(compatible_eps));
#endif
```

**条件融合优化**:
```cpp
if (config.enable_gelu_approximation) {
    transformers.emplace_back(std::make_unique<GeluApproximation>(compatible_eps));
}
```

**训练模式优化器组织结构**:

**源码位置**: `orttraining/orttraining/core/optimizer/graph_transformer_utils.cc`

**主要函数**:
```cpp
// 生成训练前优化器
std::vector<std::unique_ptr<GraphTransformer>> GeneratePreTrainingTransformers(
    TransformerLevel level,
    const SessionOptions& session_options,
    const IExecutionProvider& execution_provider,
    const InlinedHashSet<std::string>& weights_to_train,
    const TrainingOptimizationConfig& config);

// 生成训练优化器
std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformers(
    TransformerLevel level,
    const SessionOptions& session_options,
    const IExecutionProvider& execution_provider,
    const std::unordered_set<std::string>& weights_to_train);
```

**训练与推理的区别**:
1. **权重排除**: 训练优化器排除可训练权重的常量折叠
2. **梯度保持**: 确保所有变换保持梯度流
3. **内存优化**: 特殊的内存管理策略适应训练
4. **重计算**: 支持重计算策略减少内存使用

---

### **Level 2 优化效果**

- **性能提升**: 20-50%（在Level 1基础上）
- **稳定性**: 高，推荐用于生产环境
- **兼容性**: 支持多种Execution Provider
- **特点**: 复杂算子融合、量化优化、硬件特定优化

#### **1. ClipQuantFusion (Clip量化融合)**
**原理**: 将Clip和QuantizeLinear操作融合。

#### **2. ReluQuantFusion (ReLU量化融合)**
**原理**: 将ReLU和QuantizeLinear操作融合。

#### **3. GemmTransposeFusion (GEMM转置融合)**
**原理**: Level 2版本的GEMM转置融合，支持更复杂的模式。

### **B. Level 2 高级GraphTransformer (52个)**

#### **1. 转置优化 (CPU特定)**

##### **TransposeOptimizer (转置优化器 - CPU版)**

**原理**: 专门针对CPU执行提供者的转置优化，在Level 1基础上进行更深入的优化。

**特点**:
- 只在节点分配到CPU EP后运行
- CPU特定的转置模式识别
- 更激进的转置移动策略

---

#### **2. 量化优化集群 (8个)**

##### **QDQS8ToU8Transformer (有符号到无符号量化转换)**

**原理**: 将S8量化转换为U8量化，利用某些硬件的U8性能优势。

**应用场景**:
- AVX2精度模式下的优化
- x86架构的SIMD指令优化

##### **QDQSelectorActionTransformer (QDQ选择动作优化器)**

**原理**: 最复杂的量化优化器，处理各种QDQ模式的选择和融合。

**功能**:
- QDQ模式识别和融合
- 量化精度级别控制
- MatMulNBits精度优化
- 多线程并行优化

**配置参数**:
```cpp
// 是否允许int8量化
bool qdq_is_int8_allowed = true;

// MatMulNBits精度级别 (1-4)
int64_t qdq_matmulnbits_accuracy_level = 4;

// 线程池配置
concurrency::ThreadPool* intra_op_thread_pool;
```

---

#### **3. 矩阵运算优化集群 (6个)**

##### **GemmActivationFusion (GEMM激活融合)**

**原理**: 将GEMM操作与激活函数融合。

**支持的激活函数**: ReLU, Sigmoid, Tanh等

**优化前**:
```
Input1 ──┐
         GEMM → temp → ReLU → Output
Input2 ──┘
```

**优化后**:
```
Input1 ──┐
         GEMM_ReLU → Output
Input2 ──┘
```

##### **MatMulIntegerToFloatFusion (整数MatMul到浮点融合)**

**数学定义**:
将整数矩阵乘法与浮点转换融合，优化量化推理中的混合精度计算：
$$\text{Float}(\text{MatMulInteger}(A_{int8}, B_{int8})) \rightarrow \text{MatMulFloat}(A_{int8}, B_{int8})$$

**理论原理**:
在量化神经网络中，经常需要将整数矩阵乘法结果转换为浮点数进行后续计算。通过融合可以：
1. **减少数据类型转换开销**
2. **优化内存访问模式**
3. **利用硬件混合精度指令**

**数值转换过程**:
```
标准整数MatMul:
A[int8] × B[int8] = C[int32]  ← 防止溢出使用int32累加
Cast(C[int32]) = C[float32]   ← 显式类型转换

融合后:
A[int8] × B[int8] = C[float32] ← 直接输出浮点结果
```

**硬件加速支持**:
- **Intel VNNI**: 支持INT8×INT8→INT32，可扩展到直接输出FP32
- **ARM Neon**: int8矩阵乘法指令可配置输出类型
- **NVIDIA Tensor Core**: 支持INT8输入，FP32输出的混合精度

**性能优势**:
- **内存带宽**: 减少中间int32结果的存储
- **延迟降低**: 避免额外的类型转换pass
- **缓存效率**: 减少一次内存往返

**支持Provider**: CPU, DML, ACL

---

##### **DynamicQuantizeMatMulFusion (动态量化MatMul融合)**

**数学定义**:
将运行时量化与矩阵乘法融合，实现动态量化推理：
$$\text{MatMul}(\text{Quantize}(A_{fp32}), B_{int8}) \rightarrow \text{DynQuantMatMul}(A_{fp32}, B_{int8})$$

**理论原理**:
动态量化在推理时根据输入数据的统计特性确定量化参数：
```
量化参数计算:
scale = (max(A) - min(A)) / (2^bits - 1)
zero_point = round(-min(A) / scale)

量化过程:
A_quantized = round(A / scale) + zero_point
```

**融合优化流程**:
```
优化前 - 分离操作:
Input_A[fp32] → DynamicQuantize → A_quantized[int8] → MatMul → Output[int32]
Input_B[int8] ──────────────────────────────────────────────┘

优化后 - 融合操作:
Input_A[fp32] ──┐
                ├─→ DynamicQuantizeMatMul → Output[int32]
Input_B[int8] ──┘
```

**动态范围检测算法**:
```cpp
void compute_dynamic_quantization_params(float* input, int size,
                                        float* scale, int* zero_point) {
    // 1. 计算数据统计
    float min_val = *std::min_element(input, input + size);
    float max_val = *std::max_element(input, input + size);

    // 2. 对称量化 (zero_point = 0)
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    *scale = abs_max / 127.0f;  // int8范围: [-127, 127]
    *zero_point = 0;

    // 3. 非对称量化 (利用全部动态范围)
    // *scale = (max_val - min_val) / 255.0f;
    // *zero_point = round(-min_val / *scale) - 128;
}
```

**内存优化效果**:
- **临时存储**: 消除量化后的中间张量存储
- **带宽节省**: 直接处理FP32输入，无需额外量化数据传输

**适用场景**:
- **权重静态，激活动态**: 权重预量化，激活运行时量化
- **Transformer推理**: 注意力计算中的动态量化优化

**支持Provider**: CPU, ACL

---

##### **MatmulTransposeFusion (MatMul转置融合)**

**数学定义**:
将矩阵转置与矩阵乘法融合，利用BLAS库的转置参数：
$$\text{MatMul}(\text{Transpose}(A), B) \rightarrow \text{MatMul}(A, B, \text{transA}=\text{True})$$

**BLAS调用优化**:
```cpp
// 优化前 - 显式转置 + MatMul
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
           M, N, K, 1.0, A_transposed, K, B, N, 0.0, C, N);

// 优化后 - 直接转置参数
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
           M, N, K, 1.0, A, M, B, N, 0.0, C, N);
```

**内存访问优化**: 避免显式转置的内存重排，直接在矩阵乘法中处理不同的访问模式

**支持Provider**: CPU, CUDA, DML, ROCm

---

##### **MatMulScaleFusion (MatMul缩放融合)**

**数学定义**:
将矩阵乘法与标量缩放融合：
$$\alpha \cdot \text{MatMul}(A, B) \rightarrow \text{MatMul}(A, B, \alpha)$$

**GEMM参数利用**:
利用GEMM的alpha参数直接实现缩放，无需额外的elementwise乘法：
$$C = \alpha \cdot A \cdot B + \beta \cdot C$$

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

---

##### **MatMulActivationFusion (MatMul激活融合)**

**数学定义**:
将矩阵乘法与激活函数融合，常见于全连接层：
$$\sigma(\text{MatMul}(A, B) + \text{bias}) \rightarrow \text{MatMulActivation}(A, B, \text{bias}, \sigma)$$

**DirectML专用优化**:
利用DirectML的融合kernel实现高效的GPU计算，支持多种激活函数（ReLU, GELU, Sigmoid等）

**支持Provider**: DML专用

---

#### **4. 卷积优化集群 (3个)**

##### **ConvActivationFusion (Conv+Activation融合)**

**原理**: 将卷积和激活函数融合，避免中间结果的存储。

**支持的激活函数**: ReLU, ReLU6, Sigmoid, Tanh, Swish等

**支持Provider**: CPU, ROCm, ACL, ArmNN, JS, WebGPU

**优化前**:
```
Input → Conv → ReLU → Output
```

**优化后**:
```
Input → ConvReLU → Output
```

**实现细节**:
```cpp
// 融合后的伪代码
for (int i = 0; i < output_size; i++) {
    float conv_result = convolution(input, weights, i);
    output[i] = max(0.0f, conv_result);  // ReLU直接应用
}
```

---

#### **1. ConvActivationFusion (卷积激活融合)**

**数学定义**: 卷积操作后跟激活函数的融合：
$$\text{ConvActivation}(X, W, b) = \sigma(\text{Conv}(X, W) + b)$$

其中：
- $X \in \mathbb{R}^{N \times C_{in} \times H \times W}$ 是输入特征图
- $W \in \mathbb{R}^{C_{out} \times C_{in} \times K_H \times K_W}$ 是卷积核
- $b \in \mathbb{R}^{C_{out}}$ 是偏置
- $\sigma$ 是激活函数（ReLU, Sigmoid, Tanh等）

**理论原理**:
1. **内存局部性**: 在同一内存访问周期内完成卷积和激活计算
2. **缓存效率**: 避免中间结果写入内存再读取
3. **向量化优化**: 激活函数可以向量化应用到卷积结果

**卷积运算细节**:
$$(\text{Conv}(X, W))_{n,c,h,w} = \sum_{i=0}^{C_{in}-1} \sum_{j=0}^{K_H-1} \sum_{k=0}^{K_W-1} X_{n,i,h \cdot s+j,w \cdot s+k} \cdot W_{c,i,j,k}$$

**融合后的计算**:
$$\text{Output}_{n,c,h,w} = \sigma\left(\sum_{i,j,k} X_{n,i,h \cdot s+j,w \cdot s+k} \cdot W_{c,i,j,k} + b_c\right)$$

**内存访问模式对比**:
```
优化前 (两次内存访问):
1. Conv: X,W → Intermediate (写入内存)
2. ReLU: Intermediate → Output (从内存读取)

内存布局:
┌─────────────┬─────────────┬─────────────┐
│   Input     │ Intermediate│   Output    │
│  (Cache)    │  (Memory)   │  (Cache)    │
└─────────────┴─────────────┴─────────────┘

优化后 (一次内存访问):
1. ConvReLU: X,W → Output (直接写入)

内存布局:
┌─────────────┬─────────────┐
│   Input     │   Output    │
│  (Cache)    │  (Cache)    │
└─────────────┴─────────────┘
```

**CUDA kernel融合示例**:
```cpp
__global__ void conv_relu_fused_kernel(
    const float* input, const float* weights, const float* bias,
    float* output, int N, int C_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C_out * H_out * W_out) {
        // 计算4D索引
        int n = idx / (C_out * H_out * W_out);
        int c = (idx / (H_out * W_out)) % C_out;
        int h = (idx / W_out) % H_out;
        int w = idx % W_out;

        // 执行卷积计算
        float conv_result = 0.0f;
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int ih = h * stride + kh;
                    int iw = w * stride + kw;
                    if (ih < H_in && iw < W_in) {
                        conv_result += input[n*C_in*H_in*W_in + ic*H_in*W_in + ih*W_in + iw] *
                                      weights[c*C_in*kernel_h*kernel_w + ic*kernel_h*kernel_w + kh*kernel_w + kw];
                    }
                }
            }
        }

        // 添加偏置并应用ReLU (融合在一个kernel中)
        conv_result += bias[c];
        output[idx] = fmaxf(0.0f, conv_result);  // ReLU
    }
}
```

**性能提升量化**:
- **内存带宽减少**: $\frac{1}{2}$ (避免中间张量存储)
- **kernel启动开销**: 减少50% (2个kernel → 1个kernel)
- **缓存命中率**: 提升20-40%

**支持的激活函数**:
```cpp
enum ActivationType {
    RELU,      // max(0, x)
    RELU6,     // min(6, max(0, x))
    SIGMOID,   // 1 / (1 + exp(-x))
    TANH,      // tanh(x)
    SWISH,     // x * sigmoid(x)
    GELU       // x * Φ(x)
};
```

---

#### **2. MatMulAddFusion (矩阵乘法加法融合)**

**数学定义**: 矩阵乘法后跟元素级加法的融合：
$$\text{MatMulAdd}(A, B, C) = A \times B + C$$

其中：
- $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$ 是输入矩阵
- $C \in \mathbb{R}^{m \times n}$ 是偏置矩阵（或可广播的向量）

**理论原理**:
1. **BLAS优化**: 利用GEMM库的内置bias支持（GEMM with bias）
2. **内存带宽**: 避免MatMul结果的临时存储
3. **数值稳定性**: 减少浮点运算的累积误差

**GEMM操作扩展**:
标准GEMM: $C = \alpha A B + \beta C$
融合版本: $C = A B + C$ (其中 $\alpha=1, \beta=1$)

**广播语义处理**:
```
Case 1: 矩阵偏置
A(m×k) × B(k×n) + C(m×n) = Output(m×n)

Case 2: 行向量广播
A(m×k) × B(k×n) + C(1×n) = Output(m×n)
等价于: result[i,j] = (A×B)[i,j] + C[0,j]

Case 3: 列向量广播
A(m×k) × B(k×n) + C(m×1) = Output(m×n)
等价于: result[i,j] = (A×B)[i,j] + C[i,0]

Case 4: 标量广播
A(m×k) × B(k×n) + C(1×1) = Output(m×n)
等价于: result[i,j] = (A×B)[i,j] + C[0,0]
```

**高性能实现**:
```cpp
// 使用MKL BLAS优化
void matmul_add_fused(
    const float* A, const float* B, const float* C,
    float* output, int m, int k, int n
) {
    // 方法1: 直接使用GEMM with bias
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0f,           // alpha = 1
                A, k,           // A matrix
                B, n,           // B matrix
                0.0f,           // beta = 0 (不累加到output)
                output, n);     // output = A×B

    // 方法2: 利用GEMM的bias支持
    // output = 1.0 * A×B + 1.0 * C
    std::copy(C, C + m*n, output);  // 先复制C到output
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0f, A, k, B, n,
                1.0f, output, n);  // output = A×B + output(=C)
}
```

**性能优势**:
1. **内存访问**: $O(mk + kn + mn)$ → $O(mk + kn)$ (消除临时存储)
2. **计算密度**: 提高FLOPS与内存访问比率
3. **并行化**: 更好的线程级并行性

---

#### **3. AttentionFusion (注意力机制融合)**

**数学定义**: 多头注意力机制的完整计算：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$ (queries)
- $K \in \mathbb{R}^{n \times d_k}$ (keys)
- $V \in \mathbb{R}^{n \times d_v}$ (values)
- $d_k$ 是key维度，用于缩放

**多头注意力**:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**融合前的计算步骤**:
```
步骤1: Q×K^T           → scores(n×n)    [MatMul]
步骤2: scores/√d_k     → scaled(n×n)    [Div/Mul]
步骤3: softmax(scaled) → weights(n×n)   [Softmax]
步骤4: weights×V       → output(n×d_v)  [MatMul]
```

**融合后的优化**:
```
单个Attention Kernel:
Input: Q, K, V → Output: Attention(Q,K,V)

优化策略:
1. Tiled computation (分块计算)
2. Shared memory optimization
3. Warp-level primitives
4. Memory coalescing
```

**Flash Attention算法**:
基于分块计算减少内存复杂度从 $O(n^2)$ 到 $O(n)$：

```python
def flash_attention(Q, K, V, block_size):
    """Flash Attention的简化实现"""
    n, d = Q.shape
    output = torch.zeros_like(Q)

    # 外层循环：处理K,V的块
    for j in range(0, n, block_size):
        K_j = K[j:j+block_size]  # (block_size, d)
        V_j = V[j:j+block_size]  # (block_size, d)

        # 内层循环：处理Q的块
        for i in range(0, n, block_size):
            Q_i = Q[i:i+block_size]  # (block_size, d)

            # 计算注意力分数 (只在内存中保留当前块)
            scores_ij = Q_i @ K_j.T / math.sqrt(d)  # (block_size, block_size)
            attn_ij = F.softmax(scores_ij, dim=-1)

            # 累积输出 (关键：增量更新)
            output[i:i+block_size] += attn_ij @ V_j

    return output
```

**数学推导 - 分块Softmax**:
原始softmax: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

分块计算时需要处理全局归一化：
$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}} \cdot e^{m - m_{\text{global}}}$$

其中 $m$ 是局部最大值，$m_{\text{global}}$ 是全局最大值。

**CUDA实现示例**:
```cpp
__global__ void fused_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* output, int n, int d, int block_size
) {
    extern __shared__ float shared_mem[];
    float* shared_K = shared_mem;
    float* shared_V = shared_mem + block_size * d;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // 加载K,V块到共享内存
    for (int i = tid; i < block_size * d; i += blockDim.x) {
        shared_K[i] = K[bid * block_size * d + i];
        shared_V[i] = V[bid * block_size * d + i];
    }
    __syncthreads();

    // 计算Q×K^T和后续操作
    for (int q_idx = tid; q_idx < block_size; q_idx += blockDim.x) {
        float max_score = -INFINITY;
        float sum_exp = 0.0f;

        // 计算attention scores
        for (int k_idx = 0; k_idx < block_size; k_idx++) {
            float score = 0.0f;
            for (int d_idx = 0; d_idx < d; d_idx++) {
                score += Q[q_idx * d + d_idx] * shared_K[k_idx * d + d_idx];
            }
            score /= sqrtf((float)d);  // scaling
            max_score = fmaxf(max_score, score);
        }

        // 计算softmax (数值稳定版本)
        for (int k_idx = 0; k_idx < block_size; k_idx++) {
            float score = /* 重新计算score */;
            float exp_score = expf(score - max_score);
            sum_exp += exp_score;

            // 累积V的加权和
            for (int d_idx = 0; d_idx < d; d_idx++) {
                atomicAdd(&output[q_idx * d + d_idx],
                         exp_score * shared_V[k_idx * d + d_idx]);
            }
        }

        // 归一化
        for (int d_idx = 0; d_idx < d; d_idx++) {
            output[q_idx * d + d_idx] /= sum_exp;
        }
    }
}
```

**性能收益**:
- **内存减少**: $O(n^2) \rightarrow O(n \cdot \text{block\_size})$
- **kernel融合**: 4个分离kernel → 1个融合kernel
- **数值稳定**: 在线softmax计算，避免大数值
- **吞吐量**: 10-100x提升（取决于序列长度）

**原理**: 将多头注意力机制的多个操作融合为单个高效实现。

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

**标准Attention计算**:
```
Q = input × W_q
K = input × W_k
V = input × W_v
Attention = Softmax(Q × K^T / √d_k) × V
```

**优化前的图结构**:
```
Input
├─→ MatMul(W_q) → Q ──┐
├─→ MatMul(W_k) → K ──┤
└─→ MatMul(W_v) → V ──┤
                      ├─→ MatMul → Div(√d_k) → Softmax → MatMul → Output
                      │         ↗
                      └─────────┘
```

**优化后**:
```
Input → MultiHeadAttention(W_q, W_k, W_v, d_k) → Output
```

**融合优势**:
1. **内存优化**: 减少中间张量存储
2. **计算优化**: 使用优化的GEMM kernel
3. **并行优化**: 多头并行计算

**性能提升**: 2-4x 加速，特别在Transformer模型中效果显著。

##### **EmbedLayerNormFusion (嵌入层归一化融合)**

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

##### **GroupQueryAttentionFusion (分组查询注意力融合)**

**支持Provider**: CUDA专用

**原理**: 针对GPT等模型的分组查询注意力机制优化。

---

#### **6. 归一化优化集群 (3个)**

##### **LayerNormFusion (LayerNorm融合 - Level 2版)**

**原理**: Level 2版本的LayerNorm融合，支持更多Provider和更复杂的模式。

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

**LayerNorm公式**: `LN(x) = γ × (x - mean(x)) / sqrt(var(x) + ε) + β`

**优化前**:
```
Input → ReduceMean → Sub → Pow(2) → ReduceMean → Add(ε) → Sqrt → Div → Mul(γ) → Add(β) → Output
```

**优化后**:
```
Input → LayerNorm(γ, β, ε) → Output
```

**图解**:
```
原始LayerNorm计算步骤:
Input[B, S, D]
    ↓
1. mean = ReduceMean(input, axis=-1)      [B, S, 1]
    ↓
2. variance = ReduceMean((input-mean)², axis=-1)  [B, S, 1]
    ↓
3. std = sqrt(variance + epsilon)
    ↓
4. normalized = (input - mean) / std
    ↓
5. output = normalized * gamma + beta
    ↓
Output[B, S, D]

融合后一步完成:
Input[B, S, D] → LayerNorm_fused → Output[B, S, D]
```

**收益**: 减少70%的内存访问，提升40-60%性能。

##### **SimplifiedLayerNormFusion (简化LayerNorm融合)**

**支持Provider**: CPU, CUDA, ROCm

##### **SkipLayerNormFusion (跳跃LayerNorm融合)**

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

**原理**: 融合残差连接和LayerNorm。

**优化前**:
```
Input → Process → Add(residual) → LayerNorm → Output
   └─────────────────┘
```

**优化后**:
```
Input → SkipLayerNorm → Output
```

---

#### **7. 激活函数优化集群 (5个)**

##### **GeluFusion (GELU融合 - Level 2版)**

**原理**: Level 2版本的GELU融合，支持更多Provider。

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

**GELU公式**: `GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))`

**优化前**:
```
Input → Pow(3) → Mul(0.044715) → Add(x) → Mul(√(2/π)) → Tanh → Add(1) → Mul(0.5) → Mul(x) → Output
```

**优化后**:
```
Input → GELU → Output
```

**性能提升**: 将8-10个算子操作融合为1个，减少约80%的计算时间。

##### **FastGeluFusion (Fast GELU融合)**

**原理**: 使用GELU的快速近似版本进一步优化性能。

**支持Provider**: CPU, CUDA, DML, ROCm

**Fast GELU公式**: `FastGELU(x) = 0.5 × x × (1 + tanh(x × 0.7978845608 × (1 + 0.044715 × x²)))`

**对比**:
- 标准GELU: 高精度，慢速
- Fast GELU: 近似计算，快速
- 精度损失: < 0.1%，性能提升: 30-50%

##### **QuickGeluFusion (Quick GELU融合)**

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

##### **BiasGeluFusion (Bias+GELU融合)**

**支持Provider**: CPU, ACL, CUDA, DML, ROCm

##### **GeluApproximation (GELU近似)**

**原理**: 可选的GELU近似优化，需要手动启用。

**注意**: 有副作用，可能改变结果，需要手动启用。

---

#### **8. 专用融合优化集群 (7个)**

##### **GatherSliceToSplitFusion (Gather-Slice到Split融合)**

**支持Provider**: CPU, CUDA, ROCm

##### **GatherToSliceFusion (Gather到Slice融合)**

**支持Provider**: CPU, CUDA, ROCm

##### **BiasSoftmaxFusion (Bias+Softmax融合)**

**支持Provider**: CPU, CUDA, ROCm

##### **BiasDropoutFusion (Bias+Dropout融合)**

**支持Provider**: CUDA, ROCm

##### **BitmaskDropoutReplacement (位掩码Dropout替换)**

**支持Provider**: CUDA, ROCm

**原理**: 训练时的Dropout优化。

##### **BiasSoftmaxDropoutFusion (Bias+Softmax+Dropout融合)**

**支持Provider**: CUDA, ROCm

**原理**: 训练模式下的三重融合。

##### **SceLossGradBiasFusion (SCE损失梯度Bias融合)**

**支持Provider**: CPU, CUDA, ROCm

**原理**: 训练时的损失函数优化。

---

#### **9. 硬件特定优化 (4个)**

##### **Avx2WeightS8ToU8Transformer (AVX2权重S8到U8转换)**

**支持硬件**: AMD64/IX86 + AVX2

**原理**: 在AVX2精度模式下将S8权重转换为U8。

##### **MatMulNBitsFusion (MatMul N位融合)**

**支持Provider**: CPU

**原理**: 支持低精度（如4位、2位）矩阵乘法的融合优化。

##### **TritonFusion (Triton融合)**

**支持Provider**: CUDA (需要Triton支持)

**原理**: 使用Triton编译器进行自定义kernel融合。

---

#### **10. 特殊功能优化 (5个)**

##### **MatMulAddFusion (MatMul+Add融合 - Level 2版)**

**原理**: Level 2版本的MatMul+Add融合，支持更复杂的模式。

**配置**: `preserve_attention_pattern = false`

**说明**: 在Attention融合后清理剩余的MatMul-Add模式。

##### **QDQFinalCleanupTransformer (QDQ最终清理)**

**原理**: 必须在其他QDQ融合优化之后运行的清理优化器。

**重要性**: 防止过早移除Q/DQ节点导致其他融合失败。

---

### **Level 2 优化策略总结**

**优化分类统计**:
- **RewriteRule**: 3个
- **量化优化**: 8个
- **矩阵运算**: 6个
- **卷积优化**: 3个
- **注意力机制**: 4个
- **归一化**: 3个
- **激活函数**: 5个
- **专用融合**: 7个
- **硬件特定**: 4个
- **特殊功能**: 5个
- **其他**: 12个

**总计**: 约25个Level 2专用优化器

**关键特点**:
1. **多Provider支持**: 大多数优化器支持多个执行提供者
2. **复杂融合**: 支持3个或更多算子的融合
3. **硬件感知**: 针对特定硬件的优化
4. **训练友好**: 包含训练时的专用优化

---

### 1. **卷积相关融合**

#### **ConvBNFusion (Conv+BatchNorm融合)**

**原理**: 将卷积和批归一化融合为单个操作，这是最重要的融合优化之一。

**数学原理**:
BatchNorm公式: `BN(x) = γ × (x - μ) / σ + β`

融合后的卷积: `Conv_fused(x) = Conv(x) × γ/σ + (β - μ×γ/σ)`

**优化前**:
```
Input → Conv(W, b) → BatchNorm(γ, β, μ, σ) → Output
```

**优化后**:
```
Input → Conv(W_new, b_new) → Output
```

其中:
- `W_new = W × γ/σ`
- `b_new = b × γ/σ + (β - μ×γ/σ)`

**图解**:
```
优化前的计算流程:
Input[N,C,H,W]
    ↓
Conv: Y = W * X + b
    ↓
BN: Z = γ × (Y - μ) / σ + β
    ↓
Output[N,C,H,W]

优化后的计算流程:
Input[N,C,H,W]
    ↓
Conv_fused: Z = W_new * X + b_new
    ↓
Output[N,C,H,W]
```

**收益**: 减少50%的计算和内存访问。

---

#### **ConvActivationFusion (Conv+Activation融合)**

**原理**: 将卷积和激活函数融合，避免中间结果的存储。

**支持的激活函数**: ReLU, ReLU6, Sigmoid, Tanh, Swish等

**优化前**:
```
Input → Conv → ReLU → Output
```

**优化后**:
```
Input → ConvReLU → Output
```

**实现细节**:
```cpp
// 融合后的伪代码
for (int i = 0; i < output_size; i++) {
    float conv_result = convolution(input, weights, i);
    output[i] = max(0.0f, conv_result);  // ReLU直接应用
}
```

---

### 2. **激活函数融合**

#### **GeluFusion (GELU融合)**

**原理**: GELU激活函数通常由多个基础算子组成，融合可以显著提升性能。

**GELU公式**: `GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))`

**优化前**:
```
Input → Pow(3) → Mul(0.044715) → Add(x) → Mul(√(2/π)) → Tanh → Add(1) → Mul(0.5) → Mul(x) → Output
```

**优化后**:
```
Input → GELU → Output
```

**性能提升**: 将8-10个算子操作融合为1个，减少约80%的计算时间。

---

#### **FastGeluFusion (Fast GELU融合)**

**原理**: 使用GELU的快速近似版本进一步优化性能。

**Fast GELU公式**: `FastGELU(x) = 0.5 × x × (1 + tanh(x × 0.7978845608 × (1 + 0.044715 × x²)))`

**对比**:
- 标准GELU: 高精度，慢速
- Fast GELU: 近似计算，快速
- 精度损失: < 0.1%，性能提升: 30-50%

---

### 3. **归一化融合**

#### **LayerNormFusion (LayerNorm融合)**

**原理**: Layer Normalization通常包含多个算子，融合可以减少内存访问。

**LayerNorm公式**: `LN(x) = γ × (x - mean(x)) / sqrt(var(x) + ε) + β`

**优化前**:
```
Input → ReduceMean → Sub → Pow(2) → ReduceMean → Add(ε) → Sqrt → Div → Mul(γ) → Add(β) → Output
```

**优化后**:
```
Input → LayerNorm(γ, β, ε) → Output
```

**图解**:
```
原始LayerNorm计算步骤:
Input[B, S, D]
    ↓
1. mean = ReduceMean(input, axis=-1)      [B, S, 1]
    ↓
2. variance = ReduceMean((input-mean)², axis=-1)  [B, S, 1]
    ↓
3. std = sqrt(variance + epsilon)
    ↓
4. normalized = (input - mean) / std
    ↓
5. output = normalized * gamma + beta
    ↓
Output[B, S, D]

融合后一步完成:
Input[B, S, D] → LayerNorm_fused → Output[B, S, D]
```

**收益**: 减少70%的内存访问，提升40-60%性能。

---

### 4. **注意力机制融合**

#### **AttentionFusion (Attention融合)**

**原理**: 将多头注意力机制的多个操作融合为单个高效实现。

**标准Attention计算**:
```
Q = input × W_q
K = input × W_k
V = input × W_v
Attention = Softmax(Q × K^T / √d_k) × V
```

**优化前的图结构**:
```
Input
├─→ MatMul(W_q) → Q ──┐
├─→ MatMul(W_k) → K ──┤
└─→ MatMul(W_v) → V ──┤
                      ├─→ MatMul → Div(√d_k) → Softmax → MatMul → Output
                      │         ↗
                      └─────────┘
```

**优化后**:
```
Input → MultiHeadAttention(W_q, W_k, W_v, d_k) → Output
```

**融合优势**:
1. **内存优化**: 减少中间张量存储
2. **计算优化**: 使用优化的GEMM kernel
3. **并行优化**: 多头并行计算

**性能提升**: 2-4x 加速，特别在Transformer模型中效果显著。

---

### 5. **RNN/LSTM融合**

#### **LSTMFusion (LSTM融合)**

**原理**: 将LSTM单元的多个门控操作融合为单个kernel。

**LSTM公式**:
```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)  # 遗忘门
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)  # 输入门
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)  # 输出门
C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)  # 候选值
C_t = f_t * C_{t-1} + i_t * C̃_t      # 细胞状态
h_t = o_t * tanh(C_t)                # 隐藏状态
```

**优化前**: 每个门和状态更新都是独立的算子
**优化后**: 整个LSTM单元作为单个融合算子

**图解**:
```
优化前 - 分离的门控计算:
[h_{t-1}, x_t] ─┬─→ MatMul(W_f) ─→ Sigmoid ─→ f_t ─┐
                ├─→ MatMul(W_i) ─→ Sigmoid ─→ i_t ─┤
                ├─→ MatMul(W_o) ─→ Sigmoid ─→ o_t ─┤
                └─→ MatMul(W_C) ─→ Tanh ───→ C̃_t ─┘
                                                   ↓
                                            LSTM状态更新
                                                   ↓
                                            [h_t, C_t]

优化后 - 融合LSTM单元:
[h_{t-1}, x_t] ─→ LSTMCell(W_f,W_i,W_o,W_C) ─→ [h_t, C_t]
```

**收益**: 减少60-80%的内存带宽，提升2-3x性能。

---

### 6. **量化相关优化**

#### **QLinearMatMulFusion (量化MatMul融合)**

**原理**: 融合量化、矩阵乘法和反量化操作。

**量化MatMul流程**:
```
Input(int8) → Dequantize → MatMul(fp32) → Quantize → Output(int8)
```

**融合后**:
```
Input(int8) → QLinearMatMul → Output(int8)
```

**数学实现**:
```cpp
// 融合后直接在int8域进行计算
int32 result = int8_matmul(input_int8, weight_int8);
int8 output = quantize(result, scale, zero_point);
```

**收益**: 避免量化/反量化开销，保持int8高效计算。

---

#### **DoubleQDQPairsRemover (双重量化对消除)**

**原理**: 移除连续的量化-反量化操作对。

**优化前**:
```
Input(fp32) → Quantize → Dequantize → Quantize → Dequantize → Output(fp32)
```

**优化后**:
```
Input(fp32) → Output(fp32)
```

**应用场景**: 清理量化模型中的冗余QDQ操作。

---

## Level 3: 布局优化 (ORT_ENABLE_LAYOUT)

### **3个布局优化算法**

Level 3在Level 2基础上新增了3个布局优化算法，总计68个优化器，专门处理数据布局变换和硬件特定的布局优化。

**源码定义**: `onnxruntime/core/optimizer/graph_transformer_utils.cc:GenerateTransformers(TransformerLevel::Level3)`

---

#### **Level 3 新增GraphTransformer (3个)**

**1. NchwcTransformer** (`nchwc_transformer.h`)
```cpp
case TransformerLevel::Level3: {
#ifndef DISABLE_CONTRIB_OPS
    // Register the NCHWc layout transformer if supported by the platform.
    if (MlasNchwcGetBlockSize() > 1) {
        transformers.emplace_back(std::make_unique<NchwcTransformer>());
    }
}
```
- **作用**: NCHW到NCHWc布局转换
- **原理**: 将数据重排为分块通道格式以优化SIMD性能
- **条件**: 仅在支持的平台上启用（Intel x86-64）
- **块大小**: 由`MlasNchwcGetBlockSize()`决定

**2. NhwcTransformer** (`nhwc_transformer.h`)
```cpp
auto cpu_registry = cpu_execution_provider.GetKernelRegistry();
auto nhwc_transformer = std::make_unique<NhwcTransformer>(std::move(cpu_allocator), std::move(cpu_registry), logger);
if (nhwc_transformer->IsActive()) {
    transformers.emplace_back(std::move(nhwc_transformer));
}
```
- **作用**: NHWC布局优化器
- **原理**: 为支持NHWC的硬件（如ARM、某些GPU）优化数据布局
- **条件**: 仅在优化器检测到可以提升性能时激活
- **检查**: 通过`IsActive()`判断是否应该应用

**3. ConvAddActivationFusion** (`conv_add_act_fusion.h`) - CPU特定
```cpp
// NchwcTransformer must have a higher priority than ConvAddActivationFusion
transformers.emplace_back(std::make_unique<ConvAddActivationFusion>(cpu_ep));
```
- **作用**: 卷积+加法+激活融合（CPU特定版本）
- **原理**: 针对CPU EP的三元融合，与NCHWc配合使用
- **优先级**: 低于NchwcTransformer，确保布局优化优先执行
- **限制**: 仅在CPU EP上应用

---

### **Level 3 布局优化原理**

**数学定义**:

NCHWc布局变换：
$$\text{NCHWc}[n, c', h, w, c] = \text{NCHW}[n, c' \times \text{block\_size} + c, h, w]$$

NHWC布局变换：
$$\text{NHWC}[n, h, w, c] = \text{NCHW}[n, c, h, w]$$

**性能优势**:
- **NCHWc**: 适合x86 SIMD指令，提升向量化效率
- **NHWC**: 适合ARM NEON和某些GPU架构
- **融合**: 布局转换与算子融合减少内存拷贝

**应用策略**:
1. **硬件检测**: 根据目标硬件选择最优布局
2. **成本分析**: 评估布局转换的开销vs收益
3. **优先级**: 布局优化优先于算子融合
4. **条件应用**: 只在确认有性能提升时应用

---

### **Level 3 优化效果**

- **性能提升**: 5-20%（在Level 2基础上，硬件相关）
- **稳定性**: 中等，依赖硬件支持
- **兼容性**: 特定硬件平台（x86-64、ARM等）
- **特点**: 硬件特定的数据布局优化

**向量化优势分析**:
```
标准NCHW卷积 (逐元素):
for (int c_out = 0; c_out < C_out; c_out++) {
    for (int c_in = 0; c_in < C_in; c_in++) {
        output[c_out] += input[c_in] * weight[c_out][c_in];  // 标量运算
    }
}

NCHWc向量化卷积:
for (int c_out_group = 0; c_out_group < C_out/4; c_out_group++) {
    for (int c_in_group = 0; c_in_group < C_in/4; c_in_group++) {
        __m128 input_vec = _mm_load_ps(&input[c_in_group * 4]);     // 加载4个float
        __m128 weight_vec = _mm_load_ps(&weight[c_out_group][c_in_group * 4]);
        __m128 result = _mm_mul_ps(input_vec, weight_vec);          // 4个并行乘法
        output_vec[c_out_group] = _mm_add_ps(output_vec[c_out_group], result);
    }
}
```

**性能提升量化**:
```
Intel CPU性能数据 (AVX2):
- 标量吞吐量: 1 float/cycle
- 向量吞吐量: 8 float/cycle (256-bit SIMD)
- 理论加速比: 8x
- 实际加速比: 3-5x (考虑内存带宽和其他开销)
```

**内存访问模式对比**:
```
NCHW内存布局:
Channel 0: [pixel_0, pixel_1, ..., pixel_HW-1]
Channel 1: [pixel_0, pixel_1, ..., pixel_HW-1]
...
Channel C-1: [pixel_0, pixel_1, ..., pixel_HW-1]

NCHW访问模式 (非连续):
Step 1: 读取 C0_P0, C1_P0, C2_P0, C3_P0  ← 4次缓存失效
Step 2: 读取 C0_P1, C1_P1, C2_P1, C3_P1  ← 4次缓存失效

NCHWc内存布局:
Block 0: [C0_P0, C1_P0, C2_P0, C3_P0, C0_P1, C1_P1, C2_P1, C3_P1, ...]
Block 1: [C4_P0, C5_P0, C6_P0, C7_P0, C4_P1, C5_P1, C6_P1, C7_P1, ...]

NCHWc访问模式 (连续):
Step 1: 读取 [C0_P0, C1_P0, C2_P0, C3_P0]  ← 1次缓存加载
Step 2: 读取 [C0_P1, C1_P1, C2_P1, C3_P1]  ← 已在缓存中
```

**支持条件检测**:
```cpp
bool can_apply_nchwc_transform() {
    // 检查硬件支持
    if (!is_x86_64_architecture()) return false;

    // 检查SIMD支持
    int block_size = MlasNchwcGetBlockSize();
    if (block_size <= 1) return false;

    // 检查通道数对齐
    if (channel_count % block_size != 0) return false;

    // 检查算子支持
    return supports_nchwc_layout(current_operator);
}
```

---

### **2. NHWC变换器 (ARM/移动端优化)**

#### **NhwcTransformer (NHWC布局变换器)**

**数学定义**: NHWC布局重排数据维度顺序：
$$\text{NHWC}[n, h, w, c] = \text{NCHW}[n, c, h, w]$$

**理论原理**:
1. **ARM NEON优化**: ARM NEON指令集更适合通道交错的数据布局
2. **移动GPU友好**: 移动GPU纹理格式通常使用NHWC
3. **量化友好**: INT8量化在NHWC布局下更高效

**ARM NEON向量化示例**:
```cpp
// NCHW卷积 (需要gather操作)
float32x4_t input_nchw[4];
for (int c = 0; c < 4; c++) {
    input_nchw[c] = vld1q_f32(&input[c * H * W + h * W + w]);  // 分散加载
}

// NHWC卷积 (连续加载)
float32x4_t input_nhwc = vld1q_f32(&input[h * W * C + w * C]);  // 连续加载4个通道
```

**布局变换的计算复杂度**:
```
变换时间复杂度: O(N × H × W × C)
变换空间复杂度: O(N × H × W × C) [临时存储]

但是变换开销可以被后续计算的加速抵消:
转换开销: T_transform
计算加速: α × T_compute (其中α > 1)
净收益: α × T_compute - T_transform
```

**量化优化优势**:
```
NCHW INT8量化:
需要逐通道gather: [C0_val, C1_val, C2_val, C3_val]
每个值来自不同内存页面

NHWC INT8量化:
自然4字节对齐: [C0_val C1_val C2_val C3_val] 作为uint32
单次内存访问获得4个通道数据
```

**移动设备性能数据**:
```
ARM Cortex-A78 (NEON):
NCHW Conv2D: 45.2 ms
NHWC Conv2D: 28.7 ms
加速比: 1.57x

ARM Mali GPU:
NCHW Texture: 需要额外重排
NHWC Texture: 原生支持
性能提升: 20-40%
```

---

### **3. Conv+Add+Activation三重融合 (NCHWc布局专用)**

#### **ConvAddActivationFusion (卷积+加法+激活融合)**

**数学定义**: 三算子融合在NCHWc布局下的优化：
$$\text{ConvAddReLU}_{\text{NCHWc}}(X, W, B) = \text{ReLU}(\text{Conv}_{\text{NCHWc}}(X, W) + B)$$

**理论原理**: NCHWc布局使得三个操作可以在同一向量化循环中完成。

**融合算法 (AVX2实现)**:
```cpp
void conv_add_relu_nchwc_avx2(
    const float* input,   // NCHWc format
    const float* weights, // NCHWc format
    const float* bias,
    float* output,
    int N, int C_groups, int H, int W, int block_size
) {
    __m256 zero = _mm256_setzero_ps();

    for (int n = 0; n < N; n++) {
        for (int c_group = 0; c_group < C_groups; c_group++) {
            __m256 bias_vec = _mm256_load_ps(&bias[c_group * block_size]);

            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    // 卷积计算 (向量化)
                    __m256 conv_result = _mm256_setzero_ps();

                    // 卷积核循环...
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            __m256 input_vec = _mm256_load_ps(
                                &input[/* NCHWc索引 */]);
                            __m256 weight_vec = _mm256_load_ps(
                                &weights[/* NCHWc索引 */]);
                            conv_result = _mm256_fmadd_ps(
                                input_vec, weight_vec, conv_result);
                        }
                    }

                    // 融合: Add + ReLU (单个向量操作)
                    conv_result = _mm256_add_ps(conv_result, bias_vec);
                    conv_result = _mm256_max_ps(conv_result, zero);  // ReLU

                    // 存储结果
                    _mm256_store_ps(&output[/* NCHWc索引 */], conv_result);
                }
            }
        }
    }
}
```

**三重融合的性能优势**:
```
优化前 (3个分离kernel):
1. Conv kernel:    时间T₁, 内存带宽B₁
2. Add kernel:     时间T₂, 内存带宽B₂
3. ReLU kernel:    时间T₃, 内存带宽B₃
总时间: T₁ + T₂ + T₃
总带宽: B₁ + B₂ + B₃

优化后 (1个融合kernel):
1. ConvAddReLU:    时间T_fused, 内存带宽B_fused
理论提升: T_fused ≈ T₁, B_fused ≈ B₁
实际加速比: 2.5-3.5x
```

**融合条件检测**:
```cpp
bool can_fuse_conv_add_relu_nchwc(Graph& graph, Node& conv_node) {
    // 检查布局兼容性
    if (!is_nchwc_layout(conv_node)) return false;

    // 检查Add节点
    Node* add_node = find_single_consumer(conv_node, "Add");
    if (!add_node || !is_bias_add(add_node)) return false;

    // 检查ReLU节点
    Node* relu_node = find_single_consumer(add_node, "Relu");
    if (!relu_node) return false;

    // 检查没有其他消费者
    return (conv_node.consumers.size() == 1 &&
            add_node->consumers.size() == 1);
}
```

**Level 3的关键洞察**: 布局变换必须与算子融合协同优化。NCHWc变换为后续的三重融合创造了向量化机会，这是Level 3优化的核心价值。

---
