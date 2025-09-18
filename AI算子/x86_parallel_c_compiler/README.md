# X86/X64 并行 C 编译器

一个专门针对 x86/x64 指令集架构设计的并行 C 语言编译器，支持核心 C 语法和先进的并行计算特性。

## 项目特性

### 🚀 核心功能
- **完整的 C 语言支持**: 变量声明、函数定义、控制流、指针操作
- **x86/x64 目标架构**: 生成原生 x86_64 汇编代码
- **System V ABI 兼容**: 标准函数调用约定
- **多级优化**: O0-O3 优化等级支持

### ⚡ 并行计算特性
- **parallel_for 循环**: 自动并行化的 for 循环
- **原子操作**: atomic 变量和原子指令
- **内存屏障**: memory_barrier() 确保内存一致性
- **临界区**: critical 块提供互斥访问
- **向量化**: SSE/AVX 指令自动生成

### 🔧 高级特性
- **寄存器分配**: 智能的 x86/x64 寄存器管理
- **指令调度**: 优化的指令序列生成
- **栈帧管理**: 自动的函数调用栈处理
- **符号表**: 完整的作用域和类型检查

## 项目结构

```
x86_parallel_c_compiler/
├── src/                    # 源代码目录
│   ├── x86_cc.h           # 主头文件（寄存器、指令定义）
│   ├── lexer.c            # 词法分析器
│   ├── parser.c           # 语法分析器
│   ├── semantic.c         # 语义分析器
│   ├── codegen.c          # x86/x64 代码生成器
│   ├── utils.c            # 工具函数
│   └── main.c             # 主程序入口
├── examples/               # 示例程序
│   ├── basic.c            # 基础 C 语言示例
│   └── parallel.c         # 并行计算示例
├── tests/                  # 测试脚本
│   └── run_tests.bat      # Windows 测试脚本
├── Makefile               # Linux/macOS 构建文件
├── build.bat              # Windows 构建脚本
└── README.md              # 项目文档
```

## 快速开始

### Windows 环境

1. **安装依赖**
   ```cmd
   # 确保已安装 GCC (MinGW)
   gcc --version
   ```

2. **构建编译器**
   ```cmd
   build.bat
   ```

3. **编译示例程序**
   ```cmd
   bin\x86cc.exe examples\basic.c -o basic.s
   bin\x86cc.exe -fopenmp examples\parallel.c -o parallel.s
   ```

4. **运行测试**
   ```cmd
   cd tests
   run_tests.bat
   ```

### Linux/macOS 环境

1. **构建编译器**
   ```bash
   make
   ```

2. **编译示例程序**
   ```bash
   ./bin/x86cc examples/basic.c -o basic.s
   ./bin/x86cc -fopenmp examples/parallel.c -o parallel.s
   ```

3. **运行测试**
   ```bash
   make test
   ```

## 使用指南

### 命令行选项

```
x86cc [选项] <输入文件>

选项:
  -o <文件>     指定输出文件
  -v            详细输出
  -h            显示帮助信息
  -O<级别>      优化等级 (0-3)
  -g            生成调试信息
  -fopenmp      启用并行处理
  -fvectorize   启用向量化
```

### 基本 C 语言示例

```c
// 变量声明和初始化
int a = 10;
float pi = 3.14159;

// 函数定义
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// 控制流
for (int i = 0; i < 10; i++) {
    if (i % 2 == 0) {
        printf("Even: %d\n", i);
    }
}

// 指针操作
int *ptr = &a;
*ptr = 20;
```

### 并行计算示例

```c
// 并行 for 循环
parallel_for (int i = 0; i < 1000; i++) {
    array[i] = compute_value(i);
}

// 原子操作
atomic int counter = 0;
atomic_inc(&counter);
atomic_add(&counter, 5);

// 临界区
critical {
    shared_resource++;
}

// 内存屏障
data = 42;
memory_barrier();
flag = 1;
```

## 架构设计

### 编译流水线

1. **词法分析** (lexer.c)
   - Token 识别和分类
   - 关键字和操作符解析
   - 错误位置跟踪

2. **语法分析** (parser.c)
   - 递归下降解析器
   - AST (抽象语法树) 构建
   - 表达式优先级处理

3. **语义分析** (semantic.c)
   - 符号表管理
   - 类型检查和推导
   - 作用域验证

4. **代码生成** (codegen.c)
   - x86/x64 汇编生成
   - 寄存器分配
   - 优化和指令调度

### x86/x64 特性支持

#### 寄存器管理
- **通用寄存器**: RAX, RBX, RCX, RDX, RSI, RDI, R8-R15
- **SSE 寄存器**: XMM0-XMM15 (128-bit)
- **AVX 寄存器**: YMM0-YMM15 (256-bit)
- **调用约定**: RDI, RSI, RDX, RCX, R8, R9 传参

#### 指令集支持
- **算术指令**: ADD, SUB, IMUL, IDIV
- **控制流**: JMP, JE, JNE, JL, JG
- **SSE/AVX**: MOVSS, ADDSS, MULSS, ADDPS
- **原子指令**: LOCK, XCHG, CMPXCHG
- **内存操作**: MOV, LEA, PUSH, POP

#### 并行优化
- **指令级并行**: 指令重排和流水线优化
- **向量化**: 自动 SIMD 指令生成
- **原子操作**: LOCK 前缀和原子指令
- **内存一致性**: MFENCE, LFENCE, SFENCE

## 生成的汇编代码示例

### 基础函数
```assembly
factorial:
    pushq   %rbp
    movq    %rsp, %rbp
    movl    %edi, -4(%rbp)
    cmpl    $1, -4(%rbp)
    jg      .L2
    movl    $1, %eax
    jmp     .L3
.L2:
    movl    -4(%rbp), %eax
    subl    $1, %eax
    movl    %eax, %edi
    call    factorial
    imull   -4(%rbp), %eax
.L3:
    popq    %rbp
    ret
```

### 并行循环
```assembly
parallel_loop:
    # 线程创建和分发
    movl    $4, %edi        # 线程数
    call    omp_set_num_threads
    
    # 并行区域
    .section .parallel_for
    movl    %thread_id, %eax
    imull   $250, %eax      # 每线程处理 250 项
    movl    %eax, %esi      # 起始索引
    
.L_parallel_body:
    # 原子操作示例
    lock addl $1, global_counter(%rip)
    # 向量化操作
    movaps  vector_a(%rsi), %xmm0
    addps   vector_b(%rsi), %xmm0
    movaps  %xmm0, result(%rsi)
```

## 性能特性

### 优化等级

- **-O0**: 无优化，保持调试信息
- **-O1**: 基础优化，寄存器分配
- **-O2**: 标准优化，指令调度
- **-O3**: 积极优化，向量化和循环展开

### 并行性能

- **线程级并行**: 支持多核 CPU 并行执行
- **指令级并行**: 利用 CPU 超标量特性
- **数据级并行**: SSE/AVX 向量指令
- **内存优化**: 缓存友好的内存访问模式

## 扩展和自定义

### 添加新的内置函数

```c
// 在 semantic.c 中添加
add_builtin_function(global_scope, "my_func", 
                     create_function_type(INT_TYPE, param_types, 1));
```

### 添加新的并行原语

```c
// 在 x86_cc.h 中添加新的 AST 节点类型
typedef enum {
    // 现有类型...
    AST_MY_PARALLEL_CONSTRUCT,
} ASTNodeType;
```

### 自定义优化 Pass

```c
// 在 codegen.c 中添加优化函数
void my_optimization_pass(ASTNode *node) {
    // 自定义优化逻辑
}
```

## 测试和验证

### 单元测试
```cmd
# 运行所有测试
tests\run_tests.bat

# 检查生成的汇编代码
type output\basic.s
type output\parallel.s
```

### 性能测试
```cmd
# 编译优化版本
bin\x86cc.exe -O3 -fopenmp examples\parallel.c -o optimized.s

# 比较不同优化等级的代码大小
dir output\*.s
```

## 贡献指南

1. **代码风格**: 遵循项目现有的代码风格
2. **测试**: 添加新功能时包含相应测试
3. **文档**: 更新 README 和代码注释
4. **性能**: 考虑新功能对编译性能的影响

## 已知限制

1. **预处理器**: 不支持 #include, #define 等预处理指令
2. **标准库**: 仅支持基本的 printf 等函数声明
3. **复杂类型**: 对结构体和联合体的支持有限
4. **链接**: 生成汇编代码，需要外部汇编器和链接器

## 未来计划

- [ ] 预处理器支持
- [ ] 更多 C 标准库函数
- [ ] ARM64 目标支持
- [ ] LLVM IR 后端
- [ ] 调试信息生成
- [ ] 内联汇编支持

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
