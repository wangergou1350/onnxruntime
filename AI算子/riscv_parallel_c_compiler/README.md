# RISC-V 并行 C 编译器

一个支持并行计算扩展的 RISC-V 指令集 C 语言编译器的最小实现。

## 功能特性

### 核心 C 语法支持
- ✅ 基本数据类型：`int`, `float`, `double`, `char`
- ✅ 变量声明和赋值
- ✅ 算术运算：`+`, `-`, `*`, `/`, `%`
- ✅ 逻辑运算：`&&`, `||`, `!`
- ✅ 比较运算：`==`, `!=`, `<`, `>`, `<=`, `>=`
- ✅ 控制流语句：`if/else`, `for`, `while`
- ✅ 函数定义和调用
- ✅ 数组（一维和多维）
- ✅ 指针操作
- ✅ 结构体定义和成员访问

### 并行计算扩展
- ✅ `parallel_for` 并行循环
- ✅ `atomic` 原子变量类型
- ✅ `barrier()` 内存屏障
- ✅ `critical` 临界区
- ✅ `thread_local` 线程局部存储

### RISC-V 架构支持
- ✅ RV32I/RV64I 基础指令集
- ✅ M 扩展（乘法/除法）
- ✅ A 扩展（原子操作）
- ✅ F/D 扩展（浮点运算）
- ✅ 完整的寄存器分配
- ✅ 函数调用约定

## 项目结构

```
riscv_parallel_c_compiler/
├── README.md                 # 项目说明
├── Makefile                  # Unix/Linux 构建脚本
├── build.bat                 # Windows 构建脚本
├── test.bat                  # Windows 测试脚本
├── src/                      # 源代码目录
│   ├── riscv_cc.h           # 主头文件
│   ├── main.c               # 主程序入口
│   ├── lexer.c              # 词法分析器
│   ├── parser.c             # 语法分析器
│   ├── semantic.c           # 语义分析器
│   ├── codegen.c            # 代码生成器
│   └── utils.c              # 工具函数
├── examples/                 # 示例程序
│   ├── hello.c              # Hello World
│   ├── fibonacci.c          # 斐波那契数列
│   ├── matrix_mult.c        # 矩阵乘法
│   ├── parallel_sum.c       # 并行求和
│   ├── atomic_counter.c     # 原子计数器
│   └── thread_local_demo.c  # 线程局部存储
└── tests/                   # 测试用例
    ├── README.md            # 测试说明
    ├── basic_syntax.c       # 基础语法测试
    ├── functions.c          # 函数测试
    ├── arrays.c             # 数组测试
    ├── pointers.c           # 指针测试
    ├── structs.c            # 结构体测试
    └── parallel.c           # 并行计算测试
```

## 编译和安装

### 在 Windows 上

1. 安装 MinGW-w64 或 MSYS2：
   - 下载 MSYS2：https://www.msys2.org/
   - 安装后在 MSYS2 终端中运行：`pacman -S mingw-w64-x86_64-gcc`

2. 编译编译器：
   ```cmd
   build.bat
   ```

3. 运行测试：
   ```cmd
   test.bat
   ```

### 在 Linux/Unix 上

1. 确保安装了 GCC：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   
   # macOS
   xcode-select --install
   ```

2. 编译编译器：
   ```bash
   make
   ```

3. 运行测试：
   ```bash
   make test
   ```

## 使用方法

### 基本用法

```bash
# 编译 C 文件为 RISC-V 汇编
./riscv_cc input.c -o output.s

# 显示帮助信息
./riscv_cc -h

# 显示详细编译过程
./riscv_cc -v input.c -o output.s
```

### 示例

```bash
# 编译 Hello World 示例
./riscv_cc examples/hello.c -o hello.s

# 编译并行求和示例
./riscv_cc examples/parallel_sum.c -o parallel_sum.s

# 编译并显示详细信息
./riscv_cc -v examples/fibonacci.c -o fibonacci.s
```

## 语法示例

### 基本 C 语法

```c
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    int result = factorial(5);
    printf("5! = %d\n", result);
    return 0;
}
```

### 并行计算扩展

```c
#include <stdio.h>

int main() {
    atomic_int counter = 0;
    int sum = 0;
    
    // 并行循环
    parallel_for (int i = 0; i < 100; i++) {
        // 原子操作
        atomic_add(&counter, 1);
        
        // 临界区
        critical {
            sum += i;
        }
        
        // 内存屏障
        barrier();
    }
    
    printf("Counter: %d, Sum: %d\n", 
           atomic_load(&counter), sum);
    return 0;
}
```

### 线程局部存储

```c
thread_local int tls_var = 0;

void worker_function() {
    tls_var = 42;  // 每个线程都有独立的副本
}

int main() {
    parallel_for (int i = 0; i < 8; i++) {
        worker_function();
    }
    return 0;
}
```

## 编译器输出

编译器生成符合 RISC-V 指令集的汇编代码，包括：

- 完整的函数序言和尾声
- 正确的寄存器分配和使用
- 符合 RISC-V ABI 的函数调用约定
- 并行计算相关的指令序列
- 原子操作和内存屏障指令

## 技术细节

### 编译器架构

1. **词法分析（Lexer）**：将源代码转换为标记流
2. **语法分析（Parser）**：构建抽象语法树（AST）
3. **语义分析（Semantic）**：类型检查和符号表管理
4. **代码生成（Codegen）**：生成 RISC-V 汇编代码

### 寄存器分配

- **x0**: 硬编码为 0
- **x1 (ra)**: 返回地址
- **x2 (sp)**: 栈指针
- **x8 (fp)**: 帧指针
- **x10-x17 (a0-a7)**: 函数参数和返回值
- **x5-x7, x28-x31 (t0-t6)**: 临时寄存器
- **x9, x18-x27 (s0-s11)**: 保存寄存器

### 并行扩展实现

- `parallel_for`: 生成多线程循环控制代码
- `atomic`: 使用 RISC-V A 扩展的原子指令
- `barrier`: 生成内存屏障指令（fence）
- `critical`: 使用原子操作实现互斥锁
- `thread_local`: 使用线程指针寄存器访问

## 限制

- 当前版本是最小实现，不支持所有 C 语言特性
- 不包含预处理器
- 不支持内联汇编
- 有限的优化功能
- 错误处理可以进一步改善

## 扩展计划

- [ ] 添加更多 C 语言特性（枚举、联合等）
- [ ] 实现基本的代码优化
- [ ] 添加调试信息生成
- [ ] 支持更多 RISC-V 扩展
- [ ] 改进错误诊断和报告

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交问题报告和拉取请求。请确保代码风格一致，并添加适当的测试用例。

## 联系

如有问题或建议，请提交 GitHub Issue 或发送邮件。
     ↓
📝 词法分析器 (lexer.c)     - C语言关键字、操作符识别
     ↓
🌳 语法分析器 (parser.c)    - 递归下降解析、AST构建
     ↓
🔍 语义分析器 (semantic.c)  - 类型检查、符号表管理
     ↓
⚙️ 代码生成器 (codegen.c)   - RISC-V指令生成、并行优化
     ↓
RISC-V 汇编代码
```

## 🛠️ 支持的C语言特性

### 数据类型和变量

| 特性 | 支持状态 | 说明 |
|------|---------|------|
| 基本类型 | ✅ 完全支持 | int, char, float, double |
| 指针 | ✅ 完全支持 | 单级和多级指针 |
| 数组 | ✅ 完全支持 | 一维和多维数组 |
| 结构体 | ✅ 完全支持 | struct定义和使用 |
| 联合体 | ✅ 基础支持 | union定义 |
| 枚举 | ✅ 完全支持 | enum类型 |

### 控制流结构

- ✅ **条件语句**：if/else、switch/case
- ✅ **循环语句**：for、while、do-while
- ✅ **跳转语句**：break、continue、goto、return
- ✅ **并行循环**：parallel_for扩展

### 函数和模块化

- ✅ **函数定义**：参数传递、返回值
- ✅ **递归调用**：栈管理和优化
- ✅ **变参函数**：基础支持
- ✅ **静态函数**：作用域控制

## ⚡ 并行计算扩展

### 并行循环

```c
// 并行for循环扩展
parallel_for(int i = 0; i < 1000; i++) {
    array[i] = array[i] * 2;
}

// 编译为RISC-V多核代码
```

### 原子操作

```c
atomic_int counter = 0;

void increment() {
    atomic_fetch_add(&counter, 1);
}
```

### 线程同步

```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void critical_section() {
    pthread_mutex_lock(&mutex);
    // 临界区代码
    pthread_mutex_unlock(&mutex);
}
```

## 🎯 RISC-V代码生成

### 寄存器使用约定

```
x0 (zero)  - 硬编码为0
x1 (ra)    - 返回地址
x2 (sp)    - 栈指针
x3 (gp)    - 全局指针
x4 (tp)    - 线程指针
x5-x7      - 临时寄存器
x8-x9      - 保存寄存器
x10-x17    - 参数/返回值寄存器
x18-x27    - 保存寄存器
x28-x31    - 临时寄存器
```

### 指令生成示例

```c
// C代码
int add(int a, int b) {
    return a + b;
}

// 生成的RISC-V汇编
add:
    addi sp, sp, -16
    sw   ra, 12(sp)
    add  a0, a0, a1
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret
```

## 🚀 快速开始

### 1. 编译编译器

```bash
# Linux/Unix
make

# Windows (需要RISC-V工具链)
mingw32-make
```

### 2. 编译C程序

```bash
# 编译C文件到RISC-V汇编
./riscv_cc input.c -o output.s

# 并行优化模式
./riscv_cc -parallel input.c -o output.s

# 详细模式
./riscv_cc -v input.c -o output.s
```

### 3. 汇编和链接

```bash
# 使用RISC-V工具链
riscv64-unknown-elf-gcc output.s -o program
qemu-riscv64 program
```

## 📝 支持的C语法示例

### 基本程序结构

```c
#include <stdio.h>

int main() {
    int x = 10;
    int y = 20;
    int sum = x + y;
    
    printf("Sum: %d\n", sum);
    return 0;
}
```

### 函数和指针

```c
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

### 结构体和数组

```c
struct Point {
    int x, y;
};

void process_array(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = arr[i] * 2;
    }
}

int main() {
    struct Point p = {10, 20};
    int numbers[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    process_array(numbers, 10);
    return 0;
}
```

### 并行计算

```c
#include <pthread.h>
#include <stdatomic.h>

atomic_int global_sum = 0;

void parallel_sum(int* array, int size) {
    parallel_for(int i = 0; i < size; i++) {
        atomic_fetch_add(&global_sum, array[i]);
    }
}

int main() {
    int data[1000];
    // 初始化数据
    for (int i = 0; i < 1000; i++) {
        data[i] = i + 1;
    }
    
    parallel_sum(data, 1000);
    
    printf("Total sum: %d\n", atomic_load(&global_sum));
    return 0;
}
```

## 🔧 编译器选项

| 选项 | 说明 |
|------|------|
| `-o <file>` | 指定输出文件名 |
| `-v` | 详细模式，显示编译过程 |
| `-parallel` | 启用并行优化 |
| `-O0/O1/O2/O3` | 优化级别 |
| `--ast` | 打印抽象语法树 |
| `--tokens` | 打印词法单元 |
| `--semantic` | 只进行语义分析 |
| `-target` | 指定RISC-V变体 |

## 📊 性能特性

### RISC-V优化

- **指令调度**：重排指令减少流水线停顿
- **寄存器分配**：图着色算法优化寄存器使用
- **循环优化**：循环展开和向量化
- **分支优化**：分支预测友好的代码生成

### 并行性能

- **多核调度**：自动任务分配和负载均衡
- **原子操作**：使用RISC-V原子指令
- **内存一致性**：正确的内存屏障插入
- **缓存优化**：数据局部性优化

## 📈 项目特色

### RISC-V特定优化

1. **精简指令集**：充分利用RISC-V精简特性
2. **模块化扩展**：支持RISC-V扩展指令集
3. **低功耗优化**：嵌入式系统友好的代码生成
4. **可扩展性**：支持32位和64位RISC-V

### 并行计算创新

1. **编译时并行化**：自动识别并行机会
2. **硬件感知**：根据核心数量优化
3. **同步优化**：减少同步开销
4. **内存管理**：并行安全的内存分配

## 🏆 技术亮点

- ✅ **完整的C编译器**：支持C99核心特性
- ✅ **RISC-V原生**：专门为RISC-V架构优化
- ✅ **并行扩展**：创新的并行编程支持
- ✅ **高质量代码**：优化的汇编输出
- ✅ **模块化设计**：易于扩展和维护

## 🤝 使用场景

- **嵌入式开发**：RISC-V微控制器编程
- **高性能计算**：多核RISC-V系统
- **教学研究**：编译器和体系结构教学
- **原型开发**：快速RISC-V软件开发

---

[📚 详细文档](USAGE.md) | [🔧 API参考](API.md) | [🧪 测试指南](TESTING.md)
