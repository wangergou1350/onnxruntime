# ParallelC Compiler - 完整实现说明

## 项目概述

ParallelC 是一个支持并行计算的 C 语言编译器，它扩展了标准 C 语言，添加了并行编程原语。编译器将 ParallelC 代码转换为标准的 C 代码，使用 pthread 库实现并行执行。

## 核心特性

### 1. 并行语言构造
- `parallel_for(start, end, body)` - 并行 for 循环
- `atomic_add(var, value)` - 原子加法操作
- `atomic_sub(var, value)` - 原子减法操作
- `barrier()` - 线程同步屏障
- `thread_id()` - 获取当前线程 ID
- `num_threads()` - 获取线程数量

### 2. 编译器架构
```
源代码 (.pcc) → 词法分析 → 语法分析 → 语义分析 → 代码生成 → C 代码 (.c)
```

## 实现细节

### 词法分析器 (lexer.c)
- 识别并行关键字：`parallel_for`, `atomic_add`, `atomic_sub`, `barrier`, `thread_id`
- 支持标准 C 语言的所有 token 类型
- 处理注释、字符串字面量和数字
- 实现行号跟踪用于错误报告

### 语法分析器 (parser.c)
- 使用递归下降解析技术
- 构建抽象语法树 (AST)
- 支持所有 C 语言语法结构
- 添加并行构造的语法支持

### 语义分析器 (semantic.c)
- 符号表管理和作用域检查
- 类型检查和类型兼容性验证
- 函数声明和调用验证
- 变量使用前定义检查

### 代码生成器 (codegen.c)
- 将 AST 转换为标准 C 代码
- `parallel_for` 转换为 pthread 实现
- 原子操作映射到 GCC 内建函数
- 自动生成线程管理代码

## 使用示例

### 基础并行循环
```c
int main() {
    int sum = 0;
    
    // 并行计算 0 到 999 的和
    parallel_for(0, 1000, {
        atomic_add(&sum, i);
    });
    
    printf("Sum: %d\n", sum);
    return 0;
}
```

### 矩阵乘法
```c
void matrix_multiply(int A[100][100], int B[100][100], int C[100][100], int size) {
    parallel_for(0, size, {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    });
}
```

### 线程同步
```c
int main() {
    parallel_for(0, 4, {
        printf("Thread %d working...\n", thread_id());
        
        // 等待所有线程完成当前阶段
        barrier();
        
        if (thread_id() == 0) {
            printf("All threads synchronized\n");
        }
    });
    
    return 0;
}
```

## 编译和运行

### 1. 构建编译器
```bash
make clean
make
```

### 2. 编译 ParallelC 程序
```bash
./pcc source.pcc -o output.c
gcc -pthread -o program output.c
./program
```

### 3. 运行测试套件
```bash
# Linux/Mac
./test.sh

# Windows (PowerShell)
./test.ps1
```

## 生成的 C 代码结构

编译器生成的 C 代码包含以下组件：

### 运行时初始化
```c
static int _pcc_num_threads = 4;
static pthread_barrier_t _pcc_barrier;

void _pcc_init_runtime() {
    _pcc_num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_barrier_init(&_pcc_barrier, NULL, _pcc_num_threads);
}
```

### parallel_for 实现
```c
// 为每个 parallel_for 生成线程函数
void* _temp_1(void* arg) {
    _temp_1_data_t* data = (_temp_1_data_t*)arg;
    for (int i = data->start; i < data->end; i++) {
        // 用户代码
    }
    return NULL;
}

// 线程创建和管理
pthread_t threads[_pcc_num_threads];
for (int t = 0; t < _pcc_num_threads; t++) {
    pthread_create(&threads[t], NULL, _temp_1, &thread_data[t]);
}
```

### 原子操作映射
```c
atomic_add(&var, value) → __sync_fetch_and_add(&var, value)
atomic_sub(&var, value) → __sync_fetch_and_sub(&var, value)
barrier() → pthread_barrier_wait(&_pcc_barrier)
thread_id() → ((int)pthread_self())
```

## 项目文件结构

```
parallel_c_compiler/
├── src/
│   ├── pcc.h          # 主头文件，定义所有数据结构
│   ├── main.c         # 主程序入口
│   ├── lexer.c        # 词法分析器
│   ├── parser.c       # 语法分析器
│   ├── semantic.c     # 语义分析器
│   └── codegen.c      # 代码生成器
├── examples/
│   ├── demo.pcc       # 基础示例程序
│   └── advanced.pcc   # 高级示例程序
├── README.md          # 项目说明
├── Makefile           # 构建配置
└── test.sh            # 测试脚本
```

## 技术特点

### 1. 模块化设计
- 每个编译阶段独立实现
- 清晰的接口定义
- 易于维护和扩展

### 2. 完整的错误处理
- 词法错误检测
- 语法错误报告
- 语义错误诊断
- 带行号的错误信息

### 3. 高效的并行实现
- 基于 pthread 的线程池
- 工作负载自动分配
- 原子操作优化
- 线程同步支持

### 4. 标准 C 兼容
- 生成标准 C99 代码
- 使用标准库函数
- 支持现有 C 编译器

## 性能优化

### 1. 线程数量自适应
- 自动检测 CPU 核心数
- 动态调整线程池大小
- 避免过度并行化

### 2. 内存管理优化
- 栈分配的线程数据
- 最小化动态内存分配
- 自动内存清理

### 3. 同步开销最小化
- 无锁原子操作
- 高效的屏障实现
- 线程局部存储

## 扩展可能性

### 1. 更多并行原语
- `parallel_reduce` - 并行归约
- `parallel_scan` - 并行扫描
- `parallel_sort` - 并行排序

### 2. 高级同步机制
- 条件变量支持
- 读写锁
- 信号量

### 3. 性能分析工具
- 执行时间统计
- 线程负载分析
- 内存使用监控

## 总结

ParallelC 编译器提供了一个完整的并行编程解决方案，从语言设计到代码生成都经过精心优化。它既保持了 C 语言的简洁性，又添加了强大的并行编程能力，是学习编译器原理和并行计算的优秀项目。

通过这个项目，我们展示了：
- 编译器前端设计（词法、语法、语义分析）
- 代码生成和优化技术
- 并行编程模型实现
- 系统级编程最佳实践

这个实现为进一步研究编译器技术和并行计算提供了坚实的基础。
