# ParallelC Compiler

一个支持并行计算的 C 语言编译器，扩展了标准 C 语言，添加了并行编程原语。

## 🚀 快速开始

### 系统要求
- **Windows**: MinGW-w64 或 Visual Studio (需要 GCC)
- **Linux/Mac**: GCC 和 Make
- **pthread 库支持**

### 安装 GCC (Windows)
1. 下载 MinGW-w64: https://www.mingw-w64.org/downloads/
2. 或使用 MSYS2: https://www.msys2.org/
3. 确保 `gcc` 在 PATH 环境变量中

### 构建编译器

**Windows:**
```cmd
build.bat
```

**Linux/Mac:**
```bash
make clean
make
```

### 编译和运行 ParallelC 程序

1. **编译 ParallelC 源码到 C 代码:**
   ```bash
   ./pcc source.pcc -o output.c
   ```

2. **编译生成的 C 代码:**
   ```bash
   gcc -pthread -o program output.c
   ```

3. **运行程序:**
   ```bash
   ./program
   ```

## 📋 语言特性

### 并行编程原语

| 原语 | 功能 | 示例 |
|------|------|------|
| `parallel_for(start, end, body)` | 并行循环 | `parallel_for(0, 1000, { sum += i; })` |
| `atomic_add(&var, value)` | 原子加法 | `atomic_add(&counter, 1)` |
| `atomic_sub(&var, value)` | 原子减法 | `atomic_sub(&counter, 1)` |
| `barrier()` | 线程同步 | `barrier();` |
| `thread_id()` | 当前线程ID | `int tid = thread_id();` |
| `num_threads()` | 线程总数 | `int nt = num_threads();` |

### 基础语法示例

```c
// 并行数组求和
int parallel_sum(int arr[], int size) {
    int total = 0;
    
    parallel_for(0, size, {
        atomic_add(&total, arr[i]);
    });
    
    return total;
}

// 矩阵乘法
void matrix_multiply(int A[100][100], int B[100][100], int C[100][100]) {
    parallel_for(0, 100, {
        for (int j = 0; j < 100; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 100; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    });
}

// 线程同步示例
int main() {
    printf("Starting parallel computation with %d threads\n", num_threads());
    
    parallel_for(0, 8, {
        printf("Thread %d processing item %d\n", thread_id(), i);
        
        // 同步点
        barrier();
        
        if (thread_id() == 0) {
            printf("All threads completed phase 1\n");
        }
    });
    
    return 0;
}
```

## 📁 项目结构

```
parallel_c_compiler/
├── src/                    # 源代码
│   ├── pcc.h              # 主头文件
│   ├── main.c             # 程序入口
│   ├── lexer.c            # 词法分析器
│   ├── parser.c           # 语法分析器
│   ├── semantic.c         # 语义分析器
│   └── codegen.c          # 代码生成器
├── examples/               # 示例程序
│   ├── demo.pcc           # 基础示例
│   └── advanced.pcc       # 高级示例
├── Makefile               # Linux/Mac 构建文件
├── build.bat              # Windows 构建脚本
├── test.sh                # Linux/Mac 测试脚本
├── test.ps1               # Windows 测试脚本
└── README.md              # 项目说明
```

## 🧪 运行测试

**Windows (PowerShell):**
```powershell
.\test.ps1
```

**Linux/Mac:**
```bash
./test.sh
```

**详细测试输出:**
```powershell
.\test.ps1 -Verbose
```

## 🔧 编译器架构

### 编译流程
```
ParallelC 源码 (.pcc)
    ↓
词法分析 (Lexer)
    ↓
语法分析 (Parser) → AST
    ↓
语义分析 (Semantic)
    ↓
代码生成 (CodeGen)
    ↓
标准 C 代码 (.c)
```

### 核心模块

1. **词法分析器 (lexer.c)**
   - Token 识别和分类
   - 并行关键字处理
   - 错误检测和报告

2. **语法分析器 (parser.c)**
   - 递归下降解析
   - AST 构建
   - 语法错误恢复

3. **语义分析器 (semantic.c)**
   - 符号表管理
   - 类型检查
   - 作用域分析

4. **代码生成器 (codegen.c)**
   - C 代码生成
   - pthread 映射
   - 运行时库集成

## 💡 实现细节

### parallel_for 实现原理

ParallelC 的 `parallel_for` 会被转换为如下的 pthread 代码:

```c
// 原始 ParallelC 代码
parallel_for(0, 1000, {
    atomic_add(&sum, i);
});

// 生成的 C 代码
typedef struct {
    int start, end;
    // 共享数据
} thread_data_t;

void* worker_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    for (int i = data->start; i < data->end; i++) {
        __sync_fetch_and_add(&sum, i);
    }
    return NULL;
}

// 线程创建和管理
pthread_t threads[num_threads];
thread_data_t thread_data[num_threads];

// 工作负载分配
int chunk_size = 1000 / num_threads;
for (int t = 0; t < num_threads; t++) {
    thread_data[t].start = t * chunk_size;
    thread_data[t].end = (t + 1) * chunk_size;
    pthread_create(&threads[t], NULL, worker_thread, &thread_data[t]);
}

// 等待完成
for (int t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL);
}
```

### 原子操作映射

| ParallelC | C 实现 |
|-----------|---------|
| `atomic_add(&var, val)` | `__sync_fetch_and_add(&var, val)` |
| `atomic_sub(&var, val)` | `__sync_fetch_and_sub(&var, val)` |
| `barrier()` | `pthread_barrier_wait(&barrier)` |
| `thread_id()` | `((int)pthread_self())` |
| `num_threads()` | `_pcc_num_threads` |

## 🎯 应用场景

### 科学计算
- 矩阵运算
- 数值积分
- 蒙特卡洛模拟

### 图像处理
- 并行滤波
- 图像变换
- 特征检测

### 数据处理
- 并行排序
- 数据归约
- 统计分析

## 🔮 扩展计划

### 短期目标
- [ ] 向量化操作支持
- [ ] 更多同步原语
- [ ] 性能分析工具

### 长期目标
- [ ] GPU 计算支持
- [ ] 分布式计算
- [ ] 可视化调试器

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 问题报告: [Issues]
- 文档: [Wiki]

---

**ParallelC**: 让并行编程变得简单! 🚀
    
    // 初始化数组
    for(int i = 0; i < 1000; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    // 并行计算
    parallel_for(0, 1000, {
        int i = thread_index();
        c[i] = a[i] + b[i];
    });
    
    return 0;
}
```
