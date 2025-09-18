# Parallel C++ Compiler

一个将C++代码转换为C代码的并行编译器，专门设计用于支持C++的核心语法和并行计算扩展。

## 🚀 项目概览

这是一个创新的C++到C的转译器，支持现代C++核心特性和专为并行计算设计的语言扩展。编译器采用经典的四阶段架构：词法分析 → 语法分析 → 语义分析 → 代码生成，生成高质量的C代码。

### ✨ 核心特性

- **🔧 完整的C++核心语法支持**：类、继承、多态、运算符重载
- **⚡ 并行计算扩展**：parallel_class、parallel_for、thread_safe
- **🎯 高质量代码生成**：优化的C代码输出，支持虚函数表和线程安全
- **🛡️ 强健的错误处理**：详细的语法和语义错误报告
- **📊 调试工具**：AST可视化、token分析、详细编译过程

## 🏗️ 编译器架构

```
C++ 源代码
     ↓
📝 词法分析器 (lexer.c)     - 关键字、操作符、标识符识别
     ↓
🌳 语法分析器 (parser.c)    - 递归下降解析、AST构建
     ↓
🔍 语义分析器 (semantic.c)  - 类型检查、符号表管理
     ↓
⚙️ 代码生成器 (codegen.c)   - C++到C转换、并行代码生成
     ↓
标准 C 代码
```
    Point(int x, int y) {
        this->x = x;
        this->y = y;
    }
    
    ~Point() {
        // 析构函数
    }
    
    int getX() {
        return x;
    }
    
    Point operator+(Point other) {
        return Point(x + other.x, y + other.y);
    }
};
```

### 继承和多态
```cpp
class Shape {
public:
    virtual double area() {
        return 0.0;
    }
    
    virtual ~Shape() {}
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) {
        radius = r;
    }
    
    virtual double area() {
        return 3.14159 * radius * radius;
    }
};
```

### 并行编程扩展
```cpp
parallel_class Matrix {
private:
    int data[100][100];
    
public:
    Matrix() {
        parallel_for(0, 100, {
            for (int j = 0; j < 100; j++) {
                data[i][j] = 0;
            }
        });
    }
    
    thread_safe void multiply(Matrix& other, Matrix& result) {
        parallel_for(0, 100, {
            for (int j = 0; j < 100; j++) {
                result.data[i][j] = 0;
                for (int k = 0; k < 100; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        });
    }
    
    void parallel_process() {
        parallel_invoke([&]() {
            this->normalize();
        }, [&]() {
            this->validate();
        });
    }
};
```

## 🎯 编译流程

```
C++ 源码 (.pcpp)
    ↓
C++ 词法分析 (支持类、继承等关键字)
    ↓
C++ 语法分析 (构建 C++ AST)
    ↓
C++ 语义分析 (类型系统、继承检查)
    ↓
C 代码生成 (虚函数表、构造析构)
    ↓
标准 C 代码 (.c)
```

## 🔨 构建和使用

### 构建编译器
```bash
# Linux/Mac
make

# Windows
build.bat
```

### 编译 C++ 程序
```bash
# 编译 ParallelC++ 到 C
./pcpp source.pcpp -o output.c

# 编译生成的 C 代码
gcc -pthread -o program output.c

# 运行程序
./program
```

## 🧪 测试
```bash
# 运行测试套件
./test.sh      # Linux/Mac
./test.ps1     # Windows
```

## 💡 实现特点

### C++ 到 C 的转换
- **类**: 转换为结构体 + 函数指针（虚函数表）
- **继承**: 结构体嵌套 + 虚函数表继承
- **构造函数**: 转换为初始化函数
- **析构函数**: 转换为清理函数
- **成员函数**: 转换为以 this 指针为首参数的函数
- **操作符重载**: 转换为命名函数

### 并行扩展实现
- **parallel_for**: 在类方法中支持 this 指针传递
- **parallel_class**: 自动生成线程安全的访问方法
- **thread_safe**: 添加互斥锁保护
- **parallel_invoke**: 创建并行任务执行 lambda 表达式

---

**ParallelC++**: 将 C++ 的强大特性与并行计算完美结合! 🚀
