# Parallel C++ Compiler - 使用指南

## 概述

这是一个将C++代码转换为C代码的并行编译器，专门设计用于支持C++的核心语法和并行计算扩展。编译器将C++的面向对象特性转换为等价的C结构体和函数调用。

## 功能特性

### 支持的C++核心语法

1. **类和对象**
   - 类定义和实例化
   - 构造函数和析构函数
   - 成员变量和成员函数
   - 访问控制 (public, private, protected)

2. **继承**
   - 单继承支持
   - 虚函数和多态
   - 虚析构函数

3. **运算符重载**
   - 算术运算符 (+, -, *, /, %)
   - 比较运算符 (==, !=, <, >, <=, >=)
   - 赋值运算符 (=, +=, -=, *=, /=)
   - 数组访问运算符 ([])

4. **内存管理**
   - new/delete 操作符
   - 数组版本 new[]/delete[]
   - 自动析构函数调用

5. **模板（基础支持）**
   - 简单的类模板
   - 函数模板

### 并行计算扩展

1. **parallel_class**
   - 线程安全的类
   - 自动互斥锁管理
   - 原子操作支持

2. **parallel_for**
   - 并行循环执行
   - 自动线程分配
   - 负载均衡

3. **thread_safe 方法**
   - 自动加锁的方法
   - 线程安全保证

4. **原子操作**
   - atomic_int, atomic_float
   - 原子读写操作

## 编译和安装

### Linux/Unix系统

```bash
# 编译编译器
make

# 安装到系统目录（可选）
sudo make install

# 运行测试
make test
```

### Windows系统

```batch
# 使用提供的批处理文件
build.bat

# 或者手动编译
gcc -std=c99 -O2 src/*.c -lpthread -o pcpp.exe
```

## 使用方法

### 基本用法

```bash
# 编译C++文件到C文件
./pcpp input.cpp

# 指定输出文件
./pcpp -o output.c input.cpp

# 详细模式
./pcpp -v input.cpp
```

### 编译器选项

- `-o <file>`: 指定输出文件名（默认：output.c）
- `-v`: 详细模式，显示编译过程
- `--ast`: 打印抽象语法树并退出
- `--tokens`: 打印词法单元并退出
- `--semantic`: 只进行语义分析
- `-h, --help`: 显示帮助信息

### 示例程序

#### 1. 简单类定义

```cpp
class Point {
private:
    int x, y;

public:
    Point(int x_val, int y_val) {
        x = x_val;
        y = y_val;
    }
    
    int getX() const {
        return x;
    }
    
    int getY() const {
        return y;
    }
    
    void move(int dx, int dy) {
        x += dx;
        y += dy;
    }
};

int main() {
    Point* p = new Point(10, 20);
    p->move(5, -3);
    int x = p->getX();
    delete p;
    return 0;
}
```

#### 2. 继承和虚函数

```cpp
class Shape {
public:
    virtual float area() = 0;
    virtual ~Shape() {}
};

class Circle : public Shape {
private:
    float radius;

public:
    Circle(float r) {
        radius = r;
    }
    
    virtual float area() {
        return 3.14159f * radius * radius;
    }
};

int main() {
    Shape* shape = new Circle(5.0f);
    float a = shape->area();
    delete shape;
    return 0;
}
```

#### 3. 并行计算

```cpp
parallel_class ParallelCounter {
private:
    atomic_int count;

public:
    ParallelCounter() {
        atomic_store(&count, 0);
    }
    
    thread_safe void increment() {
        atomic_fetch_add(&count, 1);
    }
    
    thread_safe int getValue() {
        return atomic_load(&count);
    }
};

int main() {
    ParallelCounter* counter = new ParallelCounter();
    
    // 并行执行1000次增量操作
    parallel_for(0, 1000, [&](int i) {
        counter->increment();
    });
    
    int final_count = counter->getValue();
    delete counter;
    return 0;
}
```

## 生成的C代码

编译器将C++代码转换为标准C代码，包含以下特性：

### 类转换为结构体

```c
// C++类
class Point {
    int x, y;
public:
    Point(int x, int y);
    void move(int dx, int dy);
};

// 转换为C结构体
typedef struct Point_s {
    int x;
    int y;
} Point_t;

Point_t* Point_new(int x, int y);
void Point_delete(Point_t* self);
void Point_move(Point_t* self, int dx, int dy);
```

### 虚函数表

```c
// 虚函数表结构
typedef struct Shape_vtable_s {
    float (*area)(Shape_t* self);
    void (*destructor)(Shape_t* self);
} Shape_vtable_t;

// 类结构包含虚函数表指针
typedef struct Shape_s {
    Shape_vtable_t* vtable;
    // 其他成员...
} Shape_t;
```

### 并行支持

```c
// 自动生成的并行支持代码
#include <pthread.h>
#include <stdatomic.h>

#define parallel_for(start, end, body) do { \
    int num_threads = 4; \
    pthread_t threads[num_threads]; \
    /* 线程创建和管理代码 */ \
} while(0)
```

## 调试和测试

### 运行测试套件

```bash
# 运行所有测试
./test_compiler.sh

# 编译示例程序
./pcpp examples/simple.cpp
./pcpp examples/classes.cpp
./pcpp examples/parallel.cpp
```

### 调试输出

```bash
# 查看AST
./pcpp --ast examples/classes.cpp

# 查看词法单元
./pcpp --tokens examples/simple.cpp

# 详细编译过程
./pcpp -v examples/parallel.cpp
```

### 内存泄漏检查

```bash
# 使用valgrind检查内存泄漏（Linux）
valgrind --leak-check=full ./pcpp examples/classes.cpp
```

## 限制和已知问题

1. **模板支持有限**：只支持简单的类模板和函数模板
2. **异常处理**：不支持try/catch/throw
3. **STL**：不支持标准模板库
4. **多重继承**：不支持多重继承
5. **友元函数**：不支持friend关键字
6. **命名空间**：基础支持，转换为前缀

## 贡献和扩展

### 添加新功能

1. 在`pcpp.h`中添加新的AST节点类型
2. 在`lexer.c`中添加新的关键字或操作符
3. 在`parser.c`中添加语法解析规则
4. 在`semantic.c`中添加语义检查
5. 在`codegen.c`中添加代码生成逻辑

### 提交Bug报告

请在项目主页提交Issue，包含：
- 输入的C++代码
- 编译器版本和选项
- 错误信息或异常行为
- 预期的输出

## 许可证

MIT License - 详见LICENSE文件

## 联系方式

- 项目主页：[GitHub Repository]
- 邮箱：[Contact Email]
- 文档：[Documentation Website]

---

**注意**：这是一个实验性的编译器，主要用于教学和研究目的。在生产环境中使用前请充分测试。
