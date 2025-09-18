# RISC-V 并行 C 编译器测试

## 测试说明

这个目录包含了 RISC-V 并行 C 编译器的测试用例，用于验证编译器的各种功能。

## 测试文件

### 基础语法测试
- **basic_syntax.c**: 测试基本的 C 语法特性
  - 算术运算
  - 条件语句 (if/else)
  - 循环语句 (for/while)
  - 变量声明和赋值

### 函数测试
- **functions.c**: 测试函数相关功能
  - 函数定义和调用
  - 参数传递
  - 返回值
  - 递归函数

### 数组测试
- **arrays.c**: 测试数组操作
  - 一维数组
  - 多维数组
  - 数组初始化
  - 指针算术

### 指针测试
- **pointers.c**: 测试指针操作
  - 基本指针操作
  - 指针算术
  - 函数指针
  - 指针的指针

### 结构体测试
- **structs.c**: 测试结构体和复杂数据类型
  - 结构体定义
  - 成员访问
  - 嵌套结构体
  - 结构体数组

### 并行计算测试
- **parallel.c**: 测试并行计算扩展
  - parallel_for 循环
  - 原子操作 (atomic)
  - 内存屏障 (barrier)
  - 临界区 (critical)
  - 线程局部存储 (thread_local)

## 运行测试

使用 Makefile 中的测试目标：

```bash
# 编译所有测试
make test

# 编译单个测试
./riscv_cc tests/basic_syntax.c -o basic_syntax.s
./riscv_cc tests/functions.c -o functions.s
./riscv_cc tests/arrays.c -o arrays.s
./riscv_cc tests/pointers.c -o pointers.s
./riscv_cc tests/structs.c -o structs.s
./riscv_cc tests/parallel.c -o parallel.s
```

## 预期结果

每个测试文件都应该能够成功编译为 RISC-V 汇编代码，生成的汇编代码应该：

1. 语法正确
2. 符合 RISC-V 指令集规范
3. 正确实现原始 C 代码的语义
4. 对于并行测试，正确生成并行计算相关的指令序列

## 调试信息

编译器支持调试模式，可以查看详细的编译过程：

```bash
./riscv_cc -v tests/basic_syntax.c -o basic_syntax.s
```

这将输出：
- 词法分析结果
- 语法分析树
- 语义分析信息
- 代码生成过程
