@echo off
echo 测试 RISC-V 并行 C 编译器...

REM 检查编译器是否存在
if not exist bin\riscv_cc.exe (
    echo 错误: 编译器不存在，请先运行 build.bat
    pause
    exit /b 1
)

REM 创建测试输出目录
if not exist test_output mkdir test_output

echo.
echo 测试基础语法...
bin\riscv_cc.exe tests\basic_syntax.c -o test_output\basic_syntax.s
if %errorlevel% equ 0 (
    echo ✓ basic_syntax.c 编译成功
) else (
    echo ✗ basic_syntax.c 编译失败
)

echo.
echo 测试函数...
bin\riscv_cc.exe tests\functions.c -o test_output\functions.s
if %errorlevel% equ 0 (
    echo ✓ functions.c 编译成功
) else (
    echo ✗ functions.c 编译失败
)

echo.
echo 测试数组...
bin\riscv_cc.exe tests\arrays.c -o test_output\arrays.s
if %errorlevel% equ 0 (
    echo ✓ arrays.c 编译成功
) else (
    echo ✗ arrays.c 编译失败
)

echo.
echo 测试指针...
bin\riscv_cc.exe tests\pointers.c -o test_output\pointers.s
if %errorlevel% equ 0 (
    echo ✓ pointers.c 编译成功
) else (
    echo ✗ pointers.c 编译失败
)

echo.
echo 测试结构体...
bin\riscv_cc.exe tests\structs.c -o test_output\structs.s
if %errorlevel% equ 0 (
    echo ✓ structs.c 编译成功
) else (
    echo ✗ structs.c 编译失败
)

echo.
echo 测试并行计算...
bin\riscv_cc.exe tests\parallel.c -o test_output\parallel.s
if %errorlevel% equ 0 (
    echo ✓ parallel.c 编译成功
) else (
    echo ✗ parallel.c 编译失败
)

echo.
echo 测试完成！
echo 生成的汇编文件位于 test_output\ 目录中

pause
