@echo off
echo 编译 RISC-V 并行 C 编译器...

REM 检查是否安装了 GCC
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 GCC 编译器
    echo 请安装 MinGW-w64 或 MSYS2 来获取 GCC
    echo 下载地址: https://www.msys2.org/
    pause
    exit /b 1
)

REM 创建输出目录
if not exist bin mkdir bin

REM 编译编译器
echo 编译源文件...
gcc -Wall -Wextra -std=c99 -O2 -o bin\riscv_cc.exe ^
    src\main.c ^
    src\lexer.c ^
    src\parser.c ^
    src\semantic.c ^
    src\codegen.c ^
    src\utils.c

if %errorlevel% equ 0 (
    echo 编译成功！
    echo 编译器位置: bin\riscv_cc.exe
    echo.
    echo 使用方法:
    echo   bin\riscv_cc.exe input.c -o output.s
    echo   bin\riscv_cc.exe -h    显示帮助
    echo   bin\riscv_cc.exe -v input.c    显示详细信息
) else (
    echo 编译失败！
    pause
    exit /b 1
)

pause
