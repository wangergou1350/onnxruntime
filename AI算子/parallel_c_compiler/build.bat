@echo off
echo Building ParallelC Compiler...

REM Check if gcc is available
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: GCC compiler not found. Please install MinGW-w64 or similar.
    echo You can download it from: https://www.mingw-w64.org/downloads/
    pause
    exit /b 1
)

REM Compile the ParallelC compiler
echo Compiling ParallelC compiler...
gcc -o pcc.exe src\main.c src\lexer.c src\parser.c src\semantic.c src\codegen.c

if %errorlevel% neq 0 (
    echo Error: Compilation failed.
    pause
    exit /b 1
)

echo ParallelC compiler built successfully!
echo.
echo Usage: pcc.exe input.pcc -o output.c
echo Then:  gcc -pthread -o program output.c
echo.
pause
