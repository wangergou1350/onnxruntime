@echo off
REM X86/X64 并行 C 编译器 Windows 构建脚本

echo Building X86/X64 Parallel C Compiler...

REM 创建目录
if not exist obj mkdir obj
if not exist bin mkdir bin

REM 编译源文件
gcc -Wall -Wextra -std=c99 -O2 -g -Iinclude -c src/lexer.c -o obj/lexer.o
if errorlevel 1 goto error

gcc -Wall -Wextra -std=c99 -O2 -g -Iinclude -c src/parser.c -o obj/parser.o
if errorlevel 1 goto error

gcc -Wall -Wextra -std=c99 -O2 -g -Iinclude -c src/semantic.c -o obj/semantic.o
if errorlevel 1 goto error

gcc -Wall -Wextra -std=c99 -O2 -g -Iinclude -c src/codegen.c -o obj/codegen.o
if errorlevel 1 goto error

gcc -Wall -Wextra -std=c99 -O2 -g -Iinclude -c src/utils.c -o obj/utils.o
if errorlevel 1 goto error

gcc -Wall -Wextra -std=c99 -O2 -g -Iinclude -c src/main.c -o obj/main.o
if errorlevel 1 goto error

REM 链接生成可执行文件
gcc obj/lexer.o obj/parser.o obj/semantic.o obj/codegen.o obj/utils.o obj/main.o -o bin/x86cc.exe
if errorlevel 1 goto error

echo Build successful! Executable: bin/x86cc.exe
echo.
echo Usage: bin/x86cc.exe [options] input.c
echo   -o output.s    Specify output file
echo   -v             Verbose output
echo   -O1/O2/O3      Optimization level
echo   -fopenmp       Enable parallel processing
echo   -fvectorize    Enable vectorization
goto end

:error
echo Build failed!
exit /b 1

:end
