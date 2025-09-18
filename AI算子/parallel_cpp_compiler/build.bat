@echo off
REM Build script for Parallel C++ Compiler on Windows

echo Building Parallel C++ Compiler...

REM Create object directory
if not exist obj mkdir obj

REM Compile source files
echo Compiling lexer...
gcc -Wall -Wextra -std=c99 -O2 -g -c src/lexer.c -o obj/lexer.o
if errorlevel 1 goto error

echo Compiling parser...
gcc -Wall -Wextra -std=c99 -O2 -g -c src/parser.c -o obj/parser.o
if errorlevel 1 goto error

echo Compiling semantic analyzer...
gcc -Wall -Wextra -std=c99 -O2 -g -c src/semantic.c -o obj/semantic.o
if errorlevel 1 goto error

echo Compiling code generator...
gcc -Wall -Wextra -std=c99 -O2 -g -c src/codegen.c -o obj/codegen.o
if errorlevel 1 goto error

echo Compiling main...
gcc -Wall -Wextra -std=c99 -O2 -g -c src/main.c -o obj/main.o
if errorlevel 1 goto error

REM Link executable
echo Linking executable...
gcc obj/main.o obj/lexer.o obj/parser.o obj/semantic.o obj/codegen.o -lpthread -o pcpp.exe
if errorlevel 1 goto error

echo Build successful! Executable: pcpp.exe
echo.
echo Usage: pcpp.exe [options] input_file.cpp
echo   -o output.c    Specify output file
echo   -v             Verbose mode
echo   --ast          Print AST
echo   --tokens       Print tokens
echo   --semantic     Semantic analysis only
echo.
goto end

:error
echo Build failed!
pause

:end
