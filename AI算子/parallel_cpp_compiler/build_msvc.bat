@echo off
REM Build script for Parallel C++ Compiler using Microsoft C Compiler

echo Building Parallel C++ Compiler with Microsoft C Compiler...

REM Try to find Visual Studio environment
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul

REM Create object directory
if not exist obj mkdir obj

REM Check if cl is available
where cl >nul 2>nul
if errorlevel 1 (
    echo Microsoft C Compiler not found. Please install Visual Studio or use build.bat for GCC.
    echo.
    echo Alternative: Install MinGW-w64 or TDM-GCC and use build.bat
    pause
    exit /b 1
)

echo Using Microsoft C Compiler...

REM Compile source files
echo Compiling lexer...
cl /c /W3 /O2 src\lexer.c /Fo:obj\lexer.obj
if errorlevel 1 goto error

echo Compiling parser...
cl /c /W3 /O2 src\parser.c /Fo:obj\parser.obj
if errorlevel 1 goto error

echo Compiling semantic analyzer...
cl /c /W3 /O2 src\semantic.c /Fo:obj\semantic.obj
if errorlevel 1 goto error

echo Compiling code generator...
cl /c /W3 /O2 src\codegen.c /Fo:obj\codegen.obj
if errorlevel 1 goto error

echo Compiling main...
cl /c /W3 /O2 src\main.c /Fo:obj\main.obj
if errorlevel 1 goto error

REM Link executable
echo Linking executable...
link obj\main.obj obj\lexer.obj obj\parser.obj obj\semantic.obj obj\codegen.obj /OUT:pcpp.exe
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
