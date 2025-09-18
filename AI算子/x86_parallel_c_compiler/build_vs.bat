@echo off
REM X86/X64 并行 C 编译器 Visual Studio 构建脚本

echo Building X86/X64 Parallel C Compiler with Visual Studio...

REM 尝试找到 Visual Studio
set VSINSTALLDIR=
for /f "usebackq tokens=*" %%i in (`vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VSINSTALLDIR=%%i
)

if "%VSINSTALLDIR%"=="" (
    echo Visual Studio not found. Trying alternative method...
    
    REM 尝试使用环境变量
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo Error: Visual Studio or compatible C compiler not found.
        echo Please install Visual Studio Community or ensure GCC is in PATH.
        echo Alternatively, install MinGW-w64 from: https://www.mingw-w64.org/
        goto error
    )
) else (
    call "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat"
)

REM 创建目录
if not exist obj mkdir obj
if not exist bin mkdir bin

REM 编译源文件
echo Compiling lexer.c...
cl /nologo /W3 /O2 /Iinclude /c src\lexer.c /Fo:obj\lexer.obj
if errorlevel 1 goto error

echo Compiling parser.c...
cl /nologo /W3 /O2 /Iinclude /c src\parser.c /Fo:obj\parser.obj
if errorlevel 1 goto error

echo Compiling semantic.c...
cl /nologo /W3 /O2 /Iinclude /c src\semantic.c /Fo:obj\semantic.obj
if errorlevel 1 goto error

echo Compiling codegen.c...
cl /nologo /W3 /O2 /Iinclude /c src\codegen.c /Fo:obj\codegen.obj
if errorlevel 1 goto error

echo Compiling utils.c...
cl /nologo /W3 /O2 /Iinclude /c src\utils.c /Fo:obj\utils.obj
if errorlevel 1 goto error

echo Compiling main.c...
cl /nologo /W3 /O2 /Iinclude /c src\main.c /Fo:obj\main.obj
if errorlevel 1 goto error

REM 链接生成可执行文件
echo Linking executable...
link /nologo obj\lexer.obj obj\parser.obj obj\semantic.obj obj\codegen.obj obj\utils.obj obj\main.obj /out:bin\x86cc.exe
if errorlevel 1 goto error

echo Build successful! Executable: bin\x86cc.exe
echo.
echo Usage: bin\x86cc.exe [options] input.c
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
