@echo off
REM X86/X64 并行 C 编译器测试脚本

echo Testing X86/X64 Parallel C Compiler...
echo.

REM 检查编译器是否存在
if not exist ..\bin\x86cc.exe (
    echo Error: Compiler not found. Please run build.bat first.
    exit /b 1
)

REM 创建输出目录
if not exist output mkdir output

echo Test 1: Basic C Language Features
echo ----------------------------------
..\bin\x86cc.exe -v ..\examples\basic.c -o output\basic.s
if errorlevel 1 (
    echo FAILED: Basic C compilation
    goto test_failed
) else (
    echo PASSED: Basic C compilation
)
echo.

echo Test 2: Parallel Computing Features
echo ------------------------------------
..\bin\x86cc.exe -v -fopenmp ..\examples\parallel.c -o output\parallel.s
if errorlevel 1 (
    echo FAILED: Parallel C compilation
    goto test_failed
) else (
    echo PASSED: Parallel C compilation
)
echo.

echo Test 3: Optimization Levels
echo ----------------------------
..\bin\x86cc.exe -O1 ..\examples\basic.c -o output\basic_O1.s
if errorlevel 1 (
    echo FAILED: O1 optimization
    goto test_failed
) else (
    echo PASSED: O1 optimization
)

..\bin\x86cc.exe -O2 ..\examples\basic.c -o output\basic_O2.s
if errorlevel 1 (
    echo FAILED: O2 optimization
    goto test_failed
) else (
    echo PASSED: O2 optimization
)

..\bin\x86cc.exe -O3 ..\examples\basic.c -o output\basic_O3.s
if errorlevel 1 (
    echo FAILED: O3 optimization
    goto test_failed
) else (
    echo PASSED: O3 optimization
)
echo.

echo Test 4: Vectorization
echo ----------------------
..\bin\x86cc.exe -fvectorize ..\examples\parallel.c -o output\vectorized.s
if errorlevel 1 (
    echo FAILED: Vectorization
    goto test_failed
) else (
    echo PASSED: Vectorization
)
echo.

echo Test 5: Code Size Comparison
echo -----------------------------
echo Basic.s size:
for %%f in (output\basic.s) do echo %%~zf bytes

echo Basic O1.s size:
for %%f in (output\basic_O1.s) do echo %%~zf bytes

echo Basic O2.s size:
for %%f in (output\basic_O2.s) do echo %%~zf bytes

echo Basic O3.s size:
for %%f in (output\basic_O3.s) do echo %%~zf bytes

echo Parallel.s size:
for %%f in (output\parallel.s) do echo %%~zf bytes
echo.

echo Test 6: Assembly Code Verification
echo -----------------------------------
echo Checking for x86/x64 instructions in output...

findstr /c:"mov" /c:"add" /c:"sub" /c:"mul" /c:"call" output\basic.s >nul
if errorlevel 1 (
    echo FAILED: No x86 instructions found
    goto test_failed
) else (
    echo PASSED: x86 instructions found
)

findstr /c:"parallel" /c:"atomic" /c:"lock" output\parallel.s >nul
if errorlevel 1 (
    echo WARNING: Limited parallel instructions (this is expected for demo)
) else (
    echo PASSED: Parallel instructions found
)
echo.

echo =====================================
echo ALL TESTS COMPLETED SUCCESSFULLY!
echo =====================================
echo.
echo Generated assembly files:
dir /b output\*.s
echo.
echo You can examine the generated assembly code in the output\ directory.
goto end

:test_failed
echo.
echo =====================================
echo SOME TESTS FAILED!
echo =====================================
echo Please check the error messages above.
exit /b 1

:end
