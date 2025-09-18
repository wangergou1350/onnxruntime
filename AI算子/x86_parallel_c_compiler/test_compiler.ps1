# X86/X64 并行 C 编译器测试脚本
param(
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

Write-Host "X86/X64 Parallel C Compiler Test Suite" -ForegroundColor Yellow
Write-Host "=======================================" -ForegroundColor Yellow

# 测试计数器
$testsRun = 0
$testsPassed = 0

# 运行测试的函数
function Invoke-Test {
    param(
        [string]$TestName,
        [string]$InputFile,
        [string]$Options = "",
        [string]$ExpectedPattern = ""
    )
    
    Write-Host "`nRunning test: $TestName" -ForegroundColor Cyan
    $script:testsRun++
    
    try {
        $outputFile = [System.IO.Path]::ChangeExtension($InputFile, ".s")
        
        # 构建命令行
        $cmd = ".\demo_compiler.ps1 -InputFile `"$InputFile`" -OutputFile `"$outputFile`""
        if ($Options) {
            $cmd += " $Options"
        }
        
        # 执行编译器
        $result = Invoke-Expression $cmd 2>&1
        
        # PowerShell 中检查编译成功的方式
        $success = $true
        if ($result -match "Compilation failed" -or $result -match "❌") {
            $success = $false
        }
        
        if ($success) {
            Write-Host "✅ Compilation successful" -ForegroundColor Green
            
            # 检查输出文件是否生成
            if (Test-Path $outputFile) {
                Write-Host "✅ Assembly file generated" -ForegroundColor Green
                
                # 如果有期望的模式，检查输出内容
                if ($ExpectedPattern) {
                    $content = Get-Content $outputFile -Raw
                    if ($content -match $ExpectedPattern) {
                        Write-Host "✅ Output contains expected pattern" -ForegroundColor Green
                        $script:testsPassed++
                    } else {
                        Write-Host "❌ Output doesn't contain expected pattern: $ExpectedPattern" -ForegroundColor Red
                    }
                } else {
                    $script:testsPassed++
                }
                
                if ($Verbose) {
                    $lineCount = (Get-Content $outputFile | Measure-Object -Line).Lines
                    Write-Host "   Generated $lineCount lines of assembly" -ForegroundColor Gray
                }
                
            } else {
                Write-Host "❌ Assembly file not generated" -ForegroundColor Red
            }
        } else {
            Write-Host "❌ Compilation failed" -ForegroundColor Red
            if ($Verbose) {
                Write-Host "Output: $result" -ForegroundColor Gray
            }
        }
        
    } catch {
        Write-Host "❌ Test failed with exception: $($_.Exception.Message)" -ForegroundColor Red
        if ($Verbose) {
            Write-Host $_.Exception.StackTrace -ForegroundColor Gray
        }
    }
}

# 检查演示编译器是否存在
if (-not (Test-Path ".\demo_compiler.ps1")) {
    Write-Host "❌ Demo compiler not found: demo_compiler.ps1" -ForegroundColor Red
    exit 1
}

# 检查示例文件是否存在
if (-not (Test-Path "examples\basic.c")) {
    Write-Host "❌ Example files not found in examples\ directory" -ForegroundColor Red
    exit 1
}

Write-Host "`nStarting tests..." -ForegroundColor Green
Write-Host ""

# 测试 1: 基础 C 语言功能
Invoke-Test "Basic C Language Features" "examples\basic.c" "" "main:"

# 测试 2: 详细输出模式
Invoke-Test "Verbose Output Mode" "examples\basic.c" "-Verbose" "Lexical analysis"

# 测试 3: 优化等级 O1
Invoke-Test "Optimization Level O1" "examples\basic.c" "-OptLevel 1" "O1 optimizations"

# 测试 4: 优化等级 O2
Invoke-Test "Optimization Level O2" "examples\basic.c" "-OptLevel 2" "O2 optimizations"

# 测试 5: 优化等级 O3
Invoke-Test "Optimization Level O3" "examples\basic.c" "-OptLevel 3" "O3 optimizations"

# 测试 6: 并行计算示例
Invoke-Test "Parallel Computing" "examples\parallel.c" "-OpenMP" "Parallel region"

# 测试 7: 向量化优化
Invoke-Test "Vectorization" "examples\parallel.c" "-Vectorize" "SSE/AVX"

# 测试 8: 完整优化（并行 + 向量化 + O3）
Invoke-Test "Full Optimization" "examples\parallel.c" "-OpenMP -Vectorize -OptLevel 3" "Parallel region"

# 测试 9: 帮助信息
Write-Host "`nTesting help functionality..." -ForegroundColor Cyan
$script:testsRun++
try {
    $helpOutput = .\demo_compiler.ps1 -Help 2>&1
    if ($helpOutput -match "X86/X64 Parallel C Compiler") {
        Write-Host "✅ Help command works" -ForegroundColor Green
        $script:testsPassed++
    } else {
        Write-Host "❌ Help output incorrect" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Help command failed" -ForegroundColor Red
}

# 测试 10: 错误处理（不存在的文件）
Write-Host "`nTesting error handling..." -ForegroundColor Cyan
$script:testsRun++
try {
    $errorOutput = .\demo_compiler.ps1 -InputFile "nonexistent.c" 2>&1
    if ($errorOutput -match "not found" -or $LASTEXITCODE -ne 0) {
        Write-Host "✅ Error handling works for missing files" -ForegroundColor Green
        $script:testsPassed++
    } else {
        Write-Host "❌ Should fail for missing files" -ForegroundColor Red
    }
} catch {
    Write-Host "✅ Correctly threw exception for missing file" -ForegroundColor Green
    $script:testsPassed++
}

# 代码质量检查
Write-Host "`nCode Quality Checks:" -ForegroundColor Cyan
$script:testsRun++

# 检查生成的汇编代码质量
$assemblyFiles = Get-ChildItem -Path "examples" -Filter "*.s"
$qualityPassed = $true

foreach ($file in $assemblyFiles) {
    $content = Get-Content $file.FullName -Raw
    
    # 检查必需的汇编指令
    $requiredPatterns = @(
        "\.section \.text",
        "\.globl main",
        "main:",
        "pushq\s+%rbp",
        "movq\s+%rsp, %rbp",
        "ret"
    )
    
    foreach ($pattern in $requiredPatterns) {
        if ($content -notmatch $pattern) {
            Write-Host "❌ Missing pattern '$pattern' in $($file.Name)" -ForegroundColor Red
            $qualityPassed = $false
        }
    }
}

if ($qualityPassed) {
    Write-Host "✅ Assembly code quality check passed" -ForegroundColor Green
    $script:testsPassed++
} else {
    Write-Host "❌ Assembly code quality check failed" -ForegroundColor Red
}

# 性能测试
Write-Host "`nPerformance Tests:" -ForegroundColor Cyan
$script:testsRun++

$startTime = Get-Date
.\demo_compiler.ps1 -InputFile "examples\parallel.c" -OpenMP -Vectorize -OptLevel 3 | Out-Null
$endTime = Get-Date
$compilationTime = ($endTime - $startTime).TotalMilliseconds

if ($compilationTime -lt 5000) {  # 5秒内完成
    Write-Host "✅ Compilation performance acceptable ($([math]::Round($compilationTime, 0))ms)" -ForegroundColor Green
    $script:testsPassed++
} else {
    Write-Host "❌ Compilation too slow ($([math]::Round($compilationTime, 0))ms)" -ForegroundColor Red
}

# 文件大小检查
Write-Host "`nOutput Size Analysis:" -ForegroundColor Cyan
$assemblyFiles = Get-ChildItem -Path "examples" -Filter "*.s"
foreach ($file in $assemblyFiles) {
    $lineCount = (Get-Content $file.FullName | Measure-Object -Line).Lines
    $size = (Get-Item $file.FullName).Length
    Write-Host "  $($file.Name): $lineCount lines, $size bytes" -ForegroundColor Gray
}

# 清理生成的文件
Write-Host "`nCleaning up..." -ForegroundColor Yellow
Get-ChildItem -Path "examples" -Filter "*.s" | Remove-Item -Force
Write-Host "Cleanup completed" -ForegroundColor Gray

# 总结
Write-Host ""
Write-Host "Test Summary" -ForegroundColor Yellow
Write-Host "============" -ForegroundColor Yellow
Write-Host "Tests run:    $testsRun"
Write-Host "Tests passed: $testsPassed"
Write-Host "Tests failed: $($testsRun - $testsPassed)"
Write-Host "Success rate: $([math]::Round(($testsPassed / $testsRun) * 100, 1))%"

if ($testsPassed -eq $testsRun) {
    Write-Host ""
    Write-Host "🎉 ALL TESTS PASSED! 🎉" -ForegroundColor Green
    Write-Host "X86/X64 Parallel C Compiler is working correctly!" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "❌ Some tests failed. Please check the output above." -ForegroundColor Red
    exit 1
}
