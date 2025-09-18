# X86/X64 Parallel C Compiler Simple Test
param(
    [switch]$Verbose
)

Write-Host "X86/X64 Parallel C Compiler Test" -ForegroundColor Yellow
Write-Host "=================================" -ForegroundColor Yellow

$testsRun = 0
$testsPassed = 0

function Test-Compilation {
    param(
        [string]$TestName,
        [string]$InputFile,
        [string]$Options = ""
    )
    
    Write-Host "`nTest: $TestName" -ForegroundColor Cyan
    $script:testsRun++
    
    $outputFile = [System.IO.Path]::ChangeExtension($InputFile, ".s")
    $cmd = ".\demo_compiler.ps1 -InputFile `"$InputFile`" -OutputFile `"$outputFile`" $Options"
    
    try {
        $result = Invoke-Expression $cmd 2>&1
        
        if (Test-Path $outputFile) {
            Write-Host "PASS: Assembly file generated" -ForegroundColor Green
            $script:testsPassed++
            
            if ($Verbose) {
                $lineCount = (Get-Content $outputFile | Measure-Object -Line).Lines
                Write-Host "Generated $lineCount lines of assembly" -ForegroundColor Gray
            }
        } else {
            Write-Host "FAIL: No assembly file generated" -ForegroundColor Red
        }
    } catch {
        Write-Host "FAIL: Exception occurred" -ForegroundColor Red
    }
}

# Run tests
Test-Compilation "Basic C Code" "examples\basic.c"
Test-Compilation "Basic with Verbose" "examples\basic.c" "-Verbose"
Test-Compilation "O1 Optimization" "examples\basic.c" "-OptLevel 1"
Test-Compilation "O2 Optimization" "examples\basic.c" "-OptLevel 2" 
Test-Compilation "O3 Optimization" "examples\basic.c" "-OptLevel 3"
Test-Compilation "Parallel Code" "examples\parallel.c" "-OpenMP"
Test-Compilation "Vectorization" "examples\parallel.c" "-Vectorize"
Test-Compilation "Full Optimization" "examples\parallel.c" "-OpenMP -Vectorize -OptLevel 3"

# Test help
Write-Host "`nTest: Help Command" -ForegroundColor Cyan
$script:testsRun++
try {
    $helpOutput = .\demo_compiler.ps1 -Help 2>&1
    if ($helpOutput -match "X86/X64 Parallel C Compiler") {
        Write-Host "PASS: Help command works" -ForegroundColor Green
        $script:testsPassed++
    } else {
        Write-Host "FAIL: Help output incorrect" -ForegroundColor Red
    }
} catch {
    Write-Host "FAIL: Help command failed" -ForegroundColor Red
}

# Test error handling
Write-Host "`nTest: Error Handling" -ForegroundColor Cyan
$script:testsRun++
try {
    $errorOutput = .\demo_compiler.ps1 -InputFile "nonexistent.c" 2>&1
    if ($errorOutput -match "not found") {
        Write-Host "PASS: Error handling works" -ForegroundColor Green
        $script:testsPassed++
    } else {
        Write-Host "FAIL: Should fail for missing files" -ForegroundColor Red
    }
} catch {
    Write-Host "PASS: Correctly threw exception" -ForegroundColor Green
    $script:testsPassed++
}

# Cleanup
Get-ChildItem -Path "examples" -Filter "*.s" -ErrorAction SilentlyContinue | Remove-Item -Force

# Summary
Write-Host ""
Write-Host "Test Summary" -ForegroundColor Yellow
Write-Host "============" -ForegroundColor Yellow
Write-Host "Tests run:    $testsRun"
Write-Host "Tests passed: $testsPassed"  
Write-Host "Tests failed: $($testsRun - $testsPassed)"
Write-Host "Success rate: $([math]::Round(($testsPassed / $testsRun) * 100, 1))%"

if ($testsPassed -eq $testsRun) {
    Write-Host ""
    Write-Host "ALL TESTS PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "Some tests failed." -ForegroundColor Red
    exit 1
}
