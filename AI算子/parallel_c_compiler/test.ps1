# Test ParallelC Compiler on Windows
# This PowerShell script tests the compiler with various input programs

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

Write-Host "ParallelC Compiler Test Suite" -ForegroundColor Yellow
Write-Host "=============================" -ForegroundColor Yellow

# Test counters
$testsRun = 0
$testsPassed = 0

# Function to run a test
function Run-Test {
    param(
        [string]$TestName,
        [string]$InputFile,
        [string]$ExpectedOutput = ""
    )
    
    Write-Host "`nRunning test: $TestName" -ForegroundColor Cyan
    $script:testsRun++
    
    # Compile the ParallelC program
    Write-Host "Compiling $InputFile..."
    $pccResult = & .\pcc.exe $InputFile -o "test_output.c" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Compilation successful" -ForegroundColor Green
        
        # Compile the generated C code
        Write-Host "Compiling generated C code..."
        $gccResult = & gcc -pthread -o "test_program.exe" "test_output.c" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "C compilation successful" -ForegroundColor Green
            
            # Run the program
            if ([string]::IsNullOrEmpty($ExpectedOutput)) {
                Write-Host "Running program..."
                $timeout = 10
                $job = Start-Job -ScriptBlock { 
                    param($exePath)
                    & $exePath
                } -ArgumentList (Resolve-Path ".\test_program.exe")
                
                if (Wait-Job $job -Timeout $timeout) {
                    $output = Receive-Job $job
                    Remove-Job $job
                    Write-Host "Execution successful" -ForegroundColor Green
                    if ($Verbose) {
                        Write-Host "Output: $output" -ForegroundColor Gray
                    }
                    $script:testsPassed++
                } else {
                    Remove-Job $job -Force
                    Write-Host "Execution failed or timed out" -ForegroundColor Red
                }
            } else {
                # Check specific output
                Write-Host "Running program and checking output..."
                $timeout = 10
                $job = Start-Job -ScriptBlock { 
                    param($exePath)
                    & $exePath 2>&1
                } -ArgumentList (Resolve-Path ".\test_program.exe")
                
                if (Wait-Job $job -Timeout $timeout) {
                    $actualOutput = Receive-Job $job
                    Remove-Job $job
                    
                    if ($actualOutput -match $ExpectedOutput) {
                        Write-Host "Output matches expected result" -ForegroundColor Green
                        $script:testsPassed++
                    } else {
                        Write-Host "Output doesn't match expected result" -ForegroundColor Red
                        Write-Host "Expected: $ExpectedOutput" -ForegroundColor Yellow
                        Write-Host "Actual: $actualOutput" -ForegroundColor Yellow
                    }
                } else {
                    Remove-Job $job -Force
                    Write-Host "Execution failed or timed out" -ForegroundColor Red
                }
            }
            
            # Clean up
            if (Test-Path "test_program.exe") {
                Remove-Item "test_program.exe" -Force
            }
        } else {
            Write-Host "C compilation failed" -ForegroundColor Red
            if ($Verbose) {
                Write-Host "GCC output: $gccResult" -ForegroundColor Gray
            }
        }
        
        # Clean up
        if (Test-Path "test_output.c") {
            Remove-Item "test_output.c" -Force
        }
    } else {
        Write-Host "ParallelC compilation failed" -ForegroundColor Red
        if ($Verbose) {
            Write-Host "PCC output: $pccResult" -ForegroundColor Gray
        }
    }
}

# Function to create a test file
function Create-TestFile {
    param(
        [string]$FileName,
        [string]$Content
    )
    
    $Content | Out-File -FilePath $FileName -Encoding UTF8
}

Write-Host "`nBuilding ParallelC compiler..." -ForegroundColor Cyan
$makeResult = & make clean 2>&1; & make 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compiler built successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to build compiler" -ForegroundColor Red
    Write-Host "Make output: $makeResult" -ForegroundColor Gray
    exit 1
}

# Test 1: Simple arithmetic
Write-Host "`nTest 1: Simple arithmetic" -ForegroundColor Cyan
Create-TestFile "test1.pcc" @'
int main() {
    int result = 5 + 3 * 2;
    printf("Result: %d\n", result);
    return 0;
}
'@
Run-Test "Simple arithmetic" "test1.pcc" "Result: 11"

# Test 2: Variable declarations and assignments
Write-Host "`nTest 2: Variables and assignments" -ForegroundColor Cyan
Create-TestFile "test2.pcc" @'
int main() {
    int x = 10;
    int y = 20;
    int sum = x + y;
    printf("Sum: %d\n", sum);
    return 0;
}
'@
Run-Test "Variables and assignments" "test2.pcc" "Sum: 30"

# Test 3: Control flow (if statement)
Write-Host "`nTest 3: If statement" -ForegroundColor Cyan
Create-TestFile "test3.pcc" @'
int main() {
    int x = 15;
    if (x > 10) {
        printf("x is greater than 10\n");
    } else {
        printf("x is not greater than 10\n");
    }
    return 0;
}
'@
Run-Test "If statement" "test3.pcc" "x is greater than 10"

# Test 4: Loops
Write-Host "`nTest 4: For loop" -ForegroundColor Cyan
Create-TestFile "test4.pcc" @'
int main() {
    int sum = 0;
    for (int i = 1; i <= 5; i++) {
        sum = sum + i;
    }
    printf("Sum of 1 to 5: %d\n", sum);
    return 0;
}
'@
Run-Test "For loop" "test4.pcc" "Sum of 1 to 5: 15"

# Test 5: Function definitions
Write-Host "`nTest 5: Function definitions" -ForegroundColor Cyan
Create-TestFile "test5.pcc" @'
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(7, 8);
    printf("7 + 8 = %d\n", result);
    return 0;
}
'@
Run-Test "Function definitions" "test5.pcc" "7 \+ 8 = 15"

# Test 6: Arrays
Write-Host "`nTest 6: Arrays" -ForegroundColor Cyan
Create-TestFile "test6.pcc" @'
int main() {
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    printf("arr[1] = %d\n", arr[1]);
    return 0;
}
'@
Run-Test "Arrays" "test6.pcc" "arr\[1\] = 20"

# Test 7: Parallel for (basic syntax check)
Write-Host "`nTest 7: Parallel for syntax" -ForegroundColor Cyan
Create-TestFile "test7.pcc" @'
int main() {
    int sum = 0;
    parallel_for(0, 10, {
        atomic_add(&sum, i);
    });
    printf("Parallel sum completed\n");
    return 0;
}
'@
Run-Test "Parallel for syntax" "test7.pcc" "Parallel sum completed"

# Test 8: Thread functions
Write-Host "`nTest 8: Thread functions" -ForegroundColor Cyan
Create-TestFile "test8.pcc" @'
int main() {
    printf("Thread functions test\n");
    int tid = thread_id();
    int nt = num_threads();
    printf("Thread info available\n");
    return 0;
}
'@
Run-Test "Thread functions" "test8.pcc" "Thread info available"

# Test 9: Example programs
Write-Host "`nTest 9: Demo program" -ForegroundColor Cyan
if (Test-Path "examples\demo.pcc") {
    Run-Test "Demo program" "examples\demo.pcc" "ParallelC Demo Program"
} else {
    Write-Host "Demo program not found, skipping" -ForegroundColor Yellow
}

# Test 10: Advanced example
Write-Host "`nTest 10: Advanced program" -ForegroundColor Cyan
if (Test-Path "examples\advanced.pcc") {
    Run-Test "Advanced program" "examples\advanced.pcc" "Advanced ParallelC Examples"
} else {
    Write-Host "Advanced program not found, skipping" -ForegroundColor Yellow
}

# Test error cases
Write-Host "`nTesting error cases..." -ForegroundColor Cyan

# Test 11: Syntax error
Write-Host "`nTest 11: Syntax error detection" -ForegroundColor Cyan
Create-TestFile "test_error1.pcc" @'
int main() {
    int x = 5 +;  // Syntax error
    return 0;
}
'@
Write-Host "Testing syntax error detection..."
$errorResult = & .\pcc.exe "test_error1.pcc" -o "test_output.c" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Should have failed on syntax error" -ForegroundColor Red
} else {
    Write-Host "Correctly detected syntax error" -ForegroundColor Green
    $testsPassed++
}
$testsRun++

# Test 12: Undefined variable
Write-Host "`nTest 12: Undefined variable detection" -ForegroundColor Cyan
Create-TestFile "test_error2.pcc" @'
int main() {
    int x = undefined_var;  // Semantic error
    return 0;
}
'@
Write-Host "Testing undefined variable detection..."
$errorResult = & .\pcc.exe "test_error2.pcc" -o "test_output.c" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Should have failed on undefined variable" -ForegroundColor Red
} else {
    Write-Host "Correctly detected undefined variable" -ForegroundColor Green
    $testsPassed++
}
$testsRun++

# Clean up test files
Get-ChildItem -Path "." -Filter "test*.pcc" | Remove-Item -Force
Get-ChildItem -Path "." -Filter "test_error*.pcc" | Remove-Item -Force
if (Test-Path "test_output.c") { Remove-Item "test_output.c" -Force }
if (Test-Path "test_program.exe") { Remove-Item "test_program.exe" -Force }

# Summary
Write-Host "`nTest Summary" -ForegroundColor Yellow
Write-Host "============" -ForegroundColor Yellow
Write-Host "Tests run: $testsRun"
Write-Host "Tests passed: $testsPassed"
Write-Host "Tests failed: $($testsRun - $testsPassed)"

if ($testsPassed -eq $testsRun) {
    Write-Host "`nAll tests passed! 🎉" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nSome tests failed. Please check the output above." -ForegroundColor Red
    exit 1
}
