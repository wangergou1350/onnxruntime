#!/bin/bash

# Test script for ParallelC compiler
# This script tests the compiler with various input programs

set -e  # Exit on any error

echo "ParallelC Compiler Test Suite"
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
tests_run=0
tests_passed=0

# Function to run a test
run_test() {
    local test_name="$1"
    local input_file="$2"
    local expected_output="$3"
    
    echo -e "\n${YELLOW}Running test: $test_name${NC}"
    tests_run=$((tests_run + 1))
    
    # Compile the ParallelC program
    echo "Compiling $input_file..."
    if ./pcc "$input_file" -o "test_output.c"; then
        echo -e "${GREEN}Compilation successful${NC}"
        
        # Compile the generated C code
        echo "Compiling generated C code..."
        if gcc -pthread -o "test_program" "test_output.c"; then
            echo -e "${GREEN}C compilation successful${NC}"
            
            # Run the program if no specific output expected
            if [ -z "$expected_output" ]; then
                echo "Running program..."
                if timeout 10s ./test_program; then
                    echo -e "${GREEN}Execution successful${NC}"
                    tests_passed=$((tests_passed + 1))
                else
                    echo -e "${RED}Execution failed or timed out${NC}"
                fi
            else
                # Check specific output
                echo "Running program and checking output..."
                local actual_output=$(timeout 10s ./test_program 2>&1 || echo "EXECUTION_FAILED")
                
                if echo "$actual_output" | grep -q "$expected_output"; then
                    echo -e "${GREEN}Output matches expected result${NC}"
                    tests_passed=$((tests_passed + 1))
                else
                    echo -e "${RED}Output doesn't match expected result${NC}"
                    echo "Expected: $expected_output"
                    echo "Actual: $actual_output"
                fi
            fi
            
            # Clean up
            rm -f test_program
        else
            echo -e "${RED}C compilation failed${NC}"
        fi
        
        # Clean up
        rm -f test_output.c
    else
        echo -e "${RED}ParallelC compilation failed${NC}"
    fi
}

# Function to create a simple test file
create_test_file() {
    local filename="$1"
    local content="$2"
    
    echo "$content" > "$filename"
}

echo -e "\n${YELLOW}Building ParallelC compiler...${NC}"
if make clean && make; then
    echo -e "${GREEN}Compiler built successfully${NC}"
else
    echo -e "${RED}Failed to build compiler${NC}"
    exit 1
fi

# Test 1: Simple arithmetic
echo -e "\n${YELLOW}Test 1: Simple arithmetic${NC}"
create_test_file "test1.pcc" '
int main() {
    int result = 5 + 3 * 2;
    printf("Result: %d\n", result);
    return 0;
}
'
run_test "Simple arithmetic" "test1.pcc" "Result: 11"

# Test 2: Variable declarations and assignments
echo -e "\n${YELLOW}Test 2: Variables and assignments${NC}"
create_test_file "test2.pcc" '
int main() {
    int x = 10;
    int y = 20;
    int sum = x + y;
    printf("Sum: %d\n", sum);
    return 0;
}
'
run_test "Variables and assignments" "test2.pcc" "Sum: 30"

# Test 3: Control flow (if statement)
echo -e "\n${YELLOW}Test 3: If statement${NC}"
create_test_file "test3.pcc" '
int main() {
    int x = 15;
    if (x > 10) {
        printf("x is greater than 10\n");
    } else {
        printf("x is not greater than 10\n");
    }
    return 0;
}
'
run_test "If statement" "test3.pcc" "x is greater than 10"

# Test 4: Loops
echo -e "\n${YELLOW}Test 4: For loop${NC}"
create_test_file "test4.pcc" '
int main() {
    int sum = 0;
    for (int i = 1; i <= 5; i++) {
        sum = sum + i;
    }
    printf("Sum of 1 to 5: %d\n", sum);
    return 0;
}
'
run_test "For loop" "test4.pcc" "Sum of 1 to 5: 15"

# Test 5: Function definitions
echo -e "\n${YELLOW}Test 5: Function definitions${NC}"
create_test_file "test5.pcc" '
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(7, 8);
    printf("7 + 8 = %d\n", result);
    return 0;
}
'
run_test "Function definitions" "test5.pcc" "7 + 8 = 15"

# Test 6: Arrays
echo -e "\n${YELLOW}Test 6: Arrays${NC}"
create_test_file "test6.pcc" '
int main() {
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    printf("arr[1] = %d\n", arr[1]);
    return 0;
}
'
run_test "Arrays" "test6.pcc" "arr[1] = 20"

# Test 7: Parallel for (basic syntax check)
echo -e "\n${YELLOW}Test 7: Parallel for syntax${NC}"
create_test_file "test7.pcc" '
int main() {
    int sum = 0;
    parallel_for(0, 10, {
        atomic_add(&sum, i);
    });
    printf("Parallel sum completed\n");
    return 0;
}
'
run_test "Parallel for syntax" "test7.pcc" "Parallel sum completed"

# Test 8: Thread functions
echo -e "\n${YELLOW}Test 8: Thread functions${NC}"
create_test_file "test8.pcc" '
int main() {
    printf("Thread functions test\n");
    int tid = thread_id();
    int nt = num_threads();
    printf("Thread info available\n");
    return 0;
}
'
run_test "Thread functions" "test8.pcc" "Thread info available"

# Test 9: Example programs
echo -e "\n${YELLOW}Test 9: Demo program${NC}"
if [ -f "examples/demo.pcc" ]; then
    run_test "Demo program" "examples/demo.pcc" "ParallelC Demo Program"
else
    echo -e "${YELLOW}Demo program not found, skipping${NC}"
fi

# Test 10: Advanced example
echo -e "\n${YELLOW}Test 10: Advanced program${NC}"
if [ -f "examples/advanced.pcc" ]; then
    run_test "Advanced program" "examples/advanced.pcc" "Advanced ParallelC Examples"
else
    echo -e "${YELLOW}Advanced program not found, skipping${NC}"
fi

# Test error cases
echo -e "\n${YELLOW}Testing error cases...${NC}"

# Test 11: Syntax error
echo -e "\n${YELLOW}Test 11: Syntax error detection${NC}"
create_test_file "test_error1.pcc" '
int main() {
    int x = 5 +;  // Syntax error
    return 0;
}
'
echo "Testing syntax error detection..."
if ./pcc "test_error1.pcc" -o "test_output.c" 2>/dev/null; then
    echo -e "${RED}Should have failed on syntax error${NC}"
else
    echo -e "${GREEN}Correctly detected syntax error${NC}"
    tests_passed=$((tests_passed + 1))
fi
tests_run=$((tests_run + 1))

# Test 12: Undefined variable
echo -e "\n${YELLOW}Test 12: Undefined variable detection${NC}"
create_test_file "test_error2.pcc" '
int main() {
    int x = undefined_var;  // Semantic error
    return 0;
}
'
echo "Testing undefined variable detection..."
if ./pcc "test_error2.pcc" -o "test_output.c" 2>/dev/null; then
    echo -e "${RED}Should have failed on undefined variable${NC}"
else
    echo -e "${GREEN}Correctly detected undefined variable${NC}"
    tests_passed=$((tests_passed + 1))
fi
tests_run=$((tests_run + 1))

# Clean up test files
rm -f test*.pcc test_error*.pcc test_output.c test_program

# Summary
echo -e "\n${YELLOW}Test Summary${NC}"
echo "============"
echo "Tests run: $tests_run"
echo "Tests passed: $tests_passed"
echo "Tests failed: $((tests_run - tests_passed))"

if [ $tests_passed -eq $tests_run ]; then
    echo -e "\n${GREEN}All tests passed! 🎉${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Please check the output above.${NC}"
    exit 1
fi
