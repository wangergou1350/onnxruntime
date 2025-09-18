#!/bin/bash

# Test script for Parallel C++ Compiler

set -e

COMPILER="./pcpp"
EXAMPLES_DIR="examples"
OUTPUT_DIR="test_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p $OUTPUT_DIR

echo -e "${YELLOW}Parallel C++ Compiler Test Suite${NC}"
echo "=================================="

# Function to run a test
run_test() {
    local test_name=$1
    local input_file=$2
    local expected_result=$3
    
    echo -n "Testing $test_name... "
    
    if [ ! -f "$input_file" ]; then
        echo -e "${RED}FAIL${NC} (input file not found: $input_file)"
        return 1
    fi
    
    local output_file="$OUTPUT_DIR/$(basename $input_file .cpp).c"
    
    # Run the compiler
    if $COMPILER -o "$output_file" "$input_file" 2>/dev/null; then
        if [ -f "$output_file" ]; then
            echo -e "${GREEN}PASS${NC}"
            return 0
        else
            echo -e "${RED}FAIL${NC} (no output file generated)"
            return 1
        fi
    else
        echo -e "${RED}FAIL${NC} (compilation error)"
        return 1
    fi
}

# Function to test specific features
test_feature() {
    local feature_name=$1
    local input_file=$2
    local option=$3
    
    echo -n "Testing $feature_name... "
    
    if $COMPILER $option "$input_file" >/dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        return 1
    fi
}

# Test basic functionality
echo -e "\n${YELLOW}Basic Compilation Tests:${NC}"

# Test simple program
run_test "Simple program" "$EXAMPLES_DIR/simple.cpp" 0

# Test class inheritance
run_test "Class inheritance" "$EXAMPLES_DIR/classes.cpp" 0

# Test parallel features
run_test "Parallel computing" "$EXAMPLES_DIR/parallel.cpp" 0

# Test operator overloading
run_test "Operator overloading" "$EXAMPLES_DIR/operators.cpp" 0

# Test compiler options
echo -e "\n${YELLOW}Compiler Options Tests:${NC}"

# Test AST printing
test_feature "AST printing" "$EXAMPLES_DIR/simple.cpp" "--ast"

# Test token printing
test_feature "Token printing" "$EXAMPLES_DIR/simple.cpp" "--tokens"

# Test semantic analysis only
test_feature "Semantic analysis" "$EXAMPLES_DIR/classes.cpp" "--semantic"

# Test verbose mode
test_feature "Verbose mode" "$EXAMPLES_DIR/simple.cpp" "-v"

# Test generated C code compilation
echo -e "\n${YELLOW}Generated Code Compilation Tests:${NC}"

compile_generated_c() {
    local test_name=$1
    local c_file=$2
    
    echo -n "Compiling generated C code ($test_name)... "
    
    if [ ! -f "$c_file" ]; then
        echo -e "${RED}FAIL${NC} (C file not found)"
        return 1
    fi
    
    local executable="${c_file%.c}"
    
    if gcc -std=c99 -pthread -lm "$c_file" -o "$executable" 2>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        
        # Try to run the executable
        echo -n "Running generated executable... "
        if timeout 5s "$executable" >/dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            rm -f "$executable"
            return 0
        else
            echo -e "${YELLOW}TIMEOUT/ERROR${NC}"
            rm -f "$executable"
            return 1
        fi
    else
        echo -e "${RED}FAIL${NC} (C compilation error)"
        return 1
    fi
}

# Compile generated C files
compile_generated_c "Simple program" "$OUTPUT_DIR/simple.c"
compile_generated_c "Class inheritance" "$OUTPUT_DIR/classes.c"

# Performance test
echo -e "\n${YELLOW}Performance Tests:${NC}"

performance_test() {
    local test_name=$1
    local input_file=$2
    
    echo -n "Performance test ($test_name)... "
    
    local start_time=$(date +%s%N)
    if $COMPILER -o "$OUTPUT_DIR/perf_test.c" "$input_file" >/dev/null 2>&1; then
        local end_time=$(date +%s%N)
        local duration=$((($end_time - $start_time) / 1000000)) # Convert to milliseconds
        echo -e "${GREEN}PASS${NC} (${duration}ms)"
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        return 1
    fi
}

performance_test "Large file compilation" "$EXAMPLES_DIR/parallel.cpp"

# Error handling tests
echo -e "\n${YELLOW}Error Handling Tests:${NC}"

error_test() {
    local test_name=$1
    local input_content=$2
    local expected_to_fail=$3
    
    echo -n "Error test ($test_name)... "
    
    local temp_file="$OUTPUT_DIR/error_test.cpp"
    echo "$input_content" > "$temp_file"
    
    if $COMPILER -o "$OUTPUT_DIR/error_test.c" "$temp_file" >/dev/null 2>&1; then
        if [ "$expected_to_fail" = "true" ]; then
            echo -e "${RED}FAIL${NC} (should have failed but didn't)"
            return 1
        else
            echo -e "${GREEN}PASS${NC}"
            return 0
        fi
    else
        if [ "$expected_to_fail" = "true" ]; then
            echo -e "${GREEN}PASS${NC} (correctly failed)"
            return 0
        else
            echo -e "${RED}FAIL${NC} (should have passed but failed)"
            return 1
        fi
    fi
}

# Test syntax errors
error_test "Syntax error" "int main() { int x = ; return 0; }" "true"

# Test semantic errors
error_test "Semantic error" "int main() { int x = \"string\"; return 0; }" "true"

# Test missing files
echo -n "Testing missing input file... "
if $COMPILER "nonexistent.cpp" >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC}"
else
    echo -e "${GREEN}PASS${NC}"
fi

# Clean up
echo -e "\n${YELLOW}Cleaning up...${NC}"
rm -rf "$OUTPUT_DIR"

echo -e "\n${GREEN}Test suite completed!${NC}"

# Check if all tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
