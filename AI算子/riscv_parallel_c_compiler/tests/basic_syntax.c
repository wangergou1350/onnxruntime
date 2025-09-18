// 基本语法测试
// 测试编译器的基本C语言语法支持

int global_var = 100;

int test_arithmetic() {
    int a = 10;
    int b = 5;
    
    int add = a + b;
    int sub = a - b;
    int mul = a * b;
    int div = a / b;
    int mod = a % b;
    
    return add + sub + mul + div + mod;
}

int test_conditions() {
    int x = 15;
    int result = 0;
    
    if (x > 10) {
        result = 1;
    } else {
        result = 0;
    }
    
    return result;
}

int test_loops() {
    int sum = 0;
    
    for (int i = 1; i <= 10; i = i + 1) {
        sum = sum + i;
    }
    
    return sum;
}

int main() {
    int arith_result = test_arithmetic();
    int cond_result = test_conditions();
    int loop_result = test_loops();
    
    return arith_result + cond_result + loop_result;
}
