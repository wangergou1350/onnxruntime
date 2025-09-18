// 函数测试
// 测试函数定义、调用和递归

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 5;
    int y = 3;
    
    int sum = add(x, y);
    int product = multiply(x, y);
    int fact5 = factorial(5);
    int fib7 = fibonacci(7);
    
    swap(&x, &y);
    
    return sum + product + fact5 + fib7 + x + y;
}
