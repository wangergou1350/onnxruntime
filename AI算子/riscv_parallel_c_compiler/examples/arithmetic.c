// 算术运算示例
// 演示各种算术运算和表达式

int calculate(int a, int b) {
    int sum = a + b;
    int diff = a - b;
    int product = a * b;
    int quotient = a / b;
    int remainder = a % b;
    
    printf("算术运算结果:\n");
    printf("%d + %d = %d\n", a, b, sum);
    printf("%d - %d = %d\n", a, b, diff);
    printf("%d * %d = %d\n", a, b, product);
    printf("%d / %d = %d\n", a, b, quotient);
    printf("%d %% %d = %d\n", a, b, remainder);
    
    return sum;
}

int main() {
    int x = 20;
    int y = 6;
    
    int result = calculate(x, y);
    printf("函数返回值: %d\n", result);
    
    // 位运算
    int bitwise_and = x & y;
    int bitwise_or = x | y;
    int bitwise_xor = x ^ y;
    int left_shift = x << 2;
    int right_shift = x >> 1;
    
    printf("\n位运算结果:\n");
    printf("%d & %d = %d\n", x, y, bitwise_and);
    printf("%d | %d = %d\n", x, y, bitwise_or);
    printf("%d ^ %d = %d\n", x, y, bitwise_xor);
    printf("%d << 2 = %d\n", x, left_shift);
    printf("%d >> 1 = %d\n", x, right_shift);
    
    return 0;
}
