// 控制流示例
// 演示if、while、for等控制流语句

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int factorial(int n) {
    int result = 1;
    int i = 1;
    
    while (i <= n) {
        result = result * i;
        i = i + 1;
    }
    
    return result;
}

int main() {
    printf("控制流示例程序\n\n");
    
    // if-else示例
    int num = 15;
    if (num > 10) {
        printf("%d 大于 10\n", num);
    } else {
        printf("%d 小于等于 10\n", num);
    }
    
    // for循环示例
    printf("\n前10个自然数:\n");
    for (int i = 1; i <= 10; i = i + 1) {
        printf("%d ", i);
    }
    printf("\n");
    
    // while循环示例 - 计算阶乘
    printf("\n阶乘计算:\n");
    for (int i = 1; i <= 5; i = i + 1) {
        int fact = factorial(i);
        printf("%d! = %d\n", i, fact);
    }
    
    // 递归示例 - 斐波那契数列
    printf("\n斐波那契数列 (前10项):\n");
    for (int i = 0; i < 10; i = i + 1) {
        int fib = fibonacci(i);
        printf("F(%d) = %d\n", i, fib);
    }
    
    // switch语句示例
    int day = 3;
    printf("\n星期 %d 是: ", day);
    switch (day) {
        case 1:
            printf("星期一\n");
            break;
        case 2:
            printf("星期二\n");
            break;
        case 3:
            printf("星期三\n");
            break;
        case 4:
            printf("星期四\n");
            break;
        case 5:
            printf("星期五\n");
            break;
        default:
            printf("周末\n");
            break;
    }
    
    return 0;
}
