/*
 * 基础 C 语言功能示例
 * 演示基本语法、函数定义、控制流等
 */

// 全局变量
int global_counter = 0;
float pi = 3.14159;

// 函数声明
int factorial(int n);
float circle_area(float radius);
void print_array(int *arr, int size);

// 主函数
int main() {
    // 变量声明和初始化
    int a = 10;
    int b = 20;
    int result;
    
    // 基本运算
    result = a + b;
    printf("Sum: %d\n", result);
    
    result = a * b;
    printf("Product: %d\n", result);
    
    // 条件语句
    if (a < b) {
        printf("a is less than b\n");
    } else {
        printf("a is greater than or equal to b\n");
    }
    
    // 循环语句
    for (int i = 1; i <= 5; i++) {
        printf("Factorial of %d: %d\n", i, factorial(i));
    }
    
    // 数组操作
    int numbers[5] = {1, 2, 3, 4, 5};
    print_array(numbers, 5);
    
    // 浮点运算
    float radius = 5.0;
    float area = circle_area(radius);
    printf("Circle area (radius=%.1f): %.2f\n", radius, area);
    
    // while 循环
    int count = 0;
    while (count < 3) {
        global_counter++;
        count++;
    }
    printf("Global counter: %d\n", global_counter);
    
    return 0;
}

// 阶乘函数（递归）
int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

// 圆面积计算
float circle_area(float radius) {
    return pi * radius * radius;
}

// 打印数组
void print_array(int *arr, int size) {
    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 指针操作示例
void pointer_example() {
    int x = 42;
    int *ptr = &x;
    
    printf("Value: %d\n", x);
    printf("Address: %p\n", (void*)&x);
    printf("Pointer value: %d\n", *ptr);
    
    *ptr = 100;
    printf("Modified value: %d\n", x);
}

// 结构体示例
struct Point {
    float x;
    float y;
};

float distance(struct Point p1, struct Point p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return sqrt(dx * dx + dy * dy);
}
