// 数组和指针示例
// 演示数组操作和指针使用

void print_array(int *arr, int size) {
    printf("数组内容: ");
    for (int i = 0; i < size; i = i + 1) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void bubble_sort(int *arr, int size) {
    for (int i = 0; i < size - 1; i = i + 1) {
        for (int j = 0; j < size - 1 - i; j = j + 1) {
            if (arr[j] > arr[j + 1]) {
                // 交换元素
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int find_max(int *arr, int size) {
    int max = arr[0];
    for (int i = 1; i < size; i = i + 1) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

int main() {
    printf("数组和指针示例程序\n\n");
    
    // 数组声明和初始化
    int numbers[10];
    
    // 填充数组
    printf("填充数组:\n");
    for (int i = 0; i < 10; i = i + 1) {
        numbers[i] = (i + 1) * (i + 1); // 平方数
    }
    print_array(numbers, 10);
    
    // 数组求和
    int sum = 0;
    for (int i = 0; i < 10; i = i + 1) {
        sum = sum + numbers[i];
    }
    printf("数组元素和: %d\n", sum);
    
    // 查找最大值
    int max = find_max(numbers, 10);
    printf("最大值: %d\n", max);
    
    // 创建另一个数组进行排序
    int unsorted[8] = {64, 34, 25, 12, 22, 11, 90, 88};
    
    printf("\n排序前的数组:\n");
    print_array(unsorted, 8);
    
    bubble_sort(unsorted, 8);
    
    printf("冒泡排序后的数组:\n");
    print_array(unsorted, 8);
    
    // 指针示例
    int value = 42;
    int *ptr = &value;
    
    printf("\n指针示例:\n");
    printf("变量值: %d\n", value);
    printf("变量地址: %p\n", (void*)&value);
    printf("指针值 (地址): %p\n", (void*)ptr);
    printf("指针指向的值: %d\n", *ptr);
    
    // 通过指针修改值
    *ptr = 100;
    printf("通过指针修改后的值: %d\n", value);
    
    // 指针算术
    int arr[5] = {1, 2, 3, 4, 5};
    int *p = arr; // 指向数组第一个元素
    
    printf("\n指针算术示例:\n");
    for (int i = 0; i < 5; i = i + 1) {
        printf("arr[%d] = %d, *(p + %d) = %d\n", i, arr[i], i, *(p + i));
    }
    
    return 0;
}
