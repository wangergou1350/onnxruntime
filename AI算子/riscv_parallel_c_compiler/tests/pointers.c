// 指针测试
// 测试指针操作和指针算术

int test_basic_pointers() {
    int value = 42;
    int *ptr = &value;
    
    *ptr = 100;
    
    return value; // 应该返回100
}

int test_pointer_arithmetic() {
    int arr[5] = {1, 2, 3, 4, 5};
    int *p = arr;
    
    int sum = 0;
    for (int i = 0; i < 5; i = i + 1) {
        sum = sum + *(p + i);
    }
    
    return sum; // 应该返回15
}

void modify_through_pointer(int *ptr) {
    *ptr = *ptr * 2;
}

int test_function_pointers() {
    int x = 10;
    modify_through_pointer(&x);
    return x; // 应该返回20
}

int test_pointer_to_pointer() {
    int value = 5;
    int *ptr1 = &value;
    int **ptr2 = &ptr1;
    
    **ptr2 = 25;
    
    return value; // 应该返回25
}

int main() {
    int result1 = test_basic_pointers();
    int result2 = test_pointer_arithmetic();
    int result3 = test_function_pointers();
    int result4 = test_pointer_to_pointer();
    
    return result1 + result2 + result3 + result4;
}
