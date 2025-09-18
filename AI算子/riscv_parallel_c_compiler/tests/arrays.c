// 数组测试
// 测试数组声明、初始化和操作

int sum_array(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i = i + 1) {
        sum = sum + arr[i];
    }
    return sum;
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

void reverse_array(int *arr, int size) {
    for (int i = 0; i < size / 2; i = i + 1) {
        int temp = arr[i];
        arr[i] = arr[size - 1 - i];
        arr[size - 1 - i] = temp;
    }
}

int main() {
    int numbers[10];
    
    // 初始化数组
    for (int i = 0; i < 10; i = i + 1) {
        numbers[i] = (i + 1) * 2;
    }
    
    int sum = sum_array(numbers, 10);
    int max = find_max(numbers, 10);
    
    reverse_array(numbers, 10);
    
    // 多维数组测试
    int matrix[3][3];
    for (int i = 0; i < 3; i = i + 1) {
        for (int j = 0; j < 3; j = j + 1) {
            matrix[i][j] = i * 3 + j + 1;
        }
    }
    
    int matrix_sum = 0;
    for (int i = 0; i < 3; i = i + 1) {
        for (int j = 0; j < 3; j = j + 1) {
            matrix_sum = matrix_sum + matrix[i][j];
        }
    }
    
    return sum + max + matrix_sum;
}
