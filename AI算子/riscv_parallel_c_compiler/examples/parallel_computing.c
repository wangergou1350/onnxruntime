// 并行计算示例
// 演示RISC-V并行C编译器的并行计算扩展功能

#include <stdio.h>

// 全局共享数据
atomic int shared_counter = 0;
int parallel_array[1000];

// 并行矩阵乘法示例
void parallel_matrix_multiply(int *a, int *b, int *c, int size) {
    parallel_for (int i = 0; i < size; i = i + 1) {
        for (int j = 0; j < size; j = j + 1) {
            int sum = 0;
            for (int k = 0; k < size; k = k + 1) {
                sum = sum + a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = sum;
        }
    }
}

// 并行数组初始化
void parallel_array_init(int *arr, int size, int value) {
    parallel_for (int i = 0; i < size; i = i + 1) {
        arr[i] = value + i;
    }
}

// 并行归约求和
int parallel_sum(int *arr, int size) {
    atomic int total = 0;
    
    parallel_for (int i = 0; i < size; i = i + 1) {
        critical {
            total = total + arr[i];
        }
    }
    
    return total;
}

// 并行快速排序 (简化版)
void parallel_quicksort(int *arr, int left, int right) {
    if (left < right) {
        int pivot = partition(arr, left, right);
        
        // 并行处理两个子数组
        parallel_for (int thread_id = 0; thread_id < 2; thread_id = thread_id + 1) {
            if (thread_id == 0) {
                parallel_quicksort(arr, left, pivot - 1);
            } else {
                parallel_quicksort(arr, pivot + 1, right);
            }
        }
    }
}

int partition(int *arr, int left, int right) {
    int pivot = arr[right];
    int i = left - 1;
    
    for (int j = left; j < right; j = j + 1) {
        if (arr[j] <= pivot) {
            i = i + 1;
            // 交换 arr[i] 和 arr[j]
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    
    // 交换 arr[i+1] 和 arr[right]
    int temp = arr[i + 1];
    arr[i + 1] = arr[right];
    arr[right] = temp;
    
    return i + 1;
}

// 并行斐波那契计算
int parallel_fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    
    int f1, f2;
    
    parallel_for (int thread_id = 0; thread_id < 2; thread_id = thread_id + 1) {
        if (thread_id == 0) {
            f1 = parallel_fibonacci(n - 1);
        } else {
            f2 = parallel_fibonacci(n - 2);
        }
    }
    
    barrier; // 等待两个线程完成
    
    return f1 + f2;
}

// 生产者-消费者模式示例
thread_local int thread_data[100];

void producer_consumer_example() {
    atomic int buffer[10];
    atomic int buffer_count = 0;
    
    printf("生产者-消费者模式示例\n");
    
    // 生产者线程
    parallel_for (int producer_id = 0; producer_id < 2; producer_id = producer_id + 1) {
        for (int i = 0; i < 5; i = i + 1) {
            critical {
                if (buffer_count < 10) {
                    buffer[buffer_count] = producer_id * 10 + i;
                    buffer_count = buffer_count + 1;
                    printf("生产者 %d 生产了: %d\n", producer_id, buffer[buffer_count - 1]);
                }
            }
        }
    }
    
    barrier; // 等待生产完成
    
    // 消费者线程
    parallel_for (int consumer_id = 0; consumer_id < 2; consumer_id = consumer_id + 1) {
        while (buffer_count > 0) {
            critical {
                if (buffer_count > 0) {
                    buffer_count = buffer_count - 1;
                    int item = buffer[buffer_count];
                    printf("消费者 %d 消费了: %d\n", consumer_id, item);
                }
            }
        }
    }
}

int main() {
    printf("RISC-V并行计算示例程序\n\n");
    
    // 1. 并行数组初始化
    printf("1. 并行数组初始化\n");
    parallel_array_init(parallel_array, 1000, 1);
    
    // 验证前10个元素
    printf("数组前10个元素: ");
    for (int i = 0; i < 10; i = i + 1) {
        printf("%d ", parallel_array[i]);
    }
    printf("\n\n");
    
    // 2. 并行求和
    printf("2. 并行归约求和\n");
    int total = parallel_sum(parallel_array, 1000);
    printf("数组元素总和: %d\n\n", total);
    
    // 3. 并行矩阵乘法
    printf("3. 并行矩阵乘法 (4x4)\n");
    int matrix_a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int matrix_b[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; // 单位矩阵
    int matrix_c[16];
    
    parallel_matrix_multiply(matrix_a, matrix_b, matrix_c, 4);
    
    printf("结果矩阵:\n");
    for (int i = 0; i < 4; i = i + 1) {
        for (int j = 0; j < 4; j = j + 1) {
            printf("%d ", matrix_c[i * 4 + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    // 4. 并行斐波那契
    printf("4. 并行斐波那契计算\n");
    int fib_n = 10;
    int fib_result = parallel_fibonacci(fib_n);
    printf("fibonacci(%d) = %d\n\n", fib_n, fib_result);
    
    // 5. 原子操作示例
    printf("5. 原子操作示例\n");
    parallel_for (int thread_id = 0; thread_id < 10; thread_id = thread_id + 1) {
        for (int i = 0; i < 100; i = i + 1) {
            shared_counter = shared_counter + 1; // 原子递增
        }
    }
    printf("原子计数器最终值: %d (期望值: 1000)\n\n", shared_counter);
    
    // 6. 生产者消费者示例
    printf("6. 生产者-消费者模式\n");
    producer_consumer_example();
    
    // 7. 线程局部存储示例
    printf("\n7. 线程局部存储示例\n");
    parallel_for (int thread_id = 0; thread_id < 4; thread_id = thread_id + 1) {
        // 每个线程有自己的thread_data副本
        for (int i = 0; i < 10; i = i + 1) {
            thread_data[i] = thread_id * 10 + i;
        }
        
        printf("线程 %d 的局部数据前5个元素: ", thread_id);
        for (int i = 0; i < 5; i = i + 1) {
            printf("%d ", thread_data[i]);
        }
        printf("\n");
    }
    
    barrier; // 确保所有线程完成
    
    printf("\n并行计算示例程序执行完毕！\n");
    return 0;
}
