/*
 * 并行计算示例程序
 * 演示 X86/X64 并行 C 编译器的并行处理能力
 */

#include <stdio.h>
#include <stdlib.h>

// 全局数组用于并行处理
int large_array[1000];
atomic int atomic_counter = 0;

// 并行矩阵乘法
void parallel_matrix_multiply() {
    int matrix_a[100][100];
    int matrix_b[100][100];
    int result[100][100];
    
    // 初始化矩阵
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            matrix_a[i][j] = i + j;
            matrix_b[i][j] = i * j;
            result[i][j] = 0;
        }
    }
    
    // 并行矩阵乘法
    parallel_for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            for (int k = 0; k < 100; k++) {
                result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
    
    printf("Matrix multiplication completed in parallel\n");
}

// 并行数组求和
int parallel_array_sum() {
    // 初始化数组
    for (int i = 0; i < 1000; i++) {
        large_array[i] = i + 1;
    }
    
    atomic int total_sum = 0;
    
    // 并行计算数组和
    parallel_for (int i = 0; i < 1000; i++) {
        atomic_add(&total_sum, large_array[i]);
    }
    
    return total_sum;
}

// 原子操作示例
void atomic_operations_demo() {
    atomic int shared_counter = 0;
    
    printf("Atomic operations demo:\n");
    
    // 并行递增计数器
    parallel_for (int i = 0; i < 100; i++) {
        atomic_inc(&shared_counter);
        atomic_inc(&atomic_counter);
    }
    
    printf("Shared counter: %d\n", shared_counter);
    printf("Global atomic counter: %d\n", atomic_counter);
    
    // 原子交换
    int old_value = atomic_exchange(&shared_counter, 1000);
    printf("Old value: %d, New value: %d\n", old_value, shared_counter);
    
    // 比较并交换
    int expected = 1000;
    int new_value = 2000;
    bool swapped = atomic_compare_exchange(&shared_counter, &expected, new_value);
    printf("Compare and swap %s\n", swapped ? "succeeded" : "failed");
    printf("Final value: %d\n", shared_counter);
}

// 临界区示例
void critical_section_demo() {
    int shared_resource = 0;
    
    printf("Critical section demo:\n");
    
    parallel_for (int i = 0; i < 50; i++) {
        critical {
            // 临界区 - 只有一个线程可以执行
            int temp = shared_resource;
            temp += i;
            shared_resource = temp;
        }
    }
    
    printf("Shared resource value: %d\n", shared_resource);
}

// 内存屏障示例
void memory_barrier_demo() {
    volatile int flag = 0;
    volatile int data = 0;
    
    printf("Memory barrier demo:\n");
    
    // 模拟生产者-消费者模式
    parallel_for (int i = 0; i < 2; i++) {
        if (i == 0) {
            // 生产者
            data = 42;
            memory_barrier();  // 确保数据写入在标志设置之前完成
            flag = 1;
            printf("Producer: data written, flag set\n");
        } else {
            // 消费者
            while (flag == 0) {
                // 等待标志
            }
            memory_barrier();  // 确保读取数据之前标志已经被读取
            printf("Consumer: received data = %d\n", data);
        }
    }
}

// 向量化计算示例
void vectorized_computation() {
    float vector_a[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                          9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    float vector_b[16] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    float result[16];
    
    printf("Vectorized computation:\n");
    
    // 向量化加法 (使用 SSE/AVX)
    for (int i = 0; i < 16; i++) {
        result[i] = vector_a[i] + vector_b[i];
    }
    
    printf("Vector addition results: ");
    for (int i = 0; i < 16; i++) {
        printf("%.1f ", result[i]);
    }
    printf("\n");
    
    // 向量化乘法
    for (int i = 0; i < 16; i++) {
        result[i] = vector_a[i] * vector_b[i];
    }
    
    printf("Vector multiplication results: ");
    for (int i = 0; i < 16; i++) {
        printf("%.1f ", result[i]);
    }
    printf("\n");
}

// 并行归约示例
float parallel_reduction() {
    float numbers[1000];
    
    // 初始化数据
    for (int i = 0; i < 1000; i++) {
        numbers[i] = (float)(i + 1) / 10.0;
    }
    
    float sum = 0.0;
    
    // 并行归约求和
    parallel_for (int i = 0; i < 1000; i++) {
        atomic_add_float(&sum, numbers[i]);
    }
    
    return sum;
}

// 主函数
int main() {
    printf("X86/X64 Parallel C Compiler - Parallel Computing Demo\n");
    printf("=====================================================\n\n");
    
    // 矩阵乘法
    printf("1. Parallel Matrix Multiplication:\n");
    parallel_matrix_multiply();
    printf("\n");
    
    // 数组求和
    printf("2. Parallel Array Sum:\n");
    int sum = parallel_array_sum();
    printf("Array sum (1 to 1000): %d\n\n", sum);
    
    // 原子操作
    printf("3. Atomic Operations:\n");
    atomic_operations_demo();
    printf("\n");
    
    // 临界区
    printf("4. Critical Sections:\n");
    critical_section_demo();
    printf("\n");
    
    // 内存屏障
    printf("5. Memory Barriers:\n");
    memory_barrier_demo();
    printf("\n");
    
    // 向量化计算
    printf("6. Vectorized Computation:\n");
    vectorized_computation();
    printf("\n");
    
    // 并行归约
    printf("7. Parallel Reduction:\n");
    float reduction_result = parallel_reduction();
    printf("Parallel reduction sum: %.2f\n\n", reduction_result);
    
    printf("All parallel computing demos completed successfully!\n");
    
    return 0;
}
