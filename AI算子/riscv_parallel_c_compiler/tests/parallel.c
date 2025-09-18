// 并行计算测试
// 测试parallel_for和原子操作

int global_counter = 0;

void parallel_test_basic() {
    int local_sum = 0;
    
    parallel_for (int i = 0; i < 10; i = i + 1) {
        local_sum = local_sum + i;
    }
}

void parallel_test_atomic() {
    atomic_int counter = 0;
    
    parallel_for (int i = 0; i < 100; i = i + 1) {
        atomic_add(&counter, 1);
    }
    
    global_counter = atomic_load(&counter);
}

void parallel_test_critical() {
    int shared_data = 0;
    
    parallel_for (int i = 0; i < 50; i = i + 1) {
        critical {
            shared_data = shared_data + 1;
        }
    }
    
    global_counter = shared_data;
}

thread_local int tls_data = 0;

void parallel_test_thread_local() {
    parallel_for (int i = 0; i < 10; i = i + 1) {
        tls_data = i * i;
    }
}

void parallel_test_barrier() {
    int data = 0;
    
    parallel_for (int i = 0; i < 8; i = i + 1) {
        data = data + i;
        barrier();
        data = data * 2;
    }
    
    global_counter = data;
}

int main() {
    parallel_test_basic();
    parallel_test_atomic();
    parallel_test_critical();
    parallel_test_thread_local();
    parallel_test_barrier();
    
    return global_counter;
}
