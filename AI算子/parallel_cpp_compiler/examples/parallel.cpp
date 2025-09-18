// Parallel computing example with C++ extensions
#include <iostream>

class Matrix {
private:
    float* data;
    int rows;
    int cols;
    
public:
    Matrix(int r, int c) {
        rows = r;
        cols = c;
        data = new float[rows * cols];
        
        // Initialize to zero
        for (int i = 0; i < rows * cols; i++) {
            data[i] = 0.0f;
        }
    }
    
    ~Matrix() {
        delete[] data;
    }
    
    float& operator()(int row, int col) {
        return data[row * cols + col];
    }
    
    const float& operator()(int row, int col) const {
        return data[row * cols + col];
    }
    
    // Parallel matrix multiplication
    Matrix* multiply(const Matrix& other) {
        if (cols != other.rows) {
            return nullptr;
        }
        
        Matrix* result = new Matrix(rows, other.cols);
        
        parallel_for(0, rows, [&](int i) {
            for (int j = 0; j < other.cols; j++) {
                float sum = 0.0f;
                for (int k = 0; k < cols; k++) {
                    sum += (*this)(i, k) * other(k, j);
                }
                (*result)(i, j) = sum;
            }
        });
        
        return result;
    }
    
    // Parallel matrix addition
    Matrix* add(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            return nullptr;
        }
        
        Matrix* result = new Matrix(rows, cols);
        
        parallel_for(0, rows * cols, [&](int i) {
            result->data[i] = data[i] + other.data[i];
        });
        
        return result;
    }
};

parallel_class ParallelSorter {
private:
    atomic_int comparisons;
    
public:
    ParallelSorter() {
        atomic_store(&comparisons, 0);
    }
    
    thread_safe void quickSort(int* array, int low, int high) {
        if (low < high) {
            int pi = partition(array, low, high);
            
            // Parallel recursive calls
            parallel_for(0, 2, [&](int i) {
                if (i == 0) {
                    quickSort(array, low, pi - 1);
                } else {
                    quickSort(array, pi + 1, high);
                }
            });
        }
    }
    
private:
    thread_safe int partition(int* array, int low, int high) {
        int pivot = array[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            atomic_fetch_add(&comparisons, 1);
            if (array[j] < pivot) {
                i++;
                swap(array, i, j);
            }
        }
        swap(array, i + 1, high);
        return i + 1;
    }
    
    void swap(int* array, int a, int b) {
        int temp = array[a];
        array[a] = array[b];
        array[b] = temp;
    }
    
public:
    int getComparisons() {
        return atomic_load(&comparisons);
    }
};

int main() {
    // Test parallel matrix operations
    Matrix* m1 = new Matrix(100, 100);
    Matrix* m2 = new Matrix(100, 100);
    
    // Initialize matrices with some values
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            (*m1)(i, j) = i + j;
            (*m2)(i, j) = i - j;
        }
    }
    
    // Parallel matrix multiplication
    Matrix* result = m1->multiply(*m2);
    
    // Parallel matrix addition
    Matrix* sum = m1->add(*m2);
    
    // Test parallel sorting
    ParallelSorter* sorter = new ParallelSorter();
    int data[10000];
    
    // Initialize with random-like data
    for (int i = 0; i < 10000; i++) {
        data[i] = 10000 - i;
    }
    
    sorter->quickSort(data, 0, 9999);
    int comparisons = sorter->getComparisons();
    
    // Cleanup
    delete m1;
    delete m2;
    delete result;
    delete sum;
    delete sorter;
    
    return 0;
}
