// Operator overloading example
class Vector3D {
private:
    float x, y, z;
    
public:
    Vector3D(float x_val, float y_val, float z_val) {
        x = x_val;
        y = y_val;
        z = z_val;
    }
    
    Vector3D() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }
    
    // Operator overloading
    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3D operator*(float scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }
    
    float operator*(const Vector3D& other) const {
        // Dot product
        return x * other.x + y * other.y + z * other.z;
    }
    
    bool operator==(const Vector3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    Vector3D& operator+=(const Vector3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    
    // Access operators
    float& operator[](int index) {
        if (index == 0) return x;
        if (index == 1) return y;
        return z;
    }
    
    const float& operator[](int index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        return z;
    }
    
    // Member functions
    float length() const {
        return sqrt(x * x + y * y + z * z);
    }
    
    Vector3D normalize() const {
        float len = length();
        if (len > 0) {
            return Vector3D(x / len, y / len, z / len);
        }
        return Vector3D(0, 0, 0);
    }
    
    void print() const {
        printf("Vector3D(%.2f, %.2f, %.2f)\n", x, y, z);
    }
};

template<typename T>
class Stack {
private:
    T* data;
    int capacity;
    int top;
    
public:
    Stack(int size) {
        capacity = size;
        data = new T[capacity];
        top = -1;
    }
    
    ~Stack() {
        delete[] data;
    }
    
    void push(const T& item) {
        if (top < capacity - 1) {
            data[++top] = item;
        }
    }
    
    T pop() {
        if (top >= 0) {
            return data[top--];
        }
        return T(); // Default constructor
    }
    
    bool empty() const {
        return top == -1;
    }
    
    int size() const {
        return top + 1;
    }
};

int main() {
    // Test Vector3D operators
    Vector3D v1(1.0f, 2.0f, 3.0f);
    Vector3D v2(4.0f, 5.0f, 6.0f);
    
    Vector3D sum = v1 + v2;
    Vector3D diff = v2 - v1;
    Vector3D scaled = v1 * 2.0f;
    float dot_product = v1 * v2;
    
    sum.print();
    diff.print();
    scaled.print();
    
    // Test array access
    v1[0] = 10.0f;
    float y_component = v1[1];
    
    // Test compound assignment
    v1 += v2;
    v1.print();
    
    // Test comparison
    bool are_equal = (v1 == v2);
    
    // Test template class
    Stack<int>* int_stack = new Stack<int>(10);
    int_stack->push(1);
    int_stack->push(2);
    int_stack->push(3);
    
    while (!int_stack->empty()) {
        int value = int_stack->pop();
        printf("Popped: %d\n", value);
    }
    
    Stack<Vector3D>* vector_stack = new Stack<Vector3D>(5);
    vector_stack->push(Vector3D(1, 2, 3));
    vector_stack->push(Vector3D(4, 5, 6));
    
    while (!vector_stack->empty()) {
        Vector3D vec = vector_stack->pop();
        vec.print();
    }
    
    delete int_stack;
    delete vector_stack;
    
    return 0;
}
