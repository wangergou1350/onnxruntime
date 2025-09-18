// C++ class example with inheritance and virtual functions
class Animal {
public:
    int age;
    
    Animal(int a) {
        age = a;
    }
    
    virtual void speak() {
        // Base implementation
    }
    
    virtual ~Animal() {
        // Destructor
    }
};

class Dog : public Animal {
private:
    char name[50];
    
public:
    Dog(int a, char* n) : Animal(a) {
        // Copy name
        int i = 0;
        while (n[i] != '\0' && i < 49) {
            name[i] = n[i];
            i++;
        }
        name[i] = '\0';
    }
    
    virtual void speak() {
        // Dog implementation
    }
    
    void wagTail() {
        // Dog-specific method
    }
};

parallel_class ParallelCalculator {
private:
    atomic_int counter;
    
public:
    ParallelCalculator() {
        atomic_store(&counter, 0);
    }
    
    thread_safe void increment() {
        atomic_fetch_add(&counter, 1);
    }
    
    thread_safe int getCount() {
        return atomic_load(&counter);
    }
    
    void parallelSum(int* array, int size, int* result) {
        *result = 0;
        parallel_for(0, size, [&](int i) {
            atomic_fetch_add(result, array[i]);
        });
    }
};

int main() {
    // Test basic class instantiation
    Animal* animal = new Animal(5);
    Dog* dog = new Dog(3, "Rex");
    
    // Test virtual function calls
    animal->speak();
    dog->speak();
    dog->wagTail();
    
    // Test parallel class
    ParallelCalculator* calc = new ParallelCalculator();
    calc->increment();
    calc->increment();
    int count = calc->getCount();
    
    // Test parallel computation
    int numbers[1000];
    for (int i = 0; i < 1000; i++) {
        numbers[i] = i + 1;
    }
    
    int total = 0;
    calc->parallelSum(numbers, 1000, &total);
    
    // Cleanup
    delete animal;
    delete dog;
    delete calc;
    
    return 0;
}
