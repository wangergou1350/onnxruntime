// 结构体和复杂数据类型测试
// 测试结构体定义、成员访问和复杂类型操作

struct Point {
    int x;
    int y;
};

struct Rectangle {
    struct Point top_left;
    struct Point bottom_right;
};

int test_basic_struct() {
    struct Point p;
    p.x = 10;
    p.y = 20;
    
    return p.x + p.y; // 应该返回30
}

int test_nested_struct() {
    struct Rectangle rect;
    rect.top_left.x = 0;
    rect.top_left.y = 0;
    rect.bottom_right.x = 10;
    rect.bottom_right.y = 5;
    
    int width = rect.bottom_right.x - rect.top_left.x;
    int height = rect.bottom_right.y - rect.top_left.y;
    
    return width * height; // 应该返回50
}

struct Point create_point(int x, int y) {
    struct Point p;
    p.x = x;
    p.y = y;
    return p;
}

int test_struct_function() {
    struct Point p1 = create_point(3, 4);
    struct Point p2 = create_point(6, 8);
    
    return (p1.x + p2.x) * (p1.y + p2.y); // (3+6) * (4+8) = 9 * 12 = 108
}

void modify_point(struct Point *p) {
    p->x = p->x * 2;
    p->y = p->y * 2;
}

int test_struct_pointer() {
    struct Point p = create_point(5, 10);
    modify_point(&p);
    
    return p.x + p.y; // (5*2) + (10*2) = 10 + 20 = 30
}

int test_struct_array() {
    struct Point points[3];
    
    points[0] = create_point(1, 1);
    points[1] = create_point(2, 2);
    points[2] = create_point(3, 3);
    
    int sum = 0;
    for (int i = 0; i < 3; i = i + 1) {
        sum = sum + points[i].x + points[i].y;
    }
    
    return sum; // (1+1) + (2+2) + (3+3) = 2 + 4 + 6 = 12
}

int main() {
    int result1 = test_basic_struct();
    int result2 = test_nested_struct();
    int result3 = test_struct_function();
    int result4 = test_struct_pointer();
    int result5 = test_struct_array();
    
    return result1 + result2 + result3 + result4 + result5;
    // 30 + 50 + 108 + 30 + 12 = 230
}
