// 结构体示例
// 演示结构体定义、使用和操作

struct Point {
    int x;
    int y;
};

struct Rectangle {
    struct Point top_left;
    struct Point bottom_right;
};

struct Student {
    int id;
    char name[50];
    float grade;
};

void print_point(struct Point p) {
    printf("点坐标: (%d, %d)\n", p.x, p.y);
}

int calculate_area(struct Rectangle rect) {
    int width = rect.bottom_right.x - rect.top_left.x;
    int height = rect.bottom_right.y - rect.top_left.y;
    return width * height;
}

void print_student(struct Student *s) {
    printf("学生信息:\n");
    printf("  ID: %d\n", s->id);
    printf("  姓名: %s\n", s->name);
    printf("  成绩: %.2f\n", s->grade);
}

struct Point add_points(struct Point p1, struct Point p2) {
    struct Point result;
    result.x = p1.x + p2.x;
    result.y = p1.y + p2.y;
    return result;
}

int main() {
    printf("结构体示例程序\n\n");
    
    // 创建和使用点结构体
    struct Point p1;
    p1.x = 10;
    p1.y = 20;
    
    struct Point p2 = {30, 40}; // 初始化语法
    
    printf("点1: ");
    print_point(p1);
    printf("点2: ");
    print_point(p2);
    
    // 点的加法
    struct Point sum = add_points(p1, p2);
    printf("点的和: ");
    print_point(sum);
    
    // 矩形示例
    struct Rectangle rect;
    rect.top_left.x = 0;
    rect.top_left.y = 0;
    rect.bottom_right.x = 10;
    rect.bottom_right.y = 5;
    
    printf("\n矩形信息:\n");
    printf("左上角: ");
    print_point(rect.top_left);
    printf("右下角: ");
    print_point(rect.bottom_right);
    
    int area = calculate_area(rect);
    printf("矩形面积: %d\n", area);
    
    // 学生结构体示例
    struct Student students[3];
    
    // 初始化学生信息
    students[0].id = 1001;
    strcpy(students[0].name, "张三");
    students[0].grade = 85.5;
    
    students[1].id = 1002;
    strcpy(students[1].name, "李四");
    students[1].grade = 92.0;
    
    students[2].id = 1003;
    strcpy(students[2].name, "王五");
    students[2].grade = 78.5;
    
    printf("\n学生名单:\n");
    for (int i = 0; i < 3; i = i + 1) {
        print_student(&students[i]);
        printf("\n");
    }
    
    // 计算平均成绩
    float total_grade = 0.0;
    for (int i = 0; i < 3; i = i + 1) {
        total_grade = total_grade + students[i].grade;
    }
    float average = total_grade / 3;
    printf("平均成绩: %.2f\n", average);
    
    // 指向结构体的指针
    struct Point *ptr = &p1;
    printf("\n通过指针访问结构体:\n");
    printf("ptr->x = %d, ptr->y = %d\n", ptr->x, ptr->y);
    
    // 修改结构体成员
    ptr->x = 100;
    ptr->y = 200;
    printf("修改后的点1: ");
    print_point(p1);
    
    return 0;
}
