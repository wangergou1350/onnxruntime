/*
 * X86/X64 并行 C 编译器 - 主头文件
 * 支持 C 核心语法和并行计算扩展
 * 目标: Intel x86/x64 指令集
 */

#ifndef X86_CC_H
#define X86_CC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <ctype.h>
#include <assert.h>

// =============================================================================
// 基础常量定义
// =============================================================================

#define MAX_TOKEN_LENGTH    1024
#define MAX_SYMBOL_NAME     256
#define MAX_SYMBOLS         10000
#define MAX_FUNCTIONS       1000
#define MAX_LOCALS          256
#define MAX_PARAMS          32
#define MAX_EXPR_DEPTH      64
#define MAX_STRING_LITERALS 1000
#define MAX_ERROR_MSG       512

// x86/x64 寄存器数量
#define MAX_X86_REGISTERS   16
#define MAX_X86_XMM_REGS    16
#define MAX_X86_YMM_REGS    16

// =============================================================================
// 数据类型定义
// =============================================================================

// 基本数据类型
typedef enum {
    TYPE_VOID,
    TYPE_CHAR,
    TYPE_INT,
    TYPE_LONG,
    TYPE_FLOAT,
    TYPE_DOUBLE,
    TYPE_POINTER,
    TYPE_ARRAY,
    TYPE_STRUCT,
    TYPE_FUNCTION,
    // 并行计算扩展类型
    TYPE_ATOMIC_INT,
    TYPE_ATOMIC_LONG,
    TYPE_THREAD_LOCAL,
    TYPE_PARALLEL_FOR
} TypeKind;

// 数据类型结构
typedef struct Type {
    TypeKind kind;
    int size;           // 字节大小
    int align;          // 对齐要求
    struct Type *base;  // 指针/数组的基类型
    int array_size;     // 数组大小
    struct Symbol *struct_def; // 结构体定义
    struct Type *return_type;  // 函数返回类型
    struct Type **param_types; // 函数参数类型
    int param_count;    // 参数数量
    bool is_atomic;     // 是否为原子类型
    bool is_thread_local; // 是否为线程局部
} Type;

// =============================================================================
// 词法分析
// =============================================================================

typedef enum {
    // 字面量
    TOKEN_NUMBER,       // 数字
    TOKEN_FLOAT,        // 浮点数
    TOKEN_STRING,       // 字符串
    TOKEN_CHAR,         // 字符
    TOKEN_IDENTIFIER,   // 标识符
    
    // 关键字
    TOKEN_INT,
    TOKEN_FLOAT_KW,     // float 关键字
    TOKEN_DOUBLE,
    TOKEN_CHAR_KW,      // char 关键字
    TOKEN_VOID,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_FOR,
    TOKEN_WHILE,
    TOKEN_RETURN,
    TOKEN_STRUCT,
    TOKEN_SIZEOF,
    
    // 并行计算关键字
    TOKEN_PARALLEL_FOR,
    TOKEN_ATOMIC,
    TOKEN_BARRIER,
    TOKEN_CRITICAL,
    TOKEN_THREAD_LOCAL,
    
    // 运算符
    TOKEN_PLUS,         // +
    TOKEN_MINUS,        // -
    TOKEN_MULTIPLY,     // *
    TOKEN_DIVIDE,       // /
    TOKEN_MODULO,       // %
    TOKEN_ASSIGN,       // =
    TOKEN_PLUS_ASSIGN,  // +=
    TOKEN_MINUS_ASSIGN, // -=
    TOKEN_MUL_ASSIGN,   // *=
    TOKEN_DIV_ASSIGN,   // /=
    
    // 比较运算符
    TOKEN_EQ,           // ==
    TOKEN_NE,           // !=
    TOKEN_LT,           // <
    TOKEN_LE,           // <=
    TOKEN_GT,           // >
    TOKEN_GE,           // >=
    
    // 逻辑运算符
    TOKEN_AND,          // &&
    TOKEN_OR,           // ||
    TOKEN_NOT,          // !
    
    // 位运算符
    TOKEN_BIT_AND,      // &
    TOKEN_BIT_OR,       // |
    TOKEN_BIT_XOR,      // ^
    TOKEN_BIT_NOT,      // ~
    TOKEN_LSHIFT,       // <<
    TOKEN_RSHIFT,       // >>
    
    // 自增自减
    TOKEN_INCREMENT,    // ++
    TOKEN_DECREMENT,    // --
    
    // 标点符号
    TOKEN_SEMICOLON,    // ;
    TOKEN_COMMA,        // ,
    TOKEN_DOT,          // .
    TOKEN_ARROW,        // ->
    TOKEN_LPAREN,       // (
    TOKEN_RPAREN,       // )
    TOKEN_LBRACE,       // {
    TOKEN_RBRACE,       // }
    TOKEN_LBRACKET,     // [
    TOKEN_RBRACKET,     // ]
    TOKEN_QUESTION,     // ?
    TOKEN_COLON,        // :
    
    // 特殊
    TOKEN_EOF,
    TOKEN_NEWLINE,
    TOKEN_ERROR
} TokenType;

typedef struct Token {
    TokenType type;
    char *value;
    int line;
    int column;
    union {
        long int_val;
        double float_val;
        char char_val;
    };
} Token;

// 词法分析器状态
typedef struct Lexer {
    char *source;
    char *current;
    int line;
    int column;
    Token current_token;
    bool has_error;
    char error_msg[MAX_ERROR_MSG];
} Lexer;

// =============================================================================
// 抽象语法树 (AST)
// =============================================================================

typedef enum {
    // 表达式节点
    AST_NUMBER,
    AST_FLOAT,
    AST_STRING,
    AST_CHAR,
    AST_IDENTIFIER,
    AST_BINARY_OP,
    AST_UNARY_OP,
    AST_ASSIGN,
    AST_CALL,
    AST_ARRAY_ACCESS,
    AST_MEMBER_ACCESS,
    AST_CAST,
    AST_SIZEOF,
    AST_CONDITIONAL,
    
    // 语句节点
    AST_EXPRESSION_STMT,
    AST_BLOCK,
    AST_IF,
    AST_FOR,
    AST_WHILE,
    AST_RETURN,
    AST_BREAK,
    AST_CONTINUE,
    
    // 声明节点
    AST_VAR_DECL,
    AST_FUNCTION_DECL,
    AST_STRUCT_DECL,
    AST_PARAM_DECL,
    
    // 并行计算节点
    AST_PARALLEL_FOR,
    AST_ATOMIC_OP,
    AST_BARRIER,
    AST_CRITICAL
} ASTNodeType;

typedef struct ASTNode ASTNode;

struct ASTNode {
    ASTNodeType type;
    int line;
    int column;
    Type *data_type;
    
    union {
        // 字面量
        struct {
            long int_value;
        } number;
        
        struct {
            double float_value;
        } float_literal;
        
        struct {
            char *string_value;
        } string;
        
        struct {
            char char_value;
        } char_literal;
        
        // 标识符
        struct {
            char *name;
            struct Symbol *symbol;
        } identifier;
        
        // 二元运算
        struct {
            TokenType operator;
            ASTNode *left;
            ASTNode *right;
        } binary_op;
        
        // 一元运算
        struct {
            TokenType operator;
            ASTNode *operand;
        } unary_op;
        
        // 赋值
        struct {
            ASTNode *left;
            ASTNode *right;
            TokenType operator; // =, +=, -=, etc.
        } assignment;
        
        // 函数调用
        struct {
            ASTNode *function;
            ASTNode **args;
            int arg_count;
        } call;
        
        // 数组访问
        struct {
            ASTNode *array;
            ASTNode *index;
        } array_access;
        
        // 成员访问
        struct {
            ASTNode *object;
            char *member;
            bool is_pointer; // -> vs .
        } member_access;
        
        // 类型转换
        struct {
            Type *target_type;
            ASTNode *expression;
        } cast;
        
        // sizeof
        struct {
            Type *type;
            ASTNode *expression;
        } sizeof_expr;
        
        // 条件表达式 (? :)
        struct {
            ASTNode *condition;
            ASTNode *true_expr;
            ASTNode *false_expr;
        } conditional;
        
        // 表达式语句
        struct {
            ASTNode *expression;
        } expr_stmt;
        
        // 语句块
        struct {
            ASTNode **statements;
            int stmt_count;
        } block;
        
        // if 语句
        struct {
            ASTNode *condition;
            ASTNode *then_stmt;
            ASTNode *else_stmt;
        } if_stmt;
        
        // for 循环
        struct {
            ASTNode *init;
            ASTNode *condition;
            ASTNode *update;
            ASTNode *body;
        } for_stmt;
        
        // while 循环
        struct {
            ASTNode *condition;
            ASTNode *body;
        } while_stmt;
        
        // return 语句
        struct {
            ASTNode *expression;
        } return_stmt;
        
        // 变量声明
        struct {
            Type *type;
            char *name;
            ASTNode *initializer;
        } var_decl;
        
        // 函数声明
        struct {
            Type *return_type;
            char *name;
            ASTNode **parameters;
            int param_count;
            ASTNode *body;
        } func_decl;
        
        // 结构体声明
        struct {
            char *name;
            ASTNode **members;
            int member_count;
        } struct_decl;
        
        // 参数声明
        struct {
            Type *type;
            char *name;
        } param_decl;
        
        // 并行 for 循环
        struct {
            ASTNode *init;
            ASTNode *condition;
            ASTNode *update;
            ASTNode *body;
            int num_threads;    // 线程数量提示
        } parallel_for;
        
        // 原子操作
        struct {
            TokenType operation;  // add, sub, load, store, etc.
            ASTNode *target;
            ASTNode *value;
        } atomic_op;
        
        // 临界区
        struct {
            ASTNode *body;
        } critical;
    };
};

// =============================================================================
// 符号表
// =============================================================================

typedef enum {
    SYMBOL_VAR,
    SYMBOL_FUNCTION,
    SYMBOL_STRUCT,
    SYMBOL_PARAM,
    SYMBOL_MEMBER
} SymbolKind;

typedef struct Symbol {
    SymbolKind kind;
    char *name;
    Type *type;
    int offset;         // 栈偏移或结构体偏移
    int scope_level;    // 作用域层级
    bool is_global;     // 是否为全局符号
    bool is_used;       // 是否被使用
    
    // 函数特有信息
    struct {
        ASTNode *declaration;
        int local_size;     // 局部变量总大小
        bool is_defined;    // 是否已定义
    } function;
    
    // 结构体特有信息
    struct {
        struct Symbol **members;
        int member_count;
        int size;
        int alignment;
    } struct_info;
} Symbol;

// 符号表
typedef struct SymbolTable {
    Symbol *symbols[MAX_SYMBOLS];
    int count;
    int scope_level;
    struct SymbolTable *parent;
} SymbolTable;

// =============================================================================
// x86/x64 指令和寄存器定义
// =============================================================================

// x86/x64 寄存器
typedef enum {
    // 64位通用寄存器 (x64)
    REG_RAX, REG_RBX, REG_RCX, REG_RDX,
    REG_RSI, REG_RDI, REG_RSP, REG_RBP,
    REG_R8,  REG_R9,  REG_R10, REG_R11,
    REG_R12, REG_R13, REG_R14, REG_R15,
    
    // 32位寄存器 (对应64位寄存器的低32位)
    REG_EAX, REG_EBX, REG_ECX, REG_EDX,
    REG_ESI, REG_EDI, REG_ESP, REG_EBP,
    REG_R8D, REG_R9D, REG_R10D, REG_R11D,
    REG_R12D, REG_R13D, REG_R14D, REG_R15D,
    
    // XMM 寄存器 (SSE)
    REG_XMM0, REG_XMM1, REG_XMM2, REG_XMM3,
    REG_XMM4, REG_XMM5, REG_XMM6, REG_XMM7,
    REG_XMM8, REG_XMM9, REG_XMM10, REG_XMM11,
    REG_XMM12, REG_XMM13, REG_XMM14, REG_XMM15,
    
    // YMM 寄存器 (AVX)
    REG_YMM0, REG_YMM1, REG_YMM2, REG_YMM3,
    REG_YMM4, REG_YMM5, REG_YMM6, REG_YMM7,
    REG_YMM8, REG_YMM9, REG_YMM10, REG_YMM11,
    REG_YMM12, REG_YMM13, REG_YMM14, REG_YMM15,
    
    REG_NONE = -1
} X86Register;

// x86/x64 指令类型
typedef enum {
    // 数据移动
    INST_MOV,     // mov dst, src
    INST_MOVSX,   // movsx (符号扩展)
    INST_MOVZX,   // movzx (零扩展)
    INST_LEA,     // lea (加载有效地址)
    
    // 算术运算
    INST_ADD,     // add
    INST_SUB,     // sub
    INST_IMUL,    // imul (有符号乘法)
    INST_IDIV,    // idiv (有符号除法)
    INST_NEG,     // neg (取负)
    INST_INC,     // inc (自增)
    INST_DEC,     // dec (自减)
    
    // 逻辑运算
    INST_AND,     // and
    INST_OR,      // or
    INST_XOR,     // xor
    INST_NOT,     // not
    INST_SHL,     // shl (左移)
    INST_SHR,     // shr (右移)
    INST_SAR,     // sar (算术右移)
    
    // 比较和测试
    INST_CMP,     // cmp
    INST_TEST,    // test
    
    // 跳转指令
    INST_JMP,     // jmp (无条件跳转)
    INST_JE,      // je (相等跳转)
    INST_JNE,     // jne (不等跳转)
    INST_JL,      // jl (小于跳转)
    INST_JLE,     // jle (小于等于跳转)
    INST_JG,      // jg (大于跳转)
    INST_JGE,     // jge (大于等于跳转)
    INST_JZ,      // jz (零跳转)
    INST_JNZ,     // jnz (非零跳转)
    
    // 栈操作
    INST_PUSH,    // push
    INST_POP,     // pop
    
    // 函数调用
    INST_CALL,    // call
    INST_RET,     // ret
    
    // 浮点运算 (SSE/AVX)
    INST_MOVSS,   // movss (单精度浮点移动)
    INST_MOVSD,   // movsd (双精度浮点移动)
    INST_ADDSS,   // addss (单精度浮点加法)
    INST_ADDSD,   // addsd (双精度浮点加法)
    INST_SUBSS,   // subss (单精度浮点减法)
    INST_SUBSD,   // subsd (双精度浮点减法)
    INST_MULSS,   // mulss (单精度浮点乘法)
    INST_MULSD,   // mulsd (双精度浮点乘法)
    INST_DIVSS,   // divss (单精度浮点除法)
    INST_DIVSD,   // divsd (双精度浮点除法)
    
    // 并行/SIMD 指令
    INST_PADDD,   // paddd (打包双字加法)
    INST_PSUBD,   // psubd (打包双字减法)
    INST_PMULLD,  // pmulld (打包双字乘法)
    INST_MOVAPS,  // movaps (对齐打包单精度移动)
    INST_MOVUPS,  // movups (未对齐打包单精度移动)
    
    // 原子操作
    INST_LOCK,    // lock 前缀
    INST_XCHG,    // xchg (交换)
    INST_CMPXCHG, // cmpxchg (比较交换)
    INST_XADD,    // xadd (交换加法)
    
    // 内存屏障
    INST_MFENCE,  // mfence (内存围栏)
    INST_LFENCE,  // lfence (加载围栏)
    INST_SFENCE,  // sfence (存储围栏)
    
    // 其他
    INST_NOP,     // nop (空操作)
    INST_INT3,    // int3 (断点)
    INST_CDQ,     // cdq (符号扩展)
    INST_CQO      // cqo (64位符号扩展)
} X86Instruction;

// 操作数类型
typedef enum {
    OPERAND_REGISTER,
    OPERAND_IMMEDIATE,
    OPERAND_MEMORY,
    OPERAND_LABEL
} OperandType;

// 操作数
typedef struct Operand {
    OperandType type;
    union {
        X86Register reg;
        long immediate;
        struct {
            X86Register base;
            X86Register index;
            int scale;      // 1, 2, 4, 8
            int displacement;
        } memory;
        char *label;
    };
    int size;  // 操作数大小 (1, 2, 4, 8 字节)
} Operand;

// x86 指令结构
typedef struct X86Inst {
    X86Instruction opcode;
    Operand dst;
    Operand src;
    bool has_dst;
    bool has_src;
    char *comment;  // 注释
} X86Inst;

// =============================================================================
// 寄存器分配
// =============================================================================

typedef struct RegisterAllocator {
    bool general_regs[16];      // 通用寄存器使用状态
    bool xmm_regs[16];          // XMM寄存器使用状态
    bool ymm_regs[16];          // YMM寄存器使用状态
    
    // 寄存器到变量的映射
    Symbol *reg_to_symbol[16];
    
    // 调用者保存寄存器
    X86Register caller_saved[10];
    int caller_saved_count;
    
    // 被调用者保存寄存器
    X86Register callee_saved[10];
    int callee_saved_count;
    
    // 参数寄存器 (System V ABI)
    X86Register param_regs[6];
    int param_reg_count;
    
    // 返回值寄存器
    X86Register return_reg;
    X86Register return_reg_float;
    
    // 栈指针和帧指针
    X86Register stack_pointer;
    X86Register frame_pointer;
} RegisterAllocator;

// =============================================================================
// 代码生成器
// =============================================================================

typedef struct CodeGenerator {
    FILE *output;
    RegisterAllocator *reg_alloc;
    SymbolTable *current_scope;
    
    // 代码生成状态
    int label_counter;
    int temp_var_counter;
    int current_stack_offset;
    
    // 函数信息
    Symbol *current_function;
    int frame_size;
    
    // 并行代码生成
    bool in_parallel_region;
    int thread_count;
    
    // 优化开关
    bool optimize_level;
    bool vectorize_enabled;
    bool parallel_enabled;
    
    // 错误处理
    bool has_error;
    char error_msg[MAX_ERROR_MSG];
} CodeGenerator;

// =============================================================================
// 语法分析器
// =============================================================================

typedef struct Parser {
    Lexer *lexer;
    SymbolTable *global_scope;
    SymbolTable *current_scope;
    
    // 解析状态
    int current_scope_level;
    bool has_error;
    char error_msg[MAX_ERROR_MSG];
    
    // 类型信息
    Type *builtin_types[10];
    int builtin_type_count;
} Parser;

// =============================================================================
// 函数声明
// =============================================================================

// 词法分析器
Lexer *lexer_create(char *source);
void lexer_destroy(Lexer *lexer);
bool lexer_next_token(Lexer *lexer);
void lexer_error(Lexer *lexer, const char *format, ...);

// 语法分析器
Parser *parser_create(Lexer *lexer);
void parser_destroy(Parser *parser);
ASTNode *parser_parse(Parser *parser);
void parser_error(Parser *parser, const char *format, ...);

// AST 节点创建
ASTNode *ast_create_number(long value);
ASTNode *ast_create_float(double value);
ASTNode *ast_create_string(char *value);
ASTNode *ast_create_identifier(char *name);
ASTNode *ast_create_binary_op(TokenType op, ASTNode *left, ASTNode *right);
ASTNode *ast_create_unary_op(TokenType op, ASTNode *operand);
ASTNode *ast_create_assignment(ASTNode *left, ASTNode *right, TokenType op);
ASTNode *ast_create_call(ASTNode *function, ASTNode **args, int arg_count);
ASTNode *ast_create_block(ASTNode **statements, int stmt_count);
ASTNode *ast_create_if(ASTNode *condition, ASTNode *then_stmt, ASTNode *else_stmt);
ASTNode *ast_create_for(ASTNode *init, ASTNode *condition, ASTNode *update, ASTNode *body);
ASTNode *ast_create_while(ASTNode *condition, ASTNode *body);
ASTNode *ast_create_return(ASTNode *expression);
ASTNode *ast_create_var_decl(Type *type, char *name, ASTNode *initializer);
ASTNode *ast_create_function_decl(Type *return_type, char *name, ASTNode **params, int param_count, ASTNode *body);
ASTNode *ast_create_parallel_for(ASTNode *init, ASTNode *condition, ASTNode *update, ASTNode *body, int num_threads);
void ast_destroy(ASTNode *node);

// 类型系统
Type *type_create_basic(TypeKind kind);
Type *type_create_pointer(Type *base);
Type *type_create_array(Type *base, int size);
Type *type_create_function(Type *return_type, Type **param_types, int param_count);
Type *type_create_struct(char *name);
int type_size(Type *type);
int type_alignment(Type *type);
bool type_compatible(Type *a, Type *b);
Type *type_promote(Type *type);
void type_destroy(Type *type);

// 符号表
SymbolTable *symbol_table_create(SymbolTable *parent);
void symbol_table_destroy(SymbolTable *table);
Symbol *symbol_table_lookup(SymbolTable *table, char *name);
Symbol *symbol_table_lookup_current_scope(SymbolTable *table, char *name);
bool symbol_table_insert(SymbolTable *table, Symbol *symbol);
Symbol *symbol_create(SymbolKind kind, char *name, Type *type);
void symbol_destroy(Symbol *symbol);

// 语义分析
bool semantic_analyze(ASTNode *root, SymbolTable *global_scope);
bool semantic_check_node(ASTNode *node, SymbolTable *scope);
Type *semantic_check_expression(ASTNode *expr, SymbolTable *scope);
bool semantic_check_assignment(ASTNode *assignment, SymbolTable *scope);
bool semantic_check_function_call(ASTNode *call, SymbolTable *scope);

// 寄存器分配
RegisterAllocator *reg_alloc_create(void);
void reg_alloc_destroy(RegisterAllocator *alloc);
X86Register reg_alloc_get_free_reg(RegisterAllocator *alloc);
X86Register reg_alloc_get_float_reg(RegisterAllocator *alloc);
void reg_alloc_free_reg(RegisterAllocator *alloc, X86Register reg);
void reg_alloc_save_caller_saved(RegisterAllocator *alloc, CodeGenerator *gen);
void reg_alloc_restore_caller_saved(RegisterAllocator *alloc, CodeGenerator *gen);

// 代码生成
CodeGenerator *codegen_create(FILE *output);
void codegen_destroy(CodeGenerator *gen);
bool codegen_generate(CodeGenerator *gen, ASTNode *root);
void codegen_emit_instruction(CodeGenerator *gen, X86Instruction inst, Operand *dst, Operand *src);
void codegen_emit_label(CodeGenerator *gen, char *label);
void codegen_emit_comment(CodeGenerator *gen, char *comment);

// x86 指令生成
void x86_emit_mov(CodeGenerator *gen, Operand *dst, Operand *src);
void x86_emit_add(CodeGenerator *gen, Operand *dst, Operand *src);
void x86_emit_sub(CodeGenerator *gen, Operand *dst, Operand *src);
void x86_emit_mul(CodeGenerator *gen, Operand *dst, Operand *src);
void x86_emit_div(CodeGenerator *gen, Operand *dividend, Operand *divisor);
void x86_emit_cmp(CodeGenerator *gen, Operand *left, Operand *right);
void x86_emit_jump(CodeGenerator *gen, X86Instruction jmp_type, char *label);
void x86_emit_call(CodeGenerator *gen, char *function_name);
void x86_emit_ret(CodeGenerator *gen);

// 并行代码生成
void x86_emit_parallel_for(CodeGenerator *gen, ASTNode *parallel_for);
void x86_emit_atomic_operation(CodeGenerator *gen, ASTNode *atomic_op);
void x86_emit_barrier(CodeGenerator *gen);
void x86_emit_critical_section(CodeGenerator *gen, ASTNode *critical);

// 优化
void x86_optimize_basic_block(CodeGenerator *gen);
void x86_optimize_register_allocation(CodeGenerator *gen);
void x86_optimize_vectorization(CodeGenerator *gen);

// 操作数创建
Operand operand_register(X86Register reg, int size);
Operand operand_immediate(long value, int size);
Operand operand_memory(X86Register base, int displacement, int size);
Operand operand_memory_indexed(X86Register base, X86Register index, int scale, int displacement, int size);
Operand operand_label(char *label);

// 工具函数
void x86_print_register(X86Register reg);
char *x86_register_name(X86Register reg, int size);
int x86_register_size(X86Register reg);
bool x86_is_caller_saved(X86Register reg);
bool x86_is_callee_saved(X86Register reg);

// 错误处理和调试
void error_report(const char *format, ...);
void warning_report(const char *format, ...);
void debug_print(const char *format, ...);

// 内存管理
void *safe_malloc(size_t size);
void *safe_calloc(size_t count, size_t size);
char *safe_strdup(const char *str);

#endif // X86_CC_H
