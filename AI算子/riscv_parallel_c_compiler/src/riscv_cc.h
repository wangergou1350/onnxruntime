#ifndef RISCV_CC_H
#define RISCV_CC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <ctype.h>

// 编译器版本信息
#define RISCV_CC_VERSION_MAJOR 1
#define RISCV_CC_VERSION_MINOR 0
#define RISCV_CC_VERSION_PATCH 0

// 最大长度限制
#define MAX_IDENTIFIER_LENGTH 256
#define MAX_STRING_LENGTH 1024
#define MAX_TOKENS 10000
#define MAX_SYMBOLS 1000
#define MAX_INSTRUCTIONS 5000

// =============================================================================
// 词法分析器 (Lexer)
// =============================================================================

typedef enum {
    // 字面量
    TOKEN_NUMBER = 256,
    TOKEN_FLOAT,
    TOKEN_CHARACTER,
    TOKEN_STRING,
    TOKEN_IDENTIFIER,
    
    // C语言关键字
    TOKEN_AUTO,
    TOKEN_BREAK,
    TOKEN_CASE,
    TOKEN_CHAR,
    TOKEN_CONST,
    TOKEN_CONTINUE,
    TOKEN_DEFAULT,
    TOKEN_DO,
    TOKEN_DOUBLE,
    TOKEN_ELSE,
    TOKEN_ENUM,
    TOKEN_EXTERN,
    TOKEN_FLOAT_KW,
    TOKEN_FOR,
    TOKEN_GOTO,
    TOKEN_IF,
    TOKEN_INT,
    TOKEN_LONG,
    TOKEN_REGISTER,
    TOKEN_RETURN,
    TOKEN_SHORT,
    TOKEN_SIGNED,
    TOKEN_SIZEOF,
    TOKEN_STATIC,
    TOKEN_STRUCT,
    TOKEN_SWITCH,
    TOKEN_TYPEDEF,
    TOKEN_UNION,
    TOKEN_UNSIGNED,
    TOKEN_VOID,
    TOKEN_VOLATILE,
    TOKEN_WHILE,
    
    // 并行计算扩展关键字
    TOKEN_PARALLEL_FOR,
    TOKEN_ATOMIC,
    TOKEN_THREAD_LOCAL,
    TOKEN_BARRIER,
    TOKEN_CRITICAL,
    
    // 操作符
    TOKEN_ASSIGN,           // =
    TOKEN_PLUS_ASSIGN,      // +=
    TOKEN_MINUS_ASSIGN,     // -=
    TOKEN_MUL_ASSIGN,       // *=
    TOKEN_DIV_ASSIGN,       // /=
    TOKEN_MOD_ASSIGN,       // %=
    TOKEN_AND_ASSIGN,       // &=
    TOKEN_OR_ASSIGN,        // |=
    TOKEN_XOR_ASSIGN,       // ^=
    TOKEN_LSHIFT_ASSIGN,    // <<=
    TOKEN_RSHIFT_ASSIGN,    // >>=
    
    TOKEN_LOGICAL_AND,      // &&
    TOKEN_LOGICAL_OR,       // ||
    TOKEN_EQUAL,            // ==
    TOKEN_NOT_EQUAL,        // !=
    TOKEN_LESS_EQUAL,       // <=
    TOKEN_GREATER_EQUAL,    // >=
    TOKEN_LSHIFT,           // <<
    TOKEN_RSHIFT,           // >>
    TOKEN_INCREMENT,        // ++
    TOKEN_DECREMENT,        // --
    TOKEN_ARROW,            // ->
    
    // 分隔符
    TOKEN_SEMICOLON,        // ;
    TOKEN_COMMA,            // ,
    TOKEN_LPAREN,           // (
    TOKEN_RPAREN,           // )
    TOKEN_LBRACE,           // {
    TOKEN_RBRACE,           // }
    TOKEN_LBRACKET,         // [
    TOKEN_RBRACKET,         // ]
    TOKEN_DOT,              // .
    TOKEN_QUESTION,         // ?
    TOKEN_COLON,            // :
    
    // 算术和逻辑操作符 (单字符)
    TOKEN_PLUS,             // +
    TOKEN_MINUS,            // -
    TOKEN_MULTIPLY,         // *
    TOKEN_DIVIDE,           // /
    TOKEN_MODULO,           // %
    TOKEN_BITWISE_AND,      // &
    TOKEN_BITWISE_OR,       // |
    TOKEN_BITWISE_XOR,      // ^
    TOKEN_BITWISE_NOT,      // ~
    TOKEN_LOGICAL_NOT,      // !
    TOKEN_LESS,             // <
    TOKEN_GREATER,          // >
    
    // 特殊标记
    TOKEN_EOF,
    TOKEN_NEWLINE,
    TOKEN_WHITESPACE,
    TOKEN_COMMENT,
    TOKEN_PREPROCESSOR,
    TOKEN_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char *value;
    int line;
    int column;
    union {
        int int_value;
        float float_value;
        char char_value;
    };
} Token;

typedef struct {
    char *source;
    int position;
    int line;
    int column;
    int length;
} Lexer;

// =============================================================================
// 抽象语法树 (AST)
// =============================================================================

typedef enum {
    AST_PROGRAM,
    AST_FUNCTION_DEF,
    AST_VARIABLE_DECL,
    AST_STRUCT_DECL,
    AST_UNION_DECL,
    AST_ENUM_DECL,
    AST_TYPEDEF_DECL,
    
    // 语句
    AST_COMPOUND_STMT,
    AST_EXPRESSION_STMT,
    AST_IF_STMT,
    AST_WHILE_STMT,
    AST_FOR_STMT,
    AST_DO_WHILE_STMT,
    AST_SWITCH_STMT,
    AST_CASE_STMT,
    AST_DEFAULT_STMT,
    AST_BREAK_STMT,
    AST_CONTINUE_STMT,
    AST_RETURN_STMT,
    AST_GOTO_STMT,
    AST_LABEL_STMT,
    
    // 并行语句
    AST_PARALLEL_FOR_STMT,
    AST_CRITICAL_STMT,
    AST_BARRIER_STMT,
    
    // 表达式
    AST_BINARY_EXPR,
    AST_UNARY_EXPR,
    AST_TERNARY_EXPR,
    AST_ASSIGN_EXPR,
    AST_CALL_EXPR,
    AST_MEMBER_EXPR,
    AST_INDEX_EXPR,
    AST_CAST_EXPR,
    AST_SIZEOF_EXPR,
    
    // 原子表达式
    AST_IDENTIFIER,
    AST_NUMBER_LITERAL,
    AST_FLOAT_LITERAL,
    AST_CHAR_LITERAL,
    AST_STRING_LITERAL,
    AST_ARRAY_LITERAL,
    
    // 类型
    AST_TYPE_SPECIFIER,
    AST_POINTER_TYPE,
    AST_ARRAY_TYPE,
    AST_FUNCTION_TYPE,
    AST_STRUCT_TYPE,
    AST_UNION_TYPE,
    AST_ENUM_TYPE
} ASTNodeType;

typedef enum {
    TYPE_VOID,
    TYPE_CHAR,
    TYPE_SHORT,
    TYPE_INT,
    TYPE_LONG,
    TYPE_FLOAT,
    TYPE_DOUBLE,
    TYPE_POINTER,
    TYPE_ARRAY,
    TYPE_FUNCTION,
    TYPE_STRUCT,
    TYPE_UNION,
    TYPE_ENUM,
    TYPE_ATOMIC,    // 原子类型
    TYPE_UNKNOWN
} DataType;

typedef enum {
    STORAGE_AUTO,
    STORAGE_REGISTER,
    STORAGE_STATIC,
    STORAGE_EXTERN,
    STORAGE_TYPEDEF,
    STORAGE_THREAD_LOCAL
} StorageClass;

typedef struct ASTNode {
    ASTNodeType type;
    int line;
    int column;
    
    union {
        // 程序根节点
        struct {
            struct ASTNode **declarations;
            int declaration_count;
            int declaration_capacity;
        } program;
        
        // 函数定义
        struct {
            char *name;
            DataType return_type;
            struct ASTNode **parameters;
            int parameter_count;
            struct ASTNode *body;
            StorageClass storage_class;
            bool is_inline;
            bool is_parallel;    // 并行函数标记
        } function_def;
        
        // 变量声明
        struct {
            char *name;
            DataType data_type;
            struct ASTNode *init_value;
            StorageClass storage_class;
            bool is_const;
            bool is_volatile;
            bool is_atomic;      // 原子变量
            int array_size;      // -1表示不是数组
            int pointer_level;   // 指针级别
        } var_decl;
        
        // 结构体声明
        struct {
            char *name;
            struct ASTNode **members;
            int member_count;
            bool is_packed;
        } struct_decl;
        
        // 联合体声明
        struct {
            char *name;
            struct ASTNode **members;
            int member_count;
        } union_decl;
        
        // 枚举声明
        struct {
            char *name;
            struct ASTNode **members;
            int member_count;
        } enum_decl;
        
        // 复合语句
        struct {
            struct ASTNode **statements;
            int statement_count;
            int statement_capacity;
        } compound_stmt;
        
        // if语句
        struct {
            struct ASTNode *condition;
            struct ASTNode *then_stmt;
            struct ASTNode *else_stmt;
        } if_stmt;
        
        // while语句
        struct {
            struct ASTNode *condition;
            struct ASTNode *body;
        } while_stmt;
        
        // for语句
        struct {
            struct ASTNode *init;
            struct ASTNode *condition;
            struct ASTNode *increment;
            struct ASTNode *body;
        } for_stmt;
        
        // parallel_for语句
        struct {
            struct ASTNode *init;
            struct ASTNode *condition;
            struct ASTNode *increment;
            struct ASTNode *body;
            int num_threads;     // 线程数，0表示自动
        } parallel_for_stmt;
        
        // switch语句
        struct {
            struct ASTNode *expression;
            struct ASTNode **cases;
            int case_count;
            struct ASTNode *default_case;
        } switch_stmt;
        
        // case语句
        struct {
            struct ASTNode *value;
            struct ASTNode **statements;
            int statement_count;
        } case_stmt;
        
        // return语句
        struct {
            struct ASTNode *value;
        } return_stmt;
        
        // goto语句
        struct {
            char *label;
        } goto_stmt;
        
        // 标签语句
        struct {
            char *label;
            struct ASTNode *statement;
        } label_stmt;
        
        // 二元表达式
        struct {
            TokenType operator;
            struct ASTNode *left;
            struct ASTNode *right;
            DataType result_type;
        } binary_expr;
        
        // 一元表达式
        struct {
            TokenType operator;
            struct ASTNode *operand;
            DataType result_type;
        } unary_expr;
        
        // 三元表达式
        struct {
            struct ASTNode *condition;
            struct ASTNode *true_expr;
            struct ASTNode *false_expr;
            DataType result_type;
        } ternary_expr;
        
        // 赋值表达式
        struct {
            TokenType operator;
            struct ASTNode *left;
            struct ASTNode *right;
        } assign_expr;
        
        // 函数调用
        struct {
            struct ASTNode *function;
            struct ASTNode **arguments;
            int argument_count;
            DataType return_type;
        } call_expr;
        
        // 成员访问
        struct {
            struct ASTNode *object;
            char *member;
            bool is_pointer_access;  // -> vs .
            DataType member_type;
        } member_expr;
        
        // 数组索引
        struct {
            struct ASTNode *array;
            struct ASTNode *index;
            DataType element_type;
        } index_expr;
        
        // 类型转换
        struct {
            DataType target_type;
            struct ASTNode *expression;
        } cast_expr;
        
        // sizeof表达式
        struct {
            DataType target_type;
            struct ASTNode *expression;  // 如果是表达式而非类型
        } sizeof_expr;
        
        // 标识符
        struct {
            char *name;
            DataType data_type;
        } identifier;
        
        // 字面量
        struct {
            union {
                int int_value;
                float float_value;
                char char_value;
                char *string_value;
            };
            DataType literal_type;
        } literal;
        
        // 类型说明符
        struct {
            DataType base_type;
            bool is_const;
            bool is_volatile;
            bool is_signed;
            bool is_unsigned;
            bool is_atomic;
            int pointer_level;
            int array_sizes[8];  // 支持多维数组
            int array_dimensions;
        } type_spec;
    };
} ASTNode;

// =============================================================================
// 符号表和语义分析
// =============================================================================

typedef enum {
    SYMBOL_VARIABLE,
    SYMBOL_FUNCTION,
    SYMBOL_TYPE,
    SYMBOL_LABEL,
    SYMBOL_ENUM_CONSTANT
} SymbolType;

typedef struct Symbol {
    char *name;
    SymbolType symbol_type;
    DataType data_type;
    StorageClass storage_class;
    int offset;          // 栈偏移或全局地址
    int size;            // 大小（字节）
    bool is_defined;     // 是否已定义
    bool is_used;        // 是否被使用
    bool is_const;
    bool is_volatile;
    bool is_atomic;
    int scope_level;     // 作用域级别
    
    // 函数特定信息
    struct {
        DataType return_type;
        DataType *param_types;
        int param_count;
        bool is_variadic;
        bool is_inline;
        bool is_parallel;
    } function_info;
    
    // 结构体/联合体特定信息
    struct {
        struct Symbol **members;
        int member_count;
        int total_size;
        bool is_packed;
    } composite_info;
    
    struct Symbol *next;  // 链表节点
} Symbol;

typedef struct SymbolTable {
    Symbol **buckets;
    int bucket_count;
    int scope_level;
    struct SymbolTable *parent;
    struct SymbolTable *next;
} SymbolTable;

typedef struct {
    SymbolTable *global_scope;
    SymbolTable *current_scope;
    int current_scope_level;
    bool has_errors;
    char *error_message;
} SemanticAnalyzer;

// =============================================================================
// RISC-V代码生成
// =============================================================================

typedef enum {
    // RISC-V寄存器
    REG_ZERO = 0,   // x0 - 硬编码为0
    REG_RA = 1,     // x1 - 返回地址
    REG_SP = 2,     // x2 - 栈指针
    REG_GP = 3,     // x3 - 全局指针
    REG_TP = 4,     // x4 - 线程指针
    REG_T0 = 5,     // x5 - 临时寄存器
    REG_T1 = 6,     // x6
    REG_T2 = 7,     // x7
    REG_S0 = 8,     // x8/fp - 保存寄存器/帧指针
    REG_S1 = 9,     // x9
    REG_A0 = 10,    // x10 - 参数/返回值寄存器
    REG_A1 = 11,    // x11
    REG_A2 = 12,    // x12
    REG_A3 = 13,    // x13
    REG_A4 = 14,    // x14
    REG_A5 = 15,    // x15
    REG_A6 = 16,    // x16
    REG_A7 = 17,    // x17
    REG_S2 = 18,    // x18 - 保存寄存器
    REG_S3 = 19,    // x19
    REG_S4 = 20,    // x20
    REG_S5 = 21,    // x21
    REG_S6 = 22,    // x22
    REG_S7 = 23,    // x23
    REG_S8 = 24,    // x24
    REG_S9 = 25,    // x25
    REG_S10 = 26,   // x26
    REG_S11 = 27,   // x27
    REG_T3 = 28,    // x28 - 临时寄存器
    REG_T4 = 29,    // x29
    REG_T5 = 30,    // x30
    REG_T6 = 31,    // x31
    
    // 浮点寄存器
    REG_FT0 = 32,   // f0 - 临时浮点寄存器
    REG_FT1 = 33,   // f1
    REG_FT2 = 34,   // f2
    REG_FT3 = 35,   // f3
    REG_FT4 = 36,   // f4
    REG_FT5 = 37,   // f5
    REG_FT6 = 38,   // f6
    REG_FT7 = 39,   // f7
    REG_FS0 = 40,   // f8 - 保存浮点寄存器
    REG_FS1 = 41,   // f9
    REG_FA0 = 42,   // f10 - 参数/返回值浮点寄存器
    REG_FA1 = 43,   // f11
    REG_FA2 = 44,   // f12
    REG_FA3 = 45,   // f13
    REG_FA4 = 46,   // f14
    REG_FA5 = 47,   // f15
    REG_FA6 = 48,   // f16
    REG_FA7 = 49,   // f17
    REG_FS2 = 50,   // f18 - 保存浮点寄存器
    REG_FS3 = 51,   // f19
    REG_FS4 = 52,   // f20
    REG_FS5 = 53,   // f21
    REG_FS6 = 54,   // f22
    REG_FS7 = 55,   // f23
    REG_FS8 = 56,   // f24
    REG_FS9 = 57,   // f25
    REG_FS10 = 58,  // f26
    REG_FS11 = 59,  // f27
    REG_FT8 = 60,   // f28 - 临时浮点寄存器
    REG_FT9 = 61,   // f29
    REG_FT10 = 62,  // f30
    REG_FT11 = 63,  // f31
    
    REG_COUNT = 64,
    REG_NONE = -1
} RiscVRegister;

typedef enum {
    // RV32I基础指令集
    INST_LUI,      // Load Upper Immediate
    INST_AUIPC,    // Add Upper Immediate to PC
    INST_JAL,      // Jump and Link
    INST_JALR,     // Jump and Link Register
    INST_BEQ,      // Branch Equal
    INST_BNE,      // Branch Not Equal
    INST_BLT,      // Branch Less Than
    INST_BGE,      // Branch Greater Equal
    INST_BLTU,     // Branch Less Than Unsigned
    INST_BGEU,     // Branch Greater Equal Unsigned
    INST_LB,       // Load Byte
    INST_LH,       // Load Halfword
    INST_LW,       // Load Word
    INST_LBU,      // Load Byte Unsigned
    INST_LHU,      // Load Halfword Unsigned
    INST_SB,       // Store Byte
    INST_SH,       // Store Halfword
    INST_SW,       // Store Word
    INST_ADDI,     // Add Immediate
    INST_SLTI,     // Set Less Than Immediate
    INST_SLTIU,    // Set Less Than Immediate Unsigned
    INST_XORI,     // XOR Immediate
    INST_ORI,      // OR Immediate
    INST_ANDI,     // AND Immediate
    INST_SLLI,     // Shift Left Logical Immediate
    INST_SRLI,     // Shift Right Logical Immediate
    INST_SRAI,     // Shift Right Arithmetic Immediate
    INST_ADD,      // Add
    INST_SUB,      // Subtract
    INST_SLL,      // Shift Left Logical
    INST_SLT,      // Set Less Than
    INST_SLTU,     // Set Less Than Unsigned
    INST_XOR,      // XOR
    INST_SRL,      // Shift Right Logical
    INST_SRA,      // Shift Right Arithmetic
    INST_OR,       // OR
    INST_AND,      // AND
    
    // RV64I (64位扩展)
    INST_LWU,      // Load Word Unsigned
    INST_LD,       // Load Doubleword
    INST_SD,       // Store Doubleword
    INST_ADDIW,    // Add Immediate Word
    INST_SLLIW,    // Shift Left Logical Immediate Word
    INST_SRLIW,    // Shift Right Logical Immediate Word
    INST_SRAIW,    // Shift Right Arithmetic Immediate Word
    INST_ADDW,     // Add Word
    INST_SUBW,     // Subtract Word
    INST_SLLW,     // Shift Left Logical Word
    INST_SRLW,     // Shift Right Logical Word
    INST_SRAW,     // Shift Right Arithmetic Word
    
    // RV32M乘除法扩展
    INST_MUL,      // Multiply
    INST_MULH,     // Multiply High
    INST_MULHSU,   // Multiply High Signed Unsigned
    INST_MULHU,    // Multiply High Unsigned
    INST_DIV,      // Divide
    INST_DIVU,     // Divide Unsigned
    INST_REM,      // Remainder
    INST_REMU,     // Remainder Unsigned
    
    // RV64M
    INST_MULW,     // Multiply Word
    INST_DIVW,     // Divide Word
    INST_DIVUW,    // Divide Unsigned Word
    INST_REMW,     // Remainder Word
    INST_REMUW,    // Remainder Unsigned Word
    
    // RV32A原子指令扩展
    INST_LR_W,     // Load Reserved Word
    INST_SC_W,     // Store Conditional Word
    INST_AMOSWAP_W,   // Atomic Swap Word
    INST_AMOADD_W,    // Atomic Add Word
    INST_AMOXOR_W,    // Atomic XOR Word
    INST_AMOAND_W,    // Atomic AND Word
    INST_AMOOR_W,     // Atomic OR Word
    INST_AMOMIN_W,    // Atomic Min Word
    INST_AMOMAX_W,    // Atomic Max Word
    INST_AMOMINU_W,   // Atomic Min Unsigned Word
    INST_AMOMAXU_W,   // Atomic Max Unsigned Word
    
    // RV64A
    INST_LR_D,     // Load Reserved Doubleword
    INST_SC_D,     // Store Conditional Doubleword
    INST_AMOSWAP_D,   // Atomic Swap Doubleword
    INST_AMOADD_D,    // Atomic Add Doubleword
    INST_AMOXOR_D,    // Atomic XOR Doubleword
    INST_AMOAND_D,    // Atomic AND Doubleword
    INST_AMOOR_D,     // Atomic OR Doubleword
    INST_AMOMIN_D,    // Atomic Min Doubleword
    INST_AMOMAX_D,    // Atomic Max Doubleword
    INST_AMOMINU_D,   // Atomic Min Unsigned Doubleword
    INST_AMOMAXU_D,   // Atomic Max Unsigned Doubleword
    
    // RV32F单精度浮点
    INST_FLW,      // Floating Load Word
    INST_FSW,      // Floating Store Word
    INST_FMADD_S,  // Floating Multiply Add Single
    INST_FMSUB_S,  // Floating Multiply Subtract Single
    INST_FNMSUB_S, // Floating Negative Multiply Subtract Single
    INST_FNMADD_S, // Floating Negative Multiply Add Single
    INST_FADD_S,   // Floating Add Single
    INST_FSUB_S,   // Floating Subtract Single
    INST_FMUL_S,   // Floating Multiply Single
    INST_FDIV_S,   // Floating Divide Single
    INST_FSQRT_S,  // Floating Square Root Single
    
    // RV32D双精度浮点
    INST_FLD,      // Floating Load Double
    INST_FSD,      // Floating Store Double
    INST_FMADD_D,  // Floating Multiply Add Double
    INST_FMSUB_D,  // Floating Multiply Subtract Double
    INST_FNMSUB_D, // Floating Negative Multiply Subtract Double
    INST_FNMADD_D, // Floating Negative Multiply Add Double
    INST_FADD_D,   // Floating Add Double
    INST_FSUB_D,   // Floating Subtract Double
    INST_FMUL_D,   // Floating Multiply Double
    INST_FDIV_D,   // Floating Divide Double
    INST_FSQRT_D,  // Floating Square Root Double
    
    // 伪指令
    INST_NOP,      // No Operation
    INST_MV,       // Move (addi rd, rs1, 0)
    INST_NOT,      // Not (xori rd, rs1, -1)
    INST_NEG,      // Negate (sub rd, x0, rs1)
    INST_RET,      // Return (jalr x0, x1, 0)
    INST_J,        // Jump (jal x0, offset)
    INST_CALL,     // Call (auipc/jalr sequence)
    INST_TAIL,     // Tail call
    INST_LI,       // Load Immediate
    INST_LA,       // Load Address
    
    // 并行和同步指令
    INST_FENCE,    // Fence
    INST_FENCE_I,  // Fence Instruction
    
    INST_COUNT
} RiscVInstruction;

typedef enum {
    OPERAND_REGISTER,
    OPERAND_IMMEDIATE,
    OPERAND_LABEL,
    OPERAND_MEMORY,
    OPERAND_OFFSET
} OperandType;

typedef struct {
    OperandType type;
    union {
        RiscVRegister reg;
        int immediate;
        char *label;
        struct {
            RiscVRegister base;
            int offset;
        } memory;
    };
} Operand;

typedef struct Instruction {
    RiscVInstruction opcode;
    Operand operands[3];  // 最多3个操作数
    int operand_count;
    char *comment;        // 注释
    struct Instruction *next;
} Instruction;

typedef struct {
    Instruction *instructions;
    Instruction *last_instruction;
    int instruction_count;
    
    // 寄存器分配
    bool register_used[REG_COUNT];
    RiscVRegister next_temp_reg;
    RiscVRegister next_temp_freg;
    
    // 栈管理
    int stack_offset;
    int max_stack_size;
    
    // 标签管理
    int next_label_id;
    
    // 并行代码生成
    bool parallel_mode;
    int num_cores;
    
    // 优化开关
    bool optimize_registers;
    bool optimize_instructions;
    bool optimize_parallel;
} CodeGenerator;

// =============================================================================
// 编译器上下文
// =============================================================================

typedef struct {
    char *input_filename;
    char *output_filename;
    bool verbose;
    bool debug;
    bool parallel_enabled;
    int optimization_level;  // 0-3
    bool target_riscv64;     // true for RV64, false for RV32
    bool enable_extensions[32]; // 扩展启用标志
    
    // 编译阶段控制
    bool only_lex;
    bool only_parse;
    bool only_semantic;
    bool print_ast;
    bool print_tokens;
    bool print_symbols;
    
    // 错误和警告
    int error_count;
    int warning_count;
    bool warnings_as_errors;
} CompilerOptions;

typedef struct {
    CompilerOptions options;
    Lexer lexer;
    Token *tokens;
    int token_count;
    int token_capacity;
    ASTNode *ast;
    SemanticAnalyzer semantic;
    CodeGenerator codegen;
    FILE *output_file;
} Compiler;

// =============================================================================
// 函数声明
// =============================================================================

// 词法分析
Lexer *create_lexer(const char *source);
void destroy_lexer(Lexer *lexer);
Token next_token(Lexer *lexer);
Token *tokenize(Lexer *lexer, int *token_count);
const char *token_type_to_string(TokenType type);
void print_token(Token token);
void print_tokens(Token *tokens, int count);

// 语法分析
ASTNode *parse(Token *tokens, int token_count);
ASTNode *create_ast_node(ASTNodeType type);
void destroy_ast(ASTNode *node);
void print_ast(ASTNode *node, int indent);

// 语义分析
SemanticAnalyzer *create_semantic_analyzer(void);
void destroy_semantic_analyzer(SemanticAnalyzer *analyzer);
bool analyze_semantics(SemanticAnalyzer *analyzer, ASTNode *ast);
Symbol *create_symbol(const char *name, SymbolType type, DataType data_type);
SymbolTable *create_symbol_table(int scope_level);
void enter_scope(SemanticAnalyzer *analyzer);
void exit_scope(SemanticAnalyzer *analyzer);
Symbol *lookup_symbol(SymbolTable *table, const char *name);
void insert_symbol(SymbolTable *table, Symbol *symbol);

// 代码生成
CodeGenerator *create_code_generator(bool riscv64);
void destroy_code_generator(CodeGenerator *codegen);
bool generate_code(CodeGenerator *codegen, ASTNode *ast, FILE *output);
void emit_instruction(CodeGenerator *codegen, RiscVInstruction opcode, ...);
RiscVRegister allocate_register(CodeGenerator *codegen, bool is_float);
void free_register(CodeGenerator *codegen, RiscVRegister reg);
char *generate_label(CodeGenerator *codegen);

// 优化
void optimize_ast(ASTNode *ast);
void optimize_instructions(CodeGenerator *codegen);
void optimize_registers(CodeGenerator *codegen);
void optimize_parallel(CodeGenerator *codegen);

// 工具函数
void error(const char *format, ...);
void warning(const char *format, ...);
void debug_print(const char *format, ...);
char *read_file(const char *filename);
bool write_file(const char *filename, const char *content);
int get_type_size(DataType type);
bool is_signed_type(DataType type);
bool is_float_type(DataType type);
const char *data_type_to_string(DataType type);
const char *register_to_string(RiscVRegister reg);

// 主编译函数
Compiler *create_compiler(void);
void destroy_compiler(Compiler *compiler);
bool compile_file(Compiler *compiler, const char *input_file, const char *output_file);
void print_usage(const char *program_name);
int parse_arguments(int argc, char *argv[], CompilerOptions *options);

#endif // RISCV_CC_H
