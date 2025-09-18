/*
 * ParallelC++ Compiler - Main Header File
 * 
 * This file defines all data structures, enums, and function declarations
 * for a minimal C++ compiler with parallel computing extensions.
 */

#ifndef PCPP_H
#define PCPP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <ctype.h>

// Maximum limits
#define MAX_TOKEN_LENGTH 256
#define MAX_IDENTIFIER_LENGTH 64
#define MAX_STRING_LENGTH 512
#define MAX_MEMBERS 100
#define MAX_METHODS 50
#define MAX_INHERITANCE_DEPTH 10

// Token types for C++ language
typedef enum {
    // Basic tokens
    TOKEN_EOF = 0,
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_STRING,
    TOKEN_CHAR,
    
    // C++ Keywords
    TOKEN_CLASS,
    TOKEN_PUBLIC,
    TOKEN_PRIVATE,
    TOKEN_PROTECTED,
    TOKEN_VIRTUAL,
    TOKEN_STATIC,
    TOKEN_CONST,
    TOKEN_INLINE,
    TOKEN_FRIEND,
    TOKEN_NAMESPACE,
    TOKEN_USING,
    TOKEN_TEMPLATE,
    TOKEN_TYPENAME,
    TOKEN_THIS,
    TOKEN_NEW,
    TOKEN_DELETE,
    TOKEN_OPERATOR,
    TOKEN_OVERRIDE,
    TOKEN_FINAL,
    
    // C Keywords (inherited)
    TOKEN_INT,
    TOKEN_FLOAT,
    TOKEN_DOUBLE,
    TOKEN_CHAR_TYPE,
    TOKEN_VOID,
    TOKEN_BOOL,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_FOR,
    TOKEN_DO,
    TOKEN_SWITCH,
    TOKEN_CASE,
    TOKEN_DEFAULT,
    TOKEN_BREAK,
    TOKEN_CONTINUE,
    TOKEN_RETURN,
    TOKEN_SIZEOF,
    TOKEN_TYPEDEF,
    TOKEN_STRUCT,
    TOKEN_UNION,
    TOKEN_ENUM,
    
    // Parallel Computing Extensions
    TOKEN_PARALLEL_FOR,
    TOKEN_PARALLEL_INVOKE,
    TOKEN_PARALLEL_CLASS,
    TOKEN_THREAD_SAFE,
    TOKEN_ATOMIC_ADD,
    TOKEN_ATOMIC_SUB,
    TOKEN_BARRIER,
    TOKEN_THREAD_ID,
    TOKEN_NUM_THREADS,
    TOKEN_LAMBDA,
    
    // Operators
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_MODULO,
    TOKEN_ASSIGN,
    TOKEN_PLUS_ASSIGN,
    TOKEN_MINUS_ASSIGN,
    TOKEN_MULTIPLY_ASSIGN,
    TOKEN_DIVIDE_ASSIGN,
    TOKEN_INCREMENT,
    TOKEN_DECREMENT,
    
    // Comparison
    TOKEN_EQUAL,
    TOKEN_NOT_EQUAL,
    TOKEN_LESS,
    TOKEN_GREATER,
    TOKEN_LESS_EQUAL,
    TOKEN_GREATER_EQUAL,
    
    // Logical
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_NOT,
    TOKEN_LOGICAL_AND,
    TOKEN_LOGICAL_OR,
    
    // Bitwise
    TOKEN_BITWISE_AND,
    TOKEN_BITWISE_OR,
    TOKEN_BITWISE_XOR,
    TOKEN_BITWISE_NOT,
    TOKEN_LEFT_SHIFT,
    TOKEN_RIGHT_SHIFT,
    
    // Punctuation
    TOKEN_SEMICOLON,
    TOKEN_COMMA,
    TOKEN_DOT,
    TOKEN_ARROW,
    TOKEN_SCOPE_RESOLUTION,
    TOKEN_QUESTION,
    TOKEN_COLON,
    
    // Brackets
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_LBRACKET,
    TOKEN_RBRACKET,
    TOKEN_LANGLE,
    TOKEN_RANGLE,
    
    // Special
    TOKEN_NEWLINE,
    TOKEN_WHITESPACE,
    TOKEN_COMMENT,
    TOKEN_PREPROCESSOR
} TokenType;

// Data types in C++
typedef enum {
    TYPE_VOID = 0,
    TYPE_BOOL,
    TYPE_CHAR,
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_DOUBLE,
    TYPE_POINTER,
    TYPE_REFERENCE,
    TYPE_CLASS,
    TYPE_TEMPLATE,
    TYPE_AUTO
} DataType;

// Access modifiers
typedef enum {
    ACCESS_PUBLIC = 0,
    ACCESS_PRIVATE,
    ACCESS_PROTECTED
} AccessModifier;

// Member types
typedef enum {
    MEMBER_VARIABLE = 0,
    MEMBER_FUNCTION,
    MEMBER_CONSTRUCTOR,
    MEMBER_DESTRUCTOR,
    MEMBER_OPERATOR
} MemberType;

// Token structure
typedef struct Token {
    TokenType type;
    char value[MAX_TOKEN_LENGTH];
    int line;
    int column;
} Token;

// Forward declarations
typedef struct ASTNode ASTNode;
typedef struct ClassMember ClassMember;
typedef struct ClassInfo ClassInfo;

// AST Node types
typedef enum {
    // Program structure
    AST_PROGRAM = 0,
    AST_NAMESPACE,
    AST_USING,
    
    // Class-related
    AST_CLASS_DEF,
    AST_CLASS_MEMBER,
    AST_CONSTRUCTOR,
    AST_DESTRUCTOR,
    AST_METHOD,
    AST_MEMBER_ACCESS,
    AST_THIS_POINTER,
    AST_NEW_EXPR,
    AST_DELETE_EXPR,
    AST_INHERITANCE,
    
    // Functions
    AST_FUNCTION_DEF,
    AST_FUNCTION_CALL,
    AST_METHOD_CALL,
    AST_OPERATOR_OVERLOAD,
    
    // Variables and expressions
    AST_VARIABLE_DECL,
    AST_IDENTIFIER,
    AST_NUMBER,
    AST_STRING,
    AST_CHAR_LITERAL,
    AST_ASSIGNMENT,
    AST_BINARY_OP,
    AST_UNARY_OP,
    AST_ARRAY_ACCESS,
    AST_POINTER_DEREF,
    AST_ADDRESS_OF,
    AST_CAST,
    
    // Control flow
    AST_IF_STMT,
    AST_WHILE_STMT,
    AST_FOR_STMT,
    AST_DO_WHILE_STMT,
    AST_SWITCH_STMT,
    AST_CASE_STMT,
    AST_BREAK_STMT,
    AST_CONTINUE_STMT,
    AST_RETURN_STMT,
    AST_BLOCK,
    
    // Parallel extensions
    AST_PARALLEL_FOR,
    AST_PARALLEL_INVOKE,
    AST_PARALLEL_CLASS,
    AST_THREAD_SAFE_METHOD,
    AST_LAMBDA_EXPR,
    AST_ATOMIC_OP,
    AST_BARRIER_CALL,
    
    // Templates (basic)
    AST_TEMPLATE_DEF,
    AST_TEMPLATE_INSTANTIATION
} ASTNodeType;

// Class member information
struct ClassMember {
    char name[MAX_IDENTIFIER_LENGTH];
    MemberType type;
    AccessModifier access;
    DataType data_type;
    bool is_static;
    bool is_virtual;
    bool is_const;
    bool is_thread_safe;
    ASTNode *declaration;
    struct ClassMember *next;
};

// Class information
struct ClassInfo {
    char name[MAX_IDENTIFIER_LENGTH];
    char base_class[MAX_IDENTIFIER_LENGTH];
    bool has_base_class;
    bool is_parallel_class;
    ClassMember *members;
    int member_count;
    bool has_virtual_methods;
    bool has_constructor;
    bool has_destructor;
};

// AST Node structure
struct ASTNode {
    ASTNodeType type;
    DataType data_type;
    int line;
    
    union {
        // Program
        struct {
            ASTNode **declarations;
            int declaration_count;
        } program;
        
        // Class definition
        struct {
            char *name;
            char *base_class;
            bool has_base_class;
            bool is_parallel_class;
            ASTNode **members;
            int member_count;
            ClassInfo *class_info;
        } class_def;
        
        // Function/Method definition
        struct {
            char *name;
            DataType return_type;
            ASTNode **params;
            int param_count;
            ASTNode *body;
            bool is_virtual;
            bool is_static;
            bool is_const;
            bool is_thread_safe;
            char *class_name; // For methods
        } function_def;
        
        // Constructor/Destructor
        struct {
            char *class_name;
            ASTNode **params;
            int param_count;
            ASTNode *body;
            ASTNode **initializer_list;
            int initializer_count;
        } constructor;
        
        struct {
            char *class_name;
            ASTNode *body;
            bool is_virtual;
        } destructor;
        
        // Variable declaration
        struct {
            DataType var_type;
            char *name;
            ASTNode *init_value;
            bool is_static;
            bool is_const;
            int array_size;
            char *class_type; // For class instances
        } var_decl;
        
        // Expressions
        struct {
            char *name;
            char *class_scope; // For qualified names
        } identifier;
        
        struct {
            union {
                int int_value;
                float float_value;
                double double_value;
            };
        } number;
        
        struct {
            char *value;
        } string;
        
        struct {
            char value;
        } char_literal;
        
        struct {
            ASTNode *target;
            ASTNode *value;
            TokenType operator;
        } assignment;
        
        struct {
            ASTNode *left;
            ASTNode *right;
            TokenType operator;
        } binary_op;
        
        struct {
            ASTNode *operand;
            TokenType operator;
        } unary_op;
        
        // Function/Method calls
        struct {
            char *name;
            ASTNode **args;
            int arg_count;
            char *class_scope; // For qualified calls
        } function_call;
        
        struct {
            ASTNode *object;
            char *method_name;
            ASTNode **args;
            int arg_count;
            bool is_arrow; // -> vs .
        } method_call;
        
        // Member access
        struct {
            ASTNode *object;
            char *member_name;
            bool is_arrow; // -> vs .
        } member_access;
        
        // Memory management
        struct {
            DataType type;
            char *class_name;
            ASTNode **args;
            int arg_count;
            bool is_array;
            ASTNode *array_size;
        } new_expr;
        
        struct {
            ASTNode *operand;
            bool is_array;
        } delete_expr;
        
        // Control flow
        struct {
            ASTNode *condition;
            ASTNode *then_stmt;
            ASTNode *else_stmt;
        } if_stmt;
        
        struct {
            ASTNode *condition;
            ASTNode *body;
        } while_stmt;
        
        struct {
            ASTNode *init;
            ASTNode *condition;
            ASTNode *update;
            ASTNode *body;
        } for_stmt;
        
        struct {
            ASTNode *value;
        } return_stmt;
        
        struct {
            ASTNode **statements;
            int statement_count;
        } block;
        
        // Parallel extensions
        struct {
            ASTNode *start;
            ASTNode *end;
            ASTNode *body;
            char *iterator_name;
        } parallel_for;
        
        struct {
            ASTNode **lambda_exprs;
            int expr_count;
        } parallel_invoke;
        
        struct {
            ASTNode **capture_list;
            int capture_count;
            ASTNode **params;
            int param_count;
            ASTNode *body;
        } lambda_expr;
        
        struct {
            ASTNode *target;
            ASTNode *value;
            TokenType operation; // add, sub, etc.
        } atomic_op;
        
        // Array/Pointer operations
        struct {
            ASTNode *array;
            ASTNode *index;
        } array_access;
        
        // Inheritance
        struct {
            char *base_class;
            AccessModifier access;
        } inheritance;
        
        // Templates (basic)
        struct {
            char *name;
            char **type_params;
            int type_param_count;
            ASTNode *body;
        } template_def;
    };
};

// Symbol table entry
typedef struct SymbolEntry {
    char *name;
    DataType type;
    char *class_name; // For class instances
    bool is_function;
    bool is_method;
    bool is_class;
    bool is_static;
    bool is_const;
    AccessModifier access;
    int scope_level;
    ASTNode *declaration;
    struct SymbolEntry *next;
} SymbolEntry;

// Scope structure
typedef struct Scope {
    SymbolEntry *symbols;
    struct Scope *parent;
    int level;
    char *class_context; // Current class context
} Scope;

// Error handling
void error(const char *format, ...);
void warning(const char *format, ...);

// Lexer functions
Token *tokenize(const char *source, int *token_count);
void free_tokens(Token *tokens, int count);
void print_tokens(Token *tokens, int count);

// Parser functions
ASTNode *parse_cpp(Token *tokens, int token_count);
void free_ast(ASTNode *node);
void print_ast(ASTNode *node, int indent);

// Semantic analysis functions
bool semantic_analysis_cpp(ASTNode *ast);
ClassInfo *get_class_info(const char *class_name);
bool is_type_compatible(DataType from, DataType to);
bool check_access_rights(const char *class_name, const char *member_name, 
                        AccessModifier current_access);

// Code generation functions
bool generate_cpp_code(ASTNode *ast, const char *output_filename);

// Utility functions
char *string_duplicate(const char *str);
void init_compiler();
void cleanup_compiler();

// Class management
ClassInfo *create_class_info(const char *name);
void add_class_member(ClassInfo *class_info, const char *name, 
                     MemberType type, AccessModifier access, DataType data_type);
ClassMember *find_class_member(ClassInfo *class_info, const char *name);
bool is_derived_from(const char *derived_class, const char *base_class);

// Template handling (basic)
bool is_template_instantiation(const char *name);
char *resolve_template_type(const char *template_name, const char *type_arg);

#endif // PCPP_H
