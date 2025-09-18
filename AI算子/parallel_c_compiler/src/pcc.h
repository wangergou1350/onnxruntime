/*
 * ParallelC Compiler - Main Header File
 * 
 * This file contains all the core data structures and function declarations
 * for the ParallelC compiler implementation.
 */

#ifndef PCC_H
#define PCC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

/* ========== TOKEN DEFINITIONS ========== */

typedef enum {
    // Literals
    TOKEN_NUMBER,
    TOKEN_IDENTIFIER,
    TOKEN_STRING,
    
    // Keywords
    TOKEN_INT,
    TOKEN_FLOAT,
    TOKEN_CHAR,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_FOR,
    TOKEN_RETURN,
    TOKEN_VOID,
    
    // Parallel keywords
    TOKEN_PARALLEL_FOR,
    TOKEN_PARALLEL_REDUCE,
    TOKEN_ATOMIC_ADD,
    TOKEN_BARRIER,
    TOKEN_THREAD_ID,
    
    // Operators
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_ASSIGN,
    TOKEN_EQUAL,
    TOKEN_NOT_EQUAL,
    TOKEN_LESS,
    TOKEN_GREATER,
    TOKEN_LESS_EQUAL,
    TOKEN_GREATER_EQUAL,
    
    // Delimiters
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_LBRACKET,
    TOKEN_RBRACKET,
    TOKEN_SEMICOLON,
    TOKEN_COMMA,
    
    // Special
    TOKEN_EOF,
    TOKEN_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char *value;
    int line;
    int column;
} Token;

/* ========== AST NODE DEFINITIONS ========== */

typedef enum {
    AST_PROGRAM,
    AST_FUNCTION_DEF,
    AST_VARIABLE_DECL,
    AST_ASSIGNMENT,
    AST_BINARY_OP,
    AST_UNARY_OP,
    AST_FUNCTION_CALL,
    AST_IF_STMT,
    AST_WHILE_STMT,
    AST_FOR_STMT,
    AST_PARALLEL_FOR,
    AST_PARALLEL_REDUCE,
    AST_RETURN_STMT,
    AST_BLOCK,
    AST_IDENTIFIER,
    AST_NUMBER,
    AST_STRING,
    AST_ARRAY_ACCESS
} ASTNodeType;

typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_CHAR,
    TYPE_POINTER,
    TYPE_ARRAY,
    TYPE_VOID
} DataType;

typedef struct ASTNode {
    ASTNodeType type;
    DataType data_type;
    
    union {
        // Program
        struct {
            struct ASTNode **functions;
            int function_count;
        } program;
        
        // Function definition
        struct {
            char *name;
            struct ASTNode **params;
            int param_count;
            struct ASTNode *body;
            DataType return_type;
        } function_def;
        
        // Variable declaration
        struct {
            char *name;
            struct ASTNode *init_value;
            DataType var_type;
            int array_size;  // -1 if not array
        } var_decl;
        
        // Assignment
        struct {
            struct ASTNode *target;
            struct ASTNode *value;
        } assignment;
        
        // Binary operation
        struct {
            TokenType operator;
            struct ASTNode *left;
            struct ASTNode *right;
        } binary_op;
        
        // Unary operation
        struct {
            TokenType operator;
            struct ASTNode *operand;
        } unary_op;
        
        // Function call
        struct {
            char *name;
            struct ASTNode **args;
            int arg_count;
        } function_call;
        
        // Control flow
        struct {
            struct ASTNode *condition;
            struct ASTNode *then_stmt;
            struct ASTNode *else_stmt;
        } if_stmt;
        
        struct {
            struct ASTNode *condition;
            struct ASTNode *body;
        } while_stmt;
        
        struct {
            struct ASTNode *init;
            struct ASTNode *condition;
            struct ASTNode *update;
            struct ASTNode *body;
        } for_stmt;
        
        // Parallel constructs
        struct {
            struct ASTNode *start;
            struct ASTNode *end;
            struct ASTNode *body;
        } parallel_for;
        
        struct {
            struct ASTNode *array;
            struct ASTNode *size;
            TokenType operation;  // +, *, etc.
        } parallel_reduce;
        
        // Return statement
        struct {
            struct ASTNode *value;
        } return_stmt;
        
        // Block
        struct {
            struct ASTNode **statements;
            int statement_count;
        } block;
        
        // Terminals
        struct {
            char *name;
        } identifier;
        
        struct {
            union {
                int int_value;
                float float_value;
            };
        } number;
        
        struct {
            char *value;
        } string;
        
        // Array access
        struct {
            struct ASTNode *array;
            struct ASTNode *index;
        } array_access;
    };
} ASTNode;

/* ========== SYMBOL TABLE ========== */

typedef struct Symbol {
    char *name;
    DataType type;
    int scope_level;
    bool is_array;
    int array_size;
    int offset;  // Stack offset for code generation
    struct Symbol *next;
} Symbol;

typedef struct SymbolTable {
    Symbol *symbols;
    int scope_level;
    struct SymbolTable *parent;
} SymbolTable;

/* ========== CODE GENERATION ========== */

typedef struct {
    char *code;
    size_t size;
    size_t capacity;
} CodeBuffer;

typedef struct {
    int thread_count;
    bool optimization_enabled;
    bool debug_info;
} CompilerOptions;

/* ========== FUNCTION DECLARATIONS ========== */

// Lexer
Token *tokenize(const char *source, int *token_count);
void free_tokens(Token *tokens, int count);
void print_token(Token token);

// Parser
ASTNode *parse(Token *tokens, int token_count);
void free_ast(ASTNode *node);
void print_ast(ASTNode *node, int indent);

// Semantic Analysis
bool semantic_analysis(ASTNode *ast, SymbolTable *symbol_table);
SymbolTable *create_symbol_table(SymbolTable *parent);
void free_symbol_table(SymbolTable *table);
Symbol *lookup_symbol(SymbolTable *table, const char *name);
bool add_symbol(SymbolTable *table, const char *name, DataType type, bool is_array, int array_size);

// Code Generation
char *generate_code(ASTNode *ast, CompilerOptions *options);
void generate_parallel_code(ASTNode *node, CodeBuffer *buffer, CompilerOptions *options);

// Optimization
ASTNode *optimize_ast(ASTNode *ast);
ASTNode *constant_folding(ASTNode *node);
ASTNode *dead_code_elimination(ASTNode *node);

// Utility functions
CodeBuffer *create_code_buffer(void);
void append_code(CodeBuffer *buffer, const char *format, ...);
void free_code_buffer(CodeBuffer *buffer);
char *read_file(const char *filename);
void write_file(const char *filename, const char *content);

// Error handling
void error(const char *format, ...);
void warning(const char *format, ...);

// Parallel runtime support
typedef struct {
    int thread_id;
    int thread_count;
    int start;
    int end;
    void *data;
    void (*function)(void*);
} ThreadData;

void parallel_for_runtime(int start, int end, void (*func)(int, void*), void *data, int thread_count);
void parallel_reduce_runtime(void *array, int size, int element_size, void *result, 
                           void (*reduce_func)(void*, void*, void*), int thread_count);

#endif /* PCC_H */
