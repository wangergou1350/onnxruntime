/*
 * ParallelC Compiler - Semantic Analyzer
 * 
 * This module performs semantic analysis including:
 * - Type checking
 * - Symbol table management
 * - Scope management
 * - Declaration validation
 */

#include "pcc.h"

// Symbol table implementation
typedef struct SymbolEntry {
    char *name;
    DataType type;
    bool is_function;
    bool is_array;
    int array_size;
    int scope_level;
    struct SymbolEntry *next;
} SymbolEntry;

typedef struct Scope {
    SymbolEntry *symbols;
    struct Scope *parent;
    int level;
} Scope;

typedef struct SemanticAnalyzer {
    Scope *current_scope;
    bool has_errors;
    DataType current_function_return_type;
} SemanticAnalyzer;

static Scope *create_scope(Scope *parent) {
    Scope *scope = calloc(1, sizeof(Scope));
    if (!scope) {
        error("Memory allocation failed");
        return NULL;
    }
    
    scope->parent = parent;
    scope->level = parent ? parent->level + 1 : 0;
    scope->symbols = NULL;
    
    return scope;
}

static void free_scope(Scope *scope) {
    if (!scope) return;
    
    SymbolEntry *current = scope->symbols;
    while (current) {
        SymbolEntry *next = current->next;
        free(current->name);
        free(current);
        current = next;
    }
    
    free(scope);
}

static SymbolEntry *add_symbol(Scope *scope, const char *name, DataType type, 
                              bool is_function, bool is_array, int array_size) {
    if (!scope || !name) return NULL;
    
    // Check if symbol already exists in current scope
    SymbolEntry *current = scope->symbols;
    while (current) {
        if (strcmp(current->name, name) == 0) {
            error("Symbol '%s' already declared in current scope", name);
            return NULL;
        }
        current = current->next;
    }
    
    SymbolEntry *entry = calloc(1, sizeof(SymbolEntry));
    if (!entry) {
        error("Memory allocation failed");
        return NULL;
    }
    
    entry->name = strdup(name);
    entry->type = type;
    entry->is_function = is_function;
    entry->is_array = is_array;
    entry->array_size = array_size;
    entry->scope_level = scope->level;
    entry->next = scope->symbols;
    scope->symbols = entry;
    
    return entry;
}

static SymbolEntry *lookup_symbol(Scope *scope, const char *name) {
    if (!scope || !name) return NULL;
    
    Scope *current_scope = scope;
    while (current_scope) {
        SymbolEntry *entry = current_scope->symbols;
        while (entry) {
            if (strcmp(entry->name, name) == 0) {
                return entry;
            }
            entry = entry->next;
        }
        current_scope = current_scope->parent;
    }
    
    return NULL;
}

static void enter_scope(SemanticAnalyzer *analyzer) {
    Scope *new_scope = create_scope(analyzer->current_scope);
    analyzer->current_scope = new_scope;
}

static void exit_scope(SemanticAnalyzer *analyzer) {
    if (!analyzer->current_scope) return;
    
    Scope *old_scope = analyzer->current_scope;
    analyzer->current_scope = old_scope->parent;
    free_scope(old_scope);
}

static bool is_compatible_type(DataType from, DataType to) {
    if (from == to) return true;
    
    // Allow implicit conversions
    if ((from == TYPE_INT && to == TYPE_FLOAT) ||
        (from == TYPE_CHAR && to == TYPE_INT)) {
        return true;
    }
    
    return false;
}

static DataType get_binary_op_result_type(DataType left, DataType right, TokenType op) {
    // Arithmetic operations
    if (op == TOKEN_PLUS || op == TOKEN_MINUS || 
        op == TOKEN_MULTIPLY || op == TOKEN_DIVIDE) {
        
        if (left == TYPE_FLOAT || right == TYPE_FLOAT) {
            return TYPE_FLOAT;
        }
        return TYPE_INT;
    }
    
    // Comparison operations always return int (boolean)
    if (op == TOKEN_LESS || op == TOKEN_GREATER ||
        op == TOKEN_LESS_EQUAL || op == TOKEN_GREATER_EQUAL ||
        op == TOKEN_EQUAL || op == TOKEN_NOT_EQUAL) {
        return TYPE_INT;
    }
    
    return TYPE_INT; // Default
}

static DataType analyze_expression(SemanticAnalyzer *analyzer, ASTNode *node);

static DataType analyze_binary_op(SemanticAnalyzer *analyzer, ASTNode *node) {
    DataType left_type = analyze_expression(analyzer, node->binary_op.left);
    DataType right_type = analyze_expression(analyzer, node->binary_op.right);
    
    if (left_type == TYPE_VOID || right_type == TYPE_VOID) {
        error("Cannot use void type in binary operation");
        analyzer->has_errors = true;
        return TYPE_INT;
    }
    
    // Check type compatibility
    if (!is_compatible_type(left_type, right_type) && 
        !is_compatible_type(right_type, left_type)) {
        error("Type mismatch in binary operation");
        analyzer->has_errors = true;
    }
    
    return get_binary_op_result_type(left_type, right_type, node->binary_op.operator);
}

static DataType analyze_unary_op(SemanticAnalyzer *analyzer, ASTNode *node) {
    DataType operand_type = analyze_expression(analyzer, node->unary_op.operand);
    
    if (operand_type == TYPE_VOID) {
        error("Cannot use void type in unary operation");
        analyzer->has_errors = true;
        return TYPE_INT;
    }
    
    return operand_type;
}

static DataType analyze_function_call(SemanticAnalyzer *analyzer, ASTNode *node) {
    SymbolEntry *func_symbol = lookup_symbol(analyzer->current_scope, 
                                           node->function_call.name);
    
    if (!func_symbol) {
        error("Undefined function '%s'", node->function_call.name);
        analyzer->has_errors = true;
        return TYPE_INT;
    }
    
    if (!func_symbol->is_function) {
        error("'%s' is not a function", node->function_call.name);
        analyzer->has_errors = true;
        return TYPE_INT;
    }
    
    // Analyze arguments
    for (int i = 0; i < node->function_call.arg_count; i++) {
        analyze_expression(analyzer, node->function_call.args[i]);
    }
    
    return func_symbol->type;
}

static DataType analyze_identifier(SemanticAnalyzer *analyzer, ASTNode *node) {
    SymbolEntry *symbol = lookup_symbol(analyzer->current_scope, 
                                      node->identifier.name);
    
    if (!symbol) {
        error("Undefined identifier '%s'", node->identifier.name);
        analyzer->has_errors = true;
        return TYPE_INT;
    }
    
    return symbol->type;
}

static DataType analyze_array_access(SemanticAnalyzer *analyzer, ASTNode *node) {
    DataType array_type = analyze_expression(analyzer, node->array_access.array);
    DataType index_type = analyze_expression(analyzer, node->array_access.index);
    
    if (index_type != TYPE_INT) {
        error("Array index must be integer type");
        analyzer->has_errors = true;
    }
    
    // For simplicity, assume array access returns the base type
    return array_type;
}

static DataType analyze_expression(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node) return TYPE_VOID;
    
    switch (node->type) {
        case AST_NUMBER:
            return node->data_type;
            
        case AST_STRING:
            return TYPE_POINTER; // char*
            
        case AST_IDENTIFIER:
            return analyze_identifier(analyzer, node);
            
        case AST_BINARY_OP:
            return analyze_binary_op(analyzer, node);
            
        case AST_UNARY_OP:
            return analyze_unary_op(analyzer, node);
            
        case AST_FUNCTION_CALL:
            return analyze_function_call(analyzer, node);
            
        case AST_ARRAY_ACCESS:
            return analyze_array_access(analyzer, node);
            
        case AST_ASSIGNMENT: {
            DataType target_type = analyze_expression(analyzer, node->assignment.target);
            DataType value_type = analyze_expression(analyzer, node->assignment.value);
            
            if (!is_compatible_type(value_type, target_type)) {
                error("Type mismatch in assignment");
                analyzer->has_errors = true;
            }
            
            return target_type;
        }
        
        default:
            error("Unknown expression type in semantic analysis");
            analyzer->has_errors = true;
            return TYPE_INT;
    }
}

static void analyze_variable_declaration(SemanticAnalyzer *analyzer, ASTNode *node) {
    // Add symbol to current scope
    SymbolEntry *symbol = add_symbol(analyzer->current_scope,
                                   node->var_decl.name,
                                   node->var_decl.var_type,
                                   false, // not a function
                                   node->var_decl.array_size > 0,
                                   node->var_decl.array_size);
    
    if (!symbol) {
        analyzer->has_errors = true;
        return;
    }
    
    // Check initialization
    if (node->var_decl.init_value) {
        DataType init_type = analyze_expression(analyzer, node->var_decl.init_value);
        
        if (!is_compatible_type(init_type, node->var_decl.var_type)) {
            error("Type mismatch in variable initialization");
            analyzer->has_errors = true;
        }
    }
}

static void analyze_if_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    DataType cond_type = analyze_expression(analyzer, node->if_stmt.condition);
    
    // For C, any non-zero value is true, so we allow numeric types
    if (cond_type == TYPE_VOID) {
        error("If condition cannot be void type");
        analyzer->has_errors = true;
    }
    
    analyze_statement(analyzer, node->if_stmt.then_stmt);
    if (node->if_stmt.else_stmt) {
        analyze_statement(analyzer, node->if_stmt.else_stmt);
    }
}

static void analyze_while_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    DataType cond_type = analyze_expression(analyzer, node->while_stmt.condition);
    
    if (cond_type == TYPE_VOID) {
        error("While condition cannot be void type");
        analyzer->has_errors = true;
    }
    
    analyze_statement(analyzer, node->while_stmt.body);
}

static void analyze_for_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (node->for_stmt.init) {
        analyze_statement(analyzer, node->for_stmt.init);
    }
    
    if (node->for_stmt.condition) {
        DataType cond_type = analyze_expression(analyzer, node->for_stmt.condition);
        if (cond_type == TYPE_VOID) {
            error("For condition cannot be void type");
            analyzer->has_errors = true;
        }
    }
    
    if (node->for_stmt.update) {
        analyze_statement(analyzer, node->for_stmt.update);
    }
    
    analyze_statement(analyzer, node->for_stmt.body);
}

static void analyze_parallel_for(SemanticAnalyzer *analyzer, ASTNode *node) {
    DataType start_type = analyze_expression(analyzer, node->parallel_for.start);
    DataType end_type = analyze_expression(analyzer, node->parallel_for.end);
    
    if (start_type != TYPE_INT || end_type != TYPE_INT) {
        error("Parallel for range must be integer type");
        analyzer->has_errors = true;
    }
    
    analyze_statement(analyzer, node->parallel_for.body);
}

static void analyze_return_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    DataType return_type = TYPE_VOID;
    
    if (node->return_stmt.value) {
        return_type = analyze_expression(analyzer, node->return_stmt.value);
    }
    
    if (!is_compatible_type(return_type, analyzer->current_function_return_type)) {
        error("Return type mismatch");
        analyzer->has_errors = true;
    }
}

static void analyze_block(SemanticAnalyzer *analyzer, ASTNode *node) {
    enter_scope(analyzer);
    
    for (int i = 0; i < node->block.statement_count; i++) {
        analyze_statement(analyzer, node->block.statements[i]);
    }
    
    exit_scope(analyzer);
}

static void analyze_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_VARIABLE_DECL:
            analyze_variable_declaration(analyzer, node);
            break;
            
        case AST_IF_STMT:
            analyze_if_statement(analyzer, node);
            break;
            
        case AST_WHILE_STMT:
            analyze_while_statement(analyzer, node);
            break;
            
        case AST_FOR_STMT:
            analyze_for_statement(analyzer, node);
            break;
            
        case AST_PARALLEL_FOR:
            analyze_parallel_for(analyzer, node);
            break;
            
        case AST_RETURN_STMT:
            analyze_return_statement(analyzer, node);
            break;
            
        case AST_BLOCK:
            analyze_block(analyzer, node);
            break;
            
        case AST_ASSIGNMENT:
        case AST_FUNCTION_CALL:
        case AST_BINARY_OP:
        case AST_UNARY_OP:
            // Expression statements
            analyze_expression(analyzer, node);
            break;
            
        default:
            error("Unknown statement type in semantic analysis");
            analyzer->has_errors = true;
            break;
    }
}

static void analyze_function(SemanticAnalyzer *analyzer, ASTNode *node) {
    // Add function to global scope first
    add_symbol(analyzer->current_scope,
               node->function_def.name,
               node->function_def.return_type,
               true, // is function
               false, // not array
               -1);
    
    // Enter function scope
    enter_scope(analyzer);
    
    // Set current function return type for return statement checking
    analyzer->current_function_return_type = node->function_def.return_type;
    
    // Add parameters to function scope
    for (int i = 0; i < node->function_def.param_count; i++) {
        ASTNode *param = node->function_def.params[i];
        add_symbol(analyzer->current_scope,
                   param->var_decl.name,
                   param->var_decl.var_type,
                   false, // not function
                   param->var_decl.array_size > 0,
                   param->var_decl.array_size);
    }
    
    // Analyze function body
    analyze_statement(analyzer, node->function_def.body);
    
    // Exit function scope
    exit_scope(analyzer);
}

static void add_builtin_functions(SemanticAnalyzer *analyzer) {
    // Add standard library functions
    add_symbol(analyzer->current_scope, "printf", TYPE_INT, true, false, -1);
    add_symbol(analyzer->current_scope, "scanf", TYPE_INT, true, false, -1);
    add_symbol(analyzer->current_scope, "malloc", TYPE_POINTER, true, false, -1);
    add_symbol(analyzer->current_scope, "free", TYPE_VOID, true, false, -1);
    
    // Add parallel runtime functions
    add_symbol(analyzer->current_scope, "thread_id", TYPE_INT, true, false, -1);
    add_symbol(analyzer->current_scope, "num_threads", TYPE_INT, true, false, -1);
    add_symbol(analyzer->current_scope, "atomic_add", TYPE_INT, true, false, -1);
    add_symbol(analyzer->current_scope, "atomic_sub", TYPE_INT, true, false, -1);
    add_symbol(analyzer->current_scope, "barrier", TYPE_VOID, true, false, -1);
}

bool semantic_analysis(ASTNode *ast) {
    if (!ast || ast->type != AST_PROGRAM) {
        error("Invalid AST for semantic analysis");
        return false;
    }
    
    SemanticAnalyzer analyzer = {0};
    
    // Create global scope
    analyzer.current_scope = create_scope(NULL);
    analyzer.has_errors = false;
    
    // Add built-in functions
    add_builtin_functions(&analyzer);
    
    // Analyze all functions
    for (int i = 0; i < ast->program.function_count; i++) {
        analyze_function(&analyzer, ast->program.functions[i]);
    }
    
    // Check for main function
    SymbolEntry *main_func = lookup_symbol(analyzer.current_scope, "main");
    if (!main_func) {
        error("No main function found");
        analyzer.has_errors = true;
    } else if (!main_func->is_function) {
        error("main is not a function");
        analyzer.has_errors = true;
    }
    
    // Clean up
    free_scope(analyzer.current_scope);
    
    return !analyzer.has_errors;
}

void print_symbol_table(Scope *scope, int level) {
    if (!scope) return;
    
    printf("Scope level %d:\n", level);
    
    SymbolEntry *entry = scope->symbols;
    while (entry) {
        printf("  %s: type=%d, function=%s, array=%s\n",
               entry->name,
               entry->type,
               entry->is_function ? "yes" : "no",
               entry->is_array ? "yes" : "no");
        entry = entry->next;
    }
    
    printf("\n");
}
