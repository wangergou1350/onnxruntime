#include "pcpp.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Symbol table implementation
typedef struct SymbolTable {
    struct SymbolTable *parent;
    Symbol **symbols;
    int symbol_count;
    int capacity;
    char *scope_name;
} SymbolTable;

typedef struct SemanticAnalyzer {
    SymbolTable *current_scope;
    SymbolTable *global_scope;
    ClassInfo **classes;
    int class_count;
    int class_capacity;
    ASTNode *current_class;
    ASTNode *current_function;
    bool has_errors;
    char *error_message;
} SemanticAnalyzer;

static SymbolTable *create_symbol_table(const char *scope_name, SymbolTable *parent) {
    SymbolTable *table = malloc(sizeof(SymbolTable));
    if (!table) return NULL;
    
    table->parent = parent;
    table->capacity = 10;
    table->symbol_count = 0;
    table->symbols = malloc(table->capacity * sizeof(Symbol*));
    table->scope_name = strdup(scope_name);
    
    return table;
}

static void destroy_symbol_table(SymbolTable *table) {
    if (!table) return;
    
    for (int i = 0; i < table->symbol_count; i++) {
        if (table->symbols[i]) {
            free(table->symbols[i]->name);
            free(table->symbols[i]);
        }
    }
    free(table->symbols);
    free(table->scope_name);
    free(table);
}

static Symbol *create_symbol(const char *name, DataType type, SymbolType symbol_type) {
    Symbol *symbol = malloc(sizeof(Symbol));
    if (!symbol) return NULL;
    
    symbol->name = strdup(name);
    symbol->data_type = type;
    symbol->symbol_type = symbol_type;
    symbol->is_initialized = false;
    symbol->is_const = false;
    symbol->is_static = false;
    symbol->access_modifier = ACCESS_PRIVATE;
    symbol->class_name = NULL;
    symbol->array_size = -1;
    symbol->is_reference = false;
    symbol->is_pointer = false;
    
    return symbol;
}

static void add_symbol(SymbolTable *table, Symbol *symbol) {
    if (!table || !symbol) return;
    
    if (table->symbol_count >= table->capacity) {
        table->capacity *= 2;
        table->symbols = realloc(table->symbols, table->capacity * sizeof(Symbol*));
    }
    
    table->symbols[table->symbol_count++] = symbol;
}

static Symbol *lookup_symbol(SymbolTable *table, const char *name) {
    if (!table || !name) return NULL;
    
    // Search current scope
    for (int i = 0; i < table->symbol_count; i++) {
        if (table->symbols[i] && strcmp(table->symbols[i]->name, name) == 0) {
            return table->symbols[i];
        }
    }
    
    // Search parent scopes
    if (table->parent) {
        return lookup_symbol(table->parent, name);
    }
    
    return NULL;
}

static Symbol *lookup_symbol_current_scope(SymbolTable *table, const char *name) {
    if (!table || !name) return NULL;
    
    for (int i = 0; i < table->symbol_count; i++) {
        if (table->symbols[i] && strcmp(table->symbols[i]->name, name) == 0) {
            return table->symbols[i];
        }
    }
    
    return NULL;
}

static SemanticAnalyzer *create_semantic_analyzer() {
    SemanticAnalyzer *analyzer = malloc(sizeof(SemanticAnalyzer));
    if (!analyzer) return NULL;
    
    analyzer->global_scope = create_symbol_table("global", NULL);
    analyzer->current_scope = analyzer->global_scope;
    analyzer->class_capacity = 10;
    analyzer->class_count = 0;
    analyzer->classes = malloc(analyzer->class_capacity * sizeof(ClassInfo*));
    analyzer->current_class = NULL;
    analyzer->current_function = NULL;
    analyzer->has_errors = false;
    analyzer->error_message = NULL;
    
    return analyzer;
}

static void destroy_semantic_analyzer(SemanticAnalyzer *analyzer) {
    if (!analyzer) return;
    
    // Clean up symbol tables
    SymbolTable *current = analyzer->current_scope;
    while (current) {
        SymbolTable *parent = current->parent;
        destroy_symbol_table(current);
        current = parent;
    }
    
    // Clean up class info
    for (int i = 0; i < analyzer->class_count; i++) {
        if (analyzer->classes[i]) {
            free(analyzer->classes[i]->name);
            free(analyzer->classes[i]->base_class);
            
            // Clean up methods
            for (int j = 0; j < analyzer->classes[i]->method_count; j++) {
                if (analyzer->classes[i]->methods[j]) {
                    free(analyzer->classes[i]->methods[j]->name);
                    free(analyzer->classes[i]->methods[j]);
                }
            }
            free(analyzer->classes[i]->methods);
            
            // Clean up members
            for (int j = 0; j < analyzer->classes[i]->member_count; j++) {
                if (analyzer->classes[i]->members[j]) {
                    free(analyzer->classes[i]->members[j]->name);
                    free(analyzer->classes[i]->members[j]);
                }
            }
            free(analyzer->classes[i]->members);
            
            free(analyzer->classes[i]);
        }
    }
    free(analyzer->classes);
    
    if (analyzer->error_message) {
        free(analyzer->error_message);
    }
    
    free(analyzer);
}

static void semantic_error(SemanticAnalyzer *analyzer, const char *format, ...) {
    analyzer->has_errors = true;
    
    va_list args;
    va_start(args, format);
    
    // Calculate required size
    int size = vsnprintf(NULL, 0, format, args) + 1;
    va_end(args);
    
    // Allocate and format message
    analyzer->error_message = malloc(size);
    va_start(args, format);
    vsnprintf(analyzer->error_message, size, format, args);
    va_end(args);
    
    fprintf(stderr, "Semantic Error: %s\n", analyzer->error_message);
}

static void enter_scope(SemanticAnalyzer *analyzer, const char *scope_name) {
    SymbolTable *new_scope = create_symbol_table(scope_name, analyzer->current_scope);
    analyzer->current_scope = new_scope;
}

static void exit_scope(SemanticAnalyzer *analyzer) {
    if (analyzer->current_scope && analyzer->current_scope->parent) {
        SymbolTable *old_scope = analyzer->current_scope;
        analyzer->current_scope = analyzer->current_scope->parent;
        destroy_symbol_table(old_scope);
    }
}

static ClassInfo *find_class(SemanticAnalyzer *analyzer, const char *name) {
    for (int i = 0; i < analyzer->class_count; i++) {
        if (analyzer->classes[i] && strcmp(analyzer->classes[i]->name, name) == 0) {
            return analyzer->classes[i];
        }
    }
    return NULL;
}

static ClassInfo *create_class_info(const char *name) {
    ClassInfo *class_info = malloc(sizeof(ClassInfo));
    if (!class_info) return NULL;
    
    class_info->name = strdup(name);
    class_info->base_class = NULL;
    class_info->has_base_class = false;
    class_info->is_parallel_class = false;
    class_info->vtable_size = 0;
    
    class_info->method_capacity = 10;
    class_info->method_count = 0;
    class_info->methods = malloc(class_info->method_capacity * sizeof(MethodInfo*));
    
    class_info->member_capacity = 10;
    class_info->member_count = 0;
    class_info->members = malloc(class_info->member_capacity * sizeof(MemberInfo*));
    
    return class_info;
}

static void add_class(SemanticAnalyzer *analyzer, ClassInfo *class_info) {
    if (analyzer->class_count >= analyzer->class_capacity) {
        analyzer->class_capacity *= 2;
        analyzer->classes = realloc(analyzer->classes, 
                                  analyzer->class_capacity * sizeof(ClassInfo*));
    }
    
    analyzer->classes[analyzer->class_count++] = class_info;
}

static bool is_type_compatible(DataType from, DataType to) {
    if (from == to) return true;
    
    // Allow implicit conversions
    switch (to) {
        case TYPE_DOUBLE:
            return from == TYPE_FLOAT || from == TYPE_INT;
        case TYPE_FLOAT:
            return from == TYPE_INT;
        case TYPE_INT:
            return from == TYPE_CHAR;
        default:
            return false;
    }
}

static bool can_access_member(SemanticAnalyzer *analyzer, AccessModifier modifier, 
                             const char *member_class, const char *accessing_class) {
    switch (modifier) {
        case ACCESS_PUBLIC:
            return true;
        case ACCESS_PRIVATE:
            return member_class && accessing_class && 
                   strcmp(member_class, accessing_class) == 0;
        case ACCESS_PROTECTED:
            // Allow access from same class or derived classes
            if (member_class && accessing_class) {
                if (strcmp(member_class, accessing_class) == 0) return true;
                
                // Check inheritance hierarchy
                ClassInfo *accessing = find_class(analyzer, accessing_class);
                while (accessing && accessing->has_base_class) {
                    if (strcmp(accessing->base_class, member_class) == 0) {
                        return true;
                    }
                    accessing = find_class(analyzer, accessing->base_class);
                }
            }
            return false;
        default:
            return false;
    }
}

// Forward declarations for analysis functions
static void analyze_node(SemanticAnalyzer *analyzer, ASTNode *node);
static DataType analyze_expression(SemanticAnalyzer *analyzer, ASTNode *node);

static void analyze_variable_declaration(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_VARIABLE_DECL) return;
    
    // Check if variable already exists in current scope
    Symbol *existing = lookup_symbol_current_scope(analyzer->current_scope, 
                                                   node->var_decl.name);
    if (existing) {
        semantic_error(analyzer, "Variable '%s' already declared in current scope", 
                      node->var_decl.name);
        return;
    }
    
    // Create symbol
    Symbol *symbol = create_symbol(node->var_decl.name, node->var_decl.var_type, SYMBOL_VARIABLE);
    symbol->is_const = node->var_decl.is_const;
    symbol->is_static = node->var_decl.is_static;
    symbol->array_size = node->var_decl.array_size;
    
    if (analyzer->current_class) {
        symbol->class_name = strdup(analyzer->current_class->class_def.name);
    }
    
    // Analyze initialization
    if (node->var_decl.init_value) {
        DataType init_type = analyze_expression(analyzer, node->var_decl.init_value);
        if (!is_type_compatible(init_type, node->var_decl.var_type)) {
            semantic_error(analyzer, "Type mismatch in variable initialization for '%s'", 
                          node->var_decl.name);
        }
        symbol->is_initialized = true;
    }
    
    add_symbol(analyzer->current_scope, symbol);
}

static void analyze_function_definition(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || (node->type != AST_FUNCTION_DEF && node->type != AST_METHOD)) return;
    
    ASTNode *old_function = analyzer->current_function;
    analyzer->current_function = node;
    
    // Check if function already exists
    Symbol *existing = lookup_symbol_current_scope(analyzer->current_scope, 
                                                   node->function_def.name);
    if (existing) {
        semantic_error(analyzer, "Function '%s' already declared", 
                      node->function_def.name);
        return;
    }
    
    // Create function symbol
    Symbol *symbol = create_symbol(node->function_def.name, node->function_def.return_type, 
                                  SYMBOL_FUNCTION);
    symbol->is_static = node->function_def.is_static;
    
    if (analyzer->current_class) {
        symbol->class_name = strdup(analyzer->current_class->class_def.name);
    }
    
    add_symbol(analyzer->current_scope, symbol);
    
    // Enter function scope
    enter_scope(analyzer, node->function_def.name);
    
    // Analyze parameters
    for (int i = 0; i < node->function_def.param_count; i++) {
        analyze_variable_declaration(analyzer, node->function_def.params[i]);
    }
    
    // Analyze function body
    if (node->function_def.body) {
        analyze_node(analyzer, node->function_def.body);
    }
    
    // Exit function scope
    exit_scope(analyzer);
    
    analyzer->current_function = old_function;
}

static void analyze_class_definition(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_CLASS_DEF) return;
    
    // Check if class already exists
    ClassInfo *existing = find_class(analyzer, node->class_def.name);
    if (existing) {
        semantic_error(analyzer, "Class '%s' already declared", node->class_def.name);
        return;
    }
    
    // Create class info
    ClassInfo *class_info = create_class_info(node->class_def.name);
    class_info->is_parallel_class = node->class_def.is_parallel_class;
    
    if (node->class_def.has_base_class) {
        ClassInfo *base_class = find_class(analyzer, node->class_def.base_class);
        if (!base_class) {
            semantic_error(analyzer, "Base class '%s' not found", 
                          node->class_def.base_class);
        } else {
            class_info->base_class = strdup(node->class_def.base_class);
            class_info->has_base_class = true;
        }
    }
    
    add_class(analyzer, class_info);
    
    // Set current class context
    ASTNode *old_class = analyzer->current_class;
    analyzer->current_class = node;
    
    // Enter class scope
    enter_scope(analyzer, node->class_def.name);
    
    // Analyze class members
    for (int i = 0; i < node->class_def.member_count; i++) {
        analyze_node(analyzer, node->class_def.members[i]);
    }
    
    // Exit class scope
    exit_scope(analyzer);
    
    analyzer->current_class = old_class;
}

static DataType analyze_binary_operation(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_BINARY_OP) return TYPE_VOID;
    
    DataType left_type = analyze_expression(analyzer, node->binary_op.left);
    DataType right_type = analyze_expression(analyzer, node->binary_op.right);
    
    switch (node->binary_op.operator) {
        case TOKEN_PLUS:
        case TOKEN_MINUS:
        case TOKEN_MULTIPLY:
        case TOKEN_DIVIDE:
        case TOKEN_MODULO:
            // Arithmetic operations
            if (left_type == TYPE_DOUBLE || right_type == TYPE_DOUBLE) {
                return TYPE_DOUBLE;
            } else if (left_type == TYPE_FLOAT || right_type == TYPE_FLOAT) {
                return TYPE_FLOAT;
            } else if (left_type == TYPE_INT || right_type == TYPE_INT) {
                return TYPE_INT;
            } else {
                semantic_error(analyzer, "Invalid operand types for arithmetic operation");
                return TYPE_VOID;
            }
            
        case TOKEN_EQUAL:
        case TOKEN_NOT_EQUAL:
        case TOKEN_LESS:
        case TOKEN_GREATER:
        case TOKEN_LESS_EQUAL:
        case TOKEN_GREATER_EQUAL:
            // Comparison operations
            if (!is_type_compatible(left_type, right_type) && 
                !is_type_compatible(right_type, left_type)) {
                semantic_error(analyzer, "Cannot compare incompatible types");
            }
            return TYPE_BOOL;
            
        case TOKEN_LOGICAL_AND:
        case TOKEN_LOGICAL_OR:
            // Logical operations
            if (left_type != TYPE_BOOL || right_type != TYPE_BOOL) {
                semantic_error(analyzer, "Logical operations require boolean operands");
            }
            return TYPE_BOOL;
            
        case TOKEN_BITWISE_AND:
        case TOKEN_BITWISE_OR:
        case TOKEN_BITWISE_XOR:
        case TOKEN_BITSHIFT_LEFT:
        case TOKEN_BITSHIFT_RIGHT:
            // Bitwise operations
            if (left_type != TYPE_INT || right_type != TYPE_INT) {
                semantic_error(analyzer, "Bitwise operations require integer operands");
            }
            return TYPE_INT;
            
        default:
            semantic_error(analyzer, "Unknown binary operator");
            return TYPE_VOID;
    }
}

static DataType analyze_unary_operation(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_UNARY_OP) return TYPE_VOID;
    
    DataType operand_type = analyze_expression(analyzer, node->unary_op.operand);
    
    switch (node->unary_op.operator) {
        case TOKEN_MINUS:
        case TOKEN_PLUS:
            if (operand_type != TYPE_INT && operand_type != TYPE_FLOAT && 
                operand_type != TYPE_DOUBLE) {
                semantic_error(analyzer, "Unary +/- requires numeric operand");
            }
            return operand_type;
            
        case TOKEN_LOGICAL_NOT:
            if (operand_type != TYPE_BOOL) {
                semantic_error(analyzer, "Logical NOT requires boolean operand");
            }
            return TYPE_BOOL;
            
        case TOKEN_BITWISE_NOT:
            if (operand_type != TYPE_INT) {
                semantic_error(analyzer, "Bitwise NOT requires integer operand");
            }
            return TYPE_INT;
            
        case TOKEN_MULTIPLY: // Dereference
            // TODO: Implement pointer type checking
            return TYPE_INT; // Simplified
            
        case TOKEN_BITWISE_AND: // Address-of
            // TODO: Implement pointer type creation
            return TYPE_INT; // Simplified
            
        default:
            semantic_error(analyzer, "Unknown unary operator");
            return TYPE_VOID;
    }
}

static DataType analyze_member_access(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_MEMBER_ACCESS) return TYPE_VOID;
    
    // Analyze object expression
    DataType object_type = analyze_expression(analyzer, node->member_access.object);
    
    // For now, simplified member access
    // TODO: Implement proper class member lookup
    return TYPE_INT; // Simplified
}

static DataType analyze_function_call(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION_CALL) return TYPE_VOID;
    
    // Look up function
    Symbol *function = lookup_symbol(analyzer->current_scope, node->function_call.name);
    if (!function) {
        semantic_error(analyzer, "Function '%s' not declared", node->function_call.name);
        return TYPE_VOID;
    }
    
    if (function->symbol_type != SYMBOL_FUNCTION) {
        semantic_error(analyzer, "'%s' is not a function", node->function_call.name);
        return TYPE_VOID;
    }
    
    // Analyze arguments
    for (int i = 0; i < node->function_call.arg_count; i++) {
        analyze_expression(analyzer, node->function_call.args[i]);
    }
    
    // TODO: Check argument count and types
    
    return function->data_type;
}

static DataType analyze_expression(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node) return TYPE_VOID;
    
    switch (node->type) {
        case AST_NUMBER:
            return node->number.is_float ? TYPE_FLOAT : TYPE_INT;
            
        case AST_IDENTIFIER: {
            Symbol *symbol = lookup_symbol(analyzer->current_scope, node->identifier.name);
            if (!symbol) {
                semantic_error(analyzer, "Identifier '%s' not declared", 
                              node->identifier.name);
                return TYPE_VOID;
            }
            return symbol->data_type;
        }
        
        case AST_BINARY_OP:
            return analyze_binary_operation(analyzer, node);
            
        case AST_UNARY_OP:
            return analyze_unary_operation(analyzer, node);
            
        case AST_MEMBER_ACCESS:
            return analyze_member_access(analyzer, node);
            
        case AST_FUNCTION_CALL:
            return analyze_function_call(analyzer, node);
            
        case AST_NEW_EXPR:
            // TODO: Implement new expression analysis
            return TYPE_INT; // Simplified
            
        case AST_DELETE_EXPR:
            // TODO: Implement delete expression analysis
            return TYPE_VOID;
            
        default:
            semantic_error(analyzer, "Unknown expression type");
            return TYPE_VOID;
    }
}

static void analyze_assignment(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_ASSIGNMENT) return;
    
    DataType left_type = analyze_expression(analyzer, node->assignment.left);
    DataType right_type = analyze_expression(analyzer, node->assignment.right);
    
    if (!is_type_compatible(right_type, left_type)) {
        semantic_error(analyzer, "Type mismatch in assignment");
    }
}

static void analyze_if_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_IF_STMT) return;
    
    DataType condition_type = analyze_expression(analyzer, node->if_stmt.condition);
    if (condition_type != TYPE_BOOL) {
        semantic_error(analyzer, "If condition must be boolean");
    }
    
    analyze_node(analyzer, node->if_stmt.then_stmt);
    
    if (node->if_stmt.else_stmt) {
        analyze_node(analyzer, node->if_stmt.else_stmt);
    }
}

static void analyze_while_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_WHILE_STMT) return;
    
    DataType condition_type = analyze_expression(analyzer, node->while_stmt.condition);
    if (condition_type != TYPE_BOOL) {
        semantic_error(analyzer, "While condition must be boolean");
    }
    
    analyze_node(analyzer, node->while_stmt.body);
}

static void analyze_for_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_FOR_STMT) return;
    
    enter_scope(analyzer, "for_loop");
    
    if (node->for_stmt.init) {
        analyze_node(analyzer, node->for_stmt.init);
    }
    
    if (node->for_stmt.condition) {
        DataType condition_type = analyze_expression(analyzer, node->for_stmt.condition);
        if (condition_type != TYPE_BOOL) {
            semantic_error(analyzer, "For condition must be boolean");
        }
    }
    
    if (node->for_stmt.update) {
        analyze_node(analyzer, node->for_stmt.update);
    }
    
    analyze_node(analyzer, node->for_stmt.body);
    
    exit_scope(analyzer);
}

static void analyze_return_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_RETURN_STMT) return;
    
    if (!analyzer->current_function) {
        semantic_error(analyzer, "Return statement outside function");
        return;
    }
    
    DataType expected_type = analyzer->current_function->function_def.return_type;
    
    if (node->return_stmt.value) {
        DataType return_type = analyze_expression(analyzer, node->return_stmt.value);
        if (!is_type_compatible(return_type, expected_type)) {
            semantic_error(analyzer, "Return type mismatch");
        }
    } else if (expected_type != TYPE_VOID) {
        semantic_error(analyzer, "Missing return value");
    }
}

static void analyze_block(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_BLOCK) return;
    
    enter_scope(analyzer, "block");
    
    for (int i = 0; i < node->block.statement_count; i++) {
        analyze_node(analyzer, node->block.statements[i]);
    }
    
    exit_scope(analyzer);
}

static void analyze_node(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->program.declaration_count; i++) {
                analyze_node(analyzer, node->program.declarations[i]);
            }
            break;
            
        case AST_VARIABLE_DECL:
            analyze_variable_declaration(analyzer, node);
            break;
            
        case AST_FUNCTION_DEF:
        case AST_METHOD:
            analyze_function_definition(analyzer, node);
            break;
            
        case AST_CLASS_DEF:
            analyze_class_definition(analyzer, node);
            break;
            
        case AST_ASSIGNMENT:
            analyze_assignment(analyzer, node);
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
            
        case AST_RETURN_STMT:
            analyze_return_statement(analyzer, node);
            break;
            
        case AST_BLOCK:
            analyze_block(analyzer, node);
            break;
            
        case AST_BINARY_OP:
        case AST_UNARY_OP:
        case AST_FUNCTION_CALL:
        case AST_MEMBER_ACCESS:
            analyze_expression(analyzer, node);
            break;
            
        default:
            // Expression statements or other nodes
            break;
    }
}

bool analyze_semantics(ASTNode *ast, char **error_message) {
    if (!ast) return false;
    
    SemanticAnalyzer *analyzer = create_semantic_analyzer();
    if (!analyzer) return false;
    
    analyze_node(analyzer, ast);
    
    bool success = !analyzer->has_errors;
    
    if (error_message) {
        if (analyzer->error_message) {
            *error_message = strdup(analyzer->error_message);
        } else {
            *error_message = NULL;
        }
    }
    
    destroy_semantic_analyzer(analyzer);
    
    return success;
}

void print_symbol_table(SymbolTable *table, int indent) {
    if (!table) return;
    
    for (int i = 0; i < indent; i++) printf("  ");
    printf("Scope: %s\n", table->scope_name);
    
    for (int i = 0; i < table->symbol_count; i++) {
        Symbol *symbol = table->symbols[i];
        for (int j = 0; j < indent + 1; j++) printf("  ");
        printf("Symbol: %s, Type: %d, SymbolType: %d\n", 
               symbol->name, symbol->data_type, symbol->symbol_type);
    }
}
