/*
 * X86/X64 并行 C 编译器 - 语义分析器
 * 负责类型检查、符号表管理和语义验证
 */

#include "x86_cc.h"

// =============================================================================
// 符号表实现
// =============================================================================

SymbolTable *symbol_table_create(SymbolTable *parent) {
    SymbolTable *table = (SymbolTable *)safe_malloc(sizeof(SymbolTable));
    table->count = 0;
    table->scope_level = parent ? parent->scope_level + 1 : 0;
    table->parent = parent;
    
    for (int i = 0; i < MAX_SYMBOLS; i++) {
        table->symbols[i] = NULL;
    }
    
    return table;
}

void symbol_table_destroy(SymbolTable *table) {
    if (table) {
        for (int i = 0; i < table->count; i++) {
            symbol_destroy(table->symbols[i]);
        }
        free(table);
    }
}

Symbol *symbol_table_lookup(SymbolTable *table, char *name) {
    // 从当前作用域开始向上查找
    SymbolTable *current = table;
    while (current) {
        for (int i = 0; i < current->count; i++) {
            if (current->symbols[i] && strcmp(current->symbols[i]->name, name) == 0) {
                return current->symbols[i];
            }
        }
        current = current->parent;
    }
    return NULL;
}

Symbol *symbol_table_lookup_current_scope(SymbolTable *table, char *name) {
    for (int i = 0; i < table->count; i++) {
        if (table->symbols[i] && strcmp(table->symbols[i]->name, name) == 0) {
            return table->symbols[i];
        }
    }
    return NULL;
}

bool symbol_table_insert(SymbolTable *table, Symbol *symbol) {
    if (table->count >= MAX_SYMBOLS) {
        return false;
    }
    
    // 检查当前作用域是否已有同名符号
    if (symbol_table_lookup_current_scope(table, symbol->name)) {
        return false;
    }
    
    symbol->scope_level = table->scope_level;
    table->symbols[table->count++] = symbol;
    return true;
}

Symbol *symbol_create(SymbolKind kind, char *name, Type *type) {
    Symbol *symbol = (Symbol *)safe_malloc(sizeof(Symbol));
    symbol->kind = kind;
    symbol->name = safe_strdup(name);
    symbol->type = type;
    symbol->offset = 0;
    symbol->scope_level = 0;
    symbol->is_global = false;
    symbol->is_used = false;
    
    // 初始化特定字段
    if (kind == SYMBOL_FUNCTION) {
        symbol->function.declaration = NULL;
        symbol->function.local_size = 0;
        symbol->function.is_defined = false;
    } else if (kind == SYMBOL_STRUCT) {
        symbol->struct_info.members = NULL;
        symbol->struct_info.member_count = 0;
        symbol->struct_info.size = 0;
        symbol->struct_info.alignment = 1;
    }
    
    return symbol;
}

void symbol_destroy(Symbol *symbol) {
    if (symbol) {
        free(symbol->name);
        if (symbol->kind == SYMBOL_STRUCT && symbol->struct_info.members) {
            for (int i = 0; i < symbol->struct_info.member_count; i++) {
                symbol_destroy(symbol->struct_info.members[i]);
            }
            free(symbol->struct_info.members);
        }
        free(symbol);
    }
}

// =============================================================================
// 类型系统实现
// =============================================================================

Type *type_create_basic(TypeKind kind) {
    Type *type = (Type *)safe_malloc(sizeof(Type));
    type->kind = kind;
    type->base = NULL;
    type->array_size = 0;
    type->struct_def = NULL;
    type->return_type = NULL;
    type->param_types = NULL;
    type->param_count = 0;
    type->is_atomic = false;
    type->is_thread_local = false;
    
    // 设置大小和对齐
    switch (kind) {
        case TYPE_VOID:
            type->size = 0;
            type->align = 1;
            break;
        case TYPE_CHAR:
            type->size = 1;
            type->align = 1;
            break;
        case TYPE_INT:
        case TYPE_ATOMIC_INT:
            type->size = 4;
            type->align = 4;
            break;
        case TYPE_LONG:
        case TYPE_ATOMIC_LONG:
            type->size = 8;
            type->align = 8;
            break;
        case TYPE_FLOAT:
            type->size = 4;
            type->align = 4;
            break;
        case TYPE_DOUBLE:
            type->size = 8;
            type->align = 8;
            break;
        default:
            type->size = 8; // 指针大小 (x64)
            type->align = 8;
            break;
    }
    
    return type;
}

Type *type_create_pointer(Type *base) {
    Type *type = type_create_basic(TYPE_POINTER);
    type->base = base;
    type->size = 8; // x64 指针大小
    type->align = 8;
    return type;
}

Type *type_create_array(Type *base, int size) {
    Type *type = type_create_basic(TYPE_ARRAY);
    type->base = base;
    type->array_size = size;
    type->size = base->size * size;
    type->align = base->align;
    return type;
}

Type *type_create_function(Type *return_type, Type **param_types, int param_count) {
    Type *type = type_create_basic(TYPE_FUNCTION);
    type->return_type = return_type;
    type->param_types = param_types;
    type->param_count = param_count;
    type->size = 8; // 函数指针大小
    type->align = 8;
    return type;
}

Type *type_create_struct(char *name) {
    Type *type = type_create_basic(TYPE_STRUCT);
    // struct_def 将在语义分析时设置
    return type;
}

int type_size(Type *type) {
    if (!type) return 0;
    return type->size;
}

int type_alignment(Type *type) {
    if (!type) return 1;
    return type->align;
}

bool type_compatible(Type *a, Type *b) {
    if (!a || !b) return false;
    
    // 相同类型
    if (a->kind == b->kind) {
        switch (a->kind) {
            case TYPE_POINTER:
                return type_compatible(a->base, b->base);
            case TYPE_ARRAY:
                return type_compatible(a->base, b->base) && a->array_size == b->array_size;
            case TYPE_FUNCTION:
                if (!type_compatible(a->return_type, b->return_type)) return false;
                if (a->param_count != b->param_count) return false;
                for (int i = 0; i < a->param_count; i++) {
                    if (!type_compatible(a->param_types[i], b->param_types[i])) return false;
                }
                return true;
            case TYPE_STRUCT:
                return a->struct_def == b->struct_def;
            default:
                return true;
        }
    }
    
    // 数值类型之间的兼容性
    if ((a->kind == TYPE_CHAR || a->kind == TYPE_INT || a->kind == TYPE_LONG) &&
        (b->kind == TYPE_CHAR || b->kind == TYPE_INT || b->kind == TYPE_LONG)) {
        return true;
    }
    
    if ((a->kind == TYPE_FLOAT || a->kind == TYPE_DOUBLE) &&
        (b->kind == TYPE_FLOAT || b->kind == TYPE_DOUBLE)) {
        return true;
    }
    
    // 指针和数组的兼容性
    if (a->kind == TYPE_POINTER && b->kind == TYPE_ARRAY) {
        return type_compatible(a->base, b->base);
    }
    
    if (a->kind == TYPE_ARRAY && b->kind == TYPE_POINTER) {
        return type_compatible(a->base, b->base);
    }
    
    return false;
}

Type *type_promote(Type *type) {
    if (!type) return NULL;
    
    switch (type->kind) {
        case TYPE_CHAR:
            return type_create_basic(TYPE_INT);
        case TYPE_FLOAT:
            return type_create_basic(TYPE_DOUBLE);
        default:
            return type;
    }
}

void type_destroy(Type *type) {
    if (type) {
        if (type->param_types) {
            for (int i = 0; i < type->param_count; i++) {
                type_destroy(type->param_types[i]);
            }
            free(type->param_types);
        }
        free(type);
    }
}

// =============================================================================
// 语义分析实现
// =============================================================================

static bool semantic_error(const char *format, ...) {
    va_list args;
    va_start(args, format);
    printf("Semantic Error: ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
    return false;
}

static void register_builtin_functions(SymbolTable *global_scope) {
    // printf 函数
    Type **printf_params = (Type **)safe_malloc(sizeof(Type*) * 1);
    printf_params[0] = type_create_pointer(type_create_basic(TYPE_CHAR));
    Type *printf_type = type_create_function(type_create_basic(TYPE_INT), printf_params, 1);
    Symbol *printf_symbol = symbol_create(SYMBOL_FUNCTION, "printf", printf_type);
    printf_symbol->is_global = true;
    printf_symbol->function.is_defined = true;
    symbol_table_insert(global_scope, printf_symbol);
    
    // malloc 函数
    Type **malloc_params = (Type **)safe_malloc(sizeof(Type*) * 1);
    malloc_params[0] = type_create_basic(TYPE_LONG);
    Type *malloc_type = type_create_function(type_create_pointer(type_create_basic(TYPE_VOID)), malloc_params, 1);
    Symbol *malloc_symbol = symbol_create(SYMBOL_FUNCTION, "malloc", malloc_type);
    malloc_symbol->is_global = true;
    malloc_symbol->function.is_defined = true;
    symbol_table_insert(global_scope, malloc_symbol);
    
    // free 函数
    Type **free_params = (Type **)safe_malloc(sizeof(Type*) * 1);
    free_params[0] = type_create_pointer(type_create_basic(TYPE_VOID));
    Type *free_type = type_create_function(type_create_basic(TYPE_VOID), free_params, 1);
    Symbol *free_symbol = symbol_create(SYMBOL_FUNCTION, "free", free_type);
    free_symbol->is_global = true;
    free_symbol->function.is_defined = true;
    symbol_table_insert(global_scope, free_symbol);
    
    // 原子操作函数
    // atomic_add
    Type **atomic_add_params = (Type **)safe_malloc(sizeof(Type*) * 2);
    atomic_add_params[0] = type_create_pointer(type_create_basic(TYPE_ATOMIC_INT));
    atomic_add_params[1] = type_create_basic(TYPE_INT);
    Type *atomic_add_type = type_create_function(type_create_basic(TYPE_INT), atomic_add_params, 2);
    Symbol *atomic_add_symbol = symbol_create(SYMBOL_FUNCTION, "atomic_add", atomic_add_type);
    atomic_add_symbol->is_global = true;
    atomic_add_symbol->function.is_defined = true;
    symbol_table_insert(global_scope, atomic_add_symbol);
    
    // atomic_load
    Type **atomic_load_params = (Type **)safe_malloc(sizeof(Type*) * 1);
    atomic_load_params[0] = type_create_pointer(type_create_basic(TYPE_ATOMIC_INT));
    Type *atomic_load_type = type_create_function(type_create_basic(TYPE_INT), atomic_load_params, 1);
    Symbol *atomic_load_symbol = symbol_create(SYMBOL_FUNCTION, "atomic_load", atomic_load_type);
    atomic_load_symbol->is_global = true;
    atomic_load_symbol->function.is_defined = true;
    symbol_table_insert(global_scope, atomic_load_symbol);
    
    // atomic_store
    Type **atomic_store_params = (Type **)safe_malloc(sizeof(Type*) * 2);
    atomic_store_params[0] = type_create_pointer(type_create_basic(TYPE_ATOMIC_INT));
    atomic_store_params[1] = type_create_basic(TYPE_INT);
    Type *atomic_store_type = type_create_function(type_create_basic(TYPE_VOID), atomic_store_params, 2);
    Symbol *atomic_store_symbol = symbol_create(SYMBOL_FUNCTION, "atomic_store", atomic_store_type);
    atomic_store_symbol->is_global = true;
    atomic_store_symbol->function.is_defined = true;
    symbol_table_insert(global_scope, atomic_store_symbol);
}

bool semantic_analyze(ASTNode *root, SymbolTable *global_scope) {
    if (!root || !global_scope) return false;
    
    // 注册内置函数
    register_builtin_functions(global_scope);
    
    return semantic_check_node(root, global_scope);
}

Type *semantic_check_expression(ASTNode *expr, SymbolTable *scope) {
    if (!expr) return NULL;
    
    switch (expr->type) {
        case AST_NUMBER:
            expr->data_type = type_create_basic(TYPE_INT);
            return expr->data_type;
            
        case AST_FLOAT:
            expr->data_type = type_create_basic(TYPE_DOUBLE);
            return expr->data_type;
            
        case AST_STRING:
            expr->data_type = type_create_pointer(type_create_basic(TYPE_CHAR));
            return expr->data_type;
            
        case AST_CHAR:
            expr->data_type = type_create_basic(TYPE_CHAR);
            return expr->data_type;
            
        case AST_IDENTIFIER: {
            Symbol *symbol = symbol_table_lookup(scope, expr->identifier.name);
            if (!symbol) {
                semantic_error("Undefined identifier: %s", expr->identifier.name);
                return NULL;
            }
            symbol->is_used = true;
            expr->identifier.symbol = symbol;
            expr->data_type = symbol->type;
            return symbol->type;
        }
        
        case AST_BINARY_OP: {
            Type *left_type = semantic_check_expression(expr->binary_op.left, scope);
            Type *right_type = semantic_check_expression(expr->binary_op.right, scope);
            
            if (!left_type || !right_type) return NULL;
            
            // 算术运算
            if (expr->binary_op.operator == TOKEN_PLUS || expr->binary_op.operator == TOKEN_MINUS ||
                expr->binary_op.operator == TOKEN_MULTIPLY || expr->binary_op.operator == TOKEN_DIVIDE ||
                expr->binary_op.operator == TOKEN_MODULO) {
                
                if (!type_compatible(left_type, right_type)) {
                    semantic_error("Incompatible types in binary operation");
                    return NULL;
                }
                
                // 类型提升
                if (left_type->kind == TYPE_DOUBLE || right_type->kind == TYPE_DOUBLE) {
                    expr->data_type = type_create_basic(TYPE_DOUBLE);
                } else if (left_type->kind == TYPE_FLOAT || right_type->kind == TYPE_FLOAT) {
                    expr->data_type = type_create_basic(TYPE_FLOAT);
                } else if (left_type->kind == TYPE_LONG || right_type->kind == TYPE_LONG) {
                    expr->data_type = type_create_basic(TYPE_LONG);
                } else {
                    expr->data_type = type_create_basic(TYPE_INT);
                }
                
                return expr->data_type;
            }
            
            // 比较运算
            if (expr->binary_op.operator == TOKEN_EQ || expr->binary_op.operator == TOKEN_NE ||
                expr->binary_op.operator == TOKEN_LT || expr->binary_op.operator == TOKEN_LE ||
                expr->binary_op.operator == TOKEN_GT || expr->binary_op.operator == TOKEN_GE) {
                
                if (!type_compatible(left_type, right_type)) {
                    semantic_error("Incompatible types in comparison");
                    return NULL;
                }
                
                expr->data_type = type_create_basic(TYPE_INT);
                return expr->data_type;
            }
            
            // 逻辑运算
            if (expr->binary_op.operator == TOKEN_AND || expr->binary_op.operator == TOKEN_OR) {
                expr->data_type = type_create_basic(TYPE_INT);
                return expr->data_type;
            }
            
            // 位运算
            if (expr->binary_op.operator == TOKEN_BIT_AND || expr->binary_op.operator == TOKEN_BIT_OR ||
                expr->binary_op.operator == TOKEN_BIT_XOR || expr->binary_op.operator == TOKEN_LSHIFT ||
                expr->binary_op.operator == TOKEN_RSHIFT) {
                
                if (left_type->kind != TYPE_INT && left_type->kind != TYPE_LONG) {
                    semantic_error("Bitwise operations require integer types");
                    return NULL;
                }
                
                expr->data_type = left_type;
                return expr->data_type;
            }
            
            break;
        }
        
        case AST_UNARY_OP: {
            Type *operand_type = semantic_check_expression(expr->unary_op.operand, scope);
            if (!operand_type) return NULL;
            
            switch (expr->unary_op.operator) {
                case TOKEN_PLUS:
                case TOKEN_MINUS:
                    if (operand_type->kind != TYPE_INT && operand_type->kind != TYPE_LONG &&
                        operand_type->kind != TYPE_FLOAT && operand_type->kind != TYPE_DOUBLE) {
                        semantic_error("Unary +/- requires numeric type");
                        return NULL;
                    }
                    expr->data_type = operand_type;
                    return operand_type;
                    
                case TOKEN_NOT:
                    expr->data_type = type_create_basic(TYPE_INT);
                    return expr->data_type;
                    
                case TOKEN_BIT_NOT:
                    if (operand_type->kind != TYPE_INT && operand_type->kind != TYPE_LONG) {
                        semantic_error("Bitwise NOT requires integer type");
                        return NULL;
                    }
                    expr->data_type = operand_type;
                    return operand_type;
                    
                case TOKEN_BIT_AND: // 取地址
                    expr->data_type = type_create_pointer(operand_type);
                    return expr->data_type;
                    
                case TOKEN_MULTIPLY: // 解引用
                    if (operand_type->kind != TYPE_POINTER) {
                        semantic_error("Cannot dereference non-pointer type");
                        return NULL;
                    }
                    expr->data_type = operand_type->base;
                    return expr->data_type;
                    
                case TOKEN_INCREMENT:
                case TOKEN_DECREMENT:
                    if (operand_type->kind != TYPE_INT && operand_type->kind != TYPE_LONG &&
                        operand_type->kind != TYPE_POINTER) {
                        semantic_error("Increment/decrement requires numeric or pointer type");
                        return NULL;
                    }
                    expr->data_type = operand_type;
                    return operand_type;
                    
                default:
                    semantic_error("Unknown unary operator");
                    return NULL;
            }
            break;
        }
        
        case AST_ASSIGN: {
            Type *left_type = semantic_check_expression(expr->assignment.left, scope);
            Type *right_type = semantic_check_expression(expr->assignment.right, scope);
            
            if (!left_type || !right_type) return NULL;
            
            if (!type_compatible(left_type, right_type)) {
                semantic_error("Incompatible types in assignment");
                return NULL;
            }
            
            expr->data_type = left_type;
            return left_type;
        }
        
        case AST_CALL: {
            Type *func_type = semantic_check_expression(expr->call.function, scope);
            if (!func_type) return NULL;
            
            if (func_type->kind != TYPE_FUNCTION) {
                semantic_error("Cannot call non-function");
                return NULL;
            }
            
            // 检查参数数量
            if (expr->call.arg_count != func_type->param_count) {
                semantic_error("Function call argument count mismatch: expected %d, got %d",
                             func_type->param_count, expr->call.arg_count);
                return NULL;
            }
            
            // 检查参数类型
            for (int i = 0; i < expr->call.arg_count; i++) {
                Type *arg_type = semantic_check_expression(expr->call.args[i], scope);
                if (!arg_type) return NULL;
                
                if (!type_compatible(arg_type, func_type->param_types[i])) {
                    semantic_error("Function call argument %d type mismatch", i + 1);
                    return NULL;
                }
            }
            
            expr->data_type = func_type->return_type;
            return func_type->return_type;
        }
        
        case AST_ARRAY_ACCESS: {
            Type *array_type = semantic_check_expression(expr->array_access.array, scope);
            Type *index_type = semantic_check_expression(expr->array_access.index, scope);
            
            if (!array_type || !index_type) return NULL;
            
            if (array_type->kind != TYPE_ARRAY && array_type->kind != TYPE_POINTER) {
                semantic_error("Cannot index non-array/pointer type");
                return NULL;
            }
            
            if (index_type->kind != TYPE_INT && index_type->kind != TYPE_LONG) {
                semantic_error("Array index must be integer type");
                return NULL;
            }
            
            expr->data_type = array_type->base;
            return array_type->base;
        }
        
        case AST_MEMBER_ACCESS: {
            Type *object_type = semantic_check_expression(expr->member_access.object, scope);
            if (!object_type) return NULL;
            
            Type *struct_type = object_type;
            if (expr->member_access.is_pointer) {
                if (object_type->kind != TYPE_POINTER) {
                    semantic_error("Cannot use -> on non-pointer type");
                    return NULL;
                }
                struct_type = object_type->base;
            }
            
            if (struct_type->kind != TYPE_STRUCT) {
                semantic_error("Cannot access member of non-struct type");
                return NULL;
            }
            
            // 查找结构体成员
            Symbol *struct_symbol = struct_type->struct_def;
            if (!struct_symbol) {
                semantic_error("Incomplete struct type");
                return NULL;
            }
            
            for (int i = 0; i < struct_symbol->struct_info.member_count; i++) {
                Symbol *member = struct_symbol->struct_info.members[i];
                if (strcmp(member->name, expr->member_access.member) == 0) {
                    expr->data_type = member->type;
                    return member->type;
                }
            }
            
            semantic_error("Struct has no member named '%s'", expr->member_access.member);
            return NULL;
        }
        
        case AST_SIZEOF: {
            if (expr->sizeof_expr.type) {
                expr->data_type = type_create_basic(TYPE_LONG);
                return expr->data_type;
            } else if (expr->sizeof_expr.expression) {
                Type *operand_type = semantic_check_expression(expr->sizeof_expr.expression, scope);
                if (!operand_type) return NULL;
                expr->data_type = type_create_basic(TYPE_LONG);
                return expr->data_type;
            } else {
                semantic_error("Invalid sizeof expression");
                return NULL;
            }
        }
        
        case AST_CONDITIONAL: {
            Type *cond_type = semantic_check_expression(expr->conditional.condition, scope);
            Type *true_type = semantic_check_expression(expr->conditional.true_expr, scope);
            Type *false_type = semantic_check_expression(expr->conditional.false_expr, scope);
            
            if (!cond_type || !true_type || !false_type) return NULL;
            
            if (!type_compatible(true_type, false_type)) {
                semantic_error("Conditional expression branches have incompatible types");
                return NULL;
            }
            
            expr->data_type = true_type;
            return true_type;
        }
        
        default:
            semantic_error("Unknown expression type in semantic analysis");
            return NULL;
    }
    
    return NULL;
}

bool semantic_check_assignment(ASTNode *assignment, SymbolTable *scope) {
    Type *left_type = semantic_check_expression(assignment->assignment.left, scope);
    Type *right_type = semantic_check_expression(assignment->assignment.right, scope);
    
    if (!left_type || !right_type) return false;
    
    // 检查左值
    if (assignment->assignment.left->type != AST_IDENTIFIER &&
        assignment->assignment.left->type != AST_ARRAY_ACCESS &&
        assignment->assignment.left->type != AST_MEMBER_ACCESS &&
        assignment->assignment.left->type != AST_UNARY_OP) {
        semantic_error("Invalid left-hand side in assignment");
        return false;
    }
    
    if (!type_compatible(left_type, right_type)) {
        semantic_error("Incompatible types in assignment");
        return false;
    }
    
    return true;
}

bool semantic_check_function_call(ASTNode *call, SymbolTable *scope) {
    Type *func_type = semantic_check_expression(call->call.function, scope);
    if (!func_type) return false;
    
    if (func_type->kind != TYPE_FUNCTION) {
        semantic_error("Cannot call non-function");
        return false;
    }
    
    // 参数数量检查
    if (call->call.arg_count != func_type->param_count) {
        semantic_error("Function call argument count mismatch");
        return false;
    }
    
    // 参数类型检查
    for (int i = 0; i < call->call.arg_count; i++) {
        Type *arg_type = semantic_check_expression(call->call.args[i], scope);
        if (!arg_type) return false;
        
        if (!type_compatible(arg_type, func_type->param_types[i])) {
            semantic_error("Function call argument %d type mismatch", i + 1);
            return false;
        }
    }
    
    return true;
}

bool semantic_check_node(ASTNode *node, SymbolTable *scope) {
    if (!node) return true;
    
    switch (node->type) {
        case AST_BLOCK: {
            // 创建新的作用域
            SymbolTable *block_scope = symbol_table_create(scope);
            
            for (int i = 0; i < node->block.stmt_count; i++) {
                if (!semantic_check_node(node->block.statements[i], block_scope)) {
                    symbol_table_destroy(block_scope);
                    return false;
                }
            }
            
            symbol_table_destroy(block_scope);
            return true;
        }
        
        case AST_VAR_DECL: {
            // 检查是否重复声明
            if (symbol_table_lookup_current_scope(scope, node->var_decl.name)) {
                semantic_error("Variable '%s' already declared in this scope", node->var_decl.name);
                return false;
            }
            
            // 检查初始化表达式
            if (node->var_decl.initializer) {
                Type *init_type = semantic_check_expression(node->var_decl.initializer, scope);
                if (!init_type) return false;
                
                if (!type_compatible(node->var_decl.type, init_type)) {
                    semantic_error("Incompatible type in variable initialization");
                    return false;
                }
            }
            
            // 添加到符号表
            Symbol *symbol = symbol_create(SYMBOL_VAR, node->var_decl.name, node->var_decl.type);
            symbol->is_global = (scope->scope_level == 0);
            if (!symbol_table_insert(scope, symbol)) {
                semantic_error("Failed to insert variable into symbol table");
                symbol_destroy(symbol);
                return false;
            }
            
            return true;
        }
        
        case AST_FUNCTION_DECL: {
            // 检查函数是否重复声明
            Symbol *existing = symbol_table_lookup_current_scope(scope, node->func_decl.name);
            if (existing) {
                if (existing->kind != SYMBOL_FUNCTION) {
                    semantic_error("Name '%s' conflicts with non-function declaration", node->func_decl.name);
                    return false;
                }
                
                // 检查函数签名是否匹配
                if (!type_compatible(existing->type, node->func_decl.return_type)) {
                    semantic_error("Function '%s' redeclared with different return type", node->func_decl.name);
                    return false;
                }
                
                if (node->func_decl.body && existing->function.is_defined) {
                    semantic_error("Function '%s' already defined", node->func_decl.name);
                    return false;
                }
            } else {
                // 创建函数类型
                Type **param_types = NULL;
                if (node->func_decl.param_count > 0) {
                    param_types = (Type **)safe_malloc(sizeof(Type*) * node->func_decl.param_count);
                    for (int i = 0; i < node->func_decl.param_count; i++) {
                        param_types[i] = node->func_decl.parameters[i]->param_decl.type;
                    }
                }
                
                Type *func_type = type_create_function(node->func_decl.return_type, param_types, node->func_decl.param_count);
                
                Symbol *func_symbol = symbol_create(SYMBOL_FUNCTION, node->func_decl.name, func_type);
                func_symbol->is_global = true;
                func_symbol->function.declaration = node;
                func_symbol->function.is_defined = (node->func_decl.body != NULL);
                
                if (!symbol_table_insert(scope, func_symbol)) {
                    semantic_error("Failed to insert function into symbol table");
                    symbol_destroy(func_symbol);
                    return false;
                }
                
                existing = func_symbol;
            }
            
            // 如果有函数体，检查函数体
            if (node->func_decl.body) {
                SymbolTable *func_scope = symbol_table_create(scope);
                
                // 添加参数到函数作用域
                for (int i = 0; i < node->func_decl.param_count; i++) {
                    ASTNode *param = node->func_decl.parameters[i];
                    Symbol *param_symbol = symbol_create(SYMBOL_PARAM, param->param_decl.name, param->param_decl.type);
                    if (!symbol_table_insert(func_scope, param_symbol)) {
                        semantic_error("Failed to insert parameter into symbol table");
                        symbol_destroy(param_symbol);
                        symbol_table_destroy(func_scope);
                        return false;
                    }
                }
                
                bool result = semantic_check_node(node->func_decl.body, func_scope);
                symbol_table_destroy(func_scope);
                
                if (!result) return false;
                existing->function.is_defined = true;
            }
            
            return true;
        }
        
        case AST_IF: {
            Type *cond_type = semantic_check_expression(node->if_stmt.condition, scope);
            if (!cond_type) return false;
            
            if (!semantic_check_node(node->if_stmt.then_stmt, scope)) return false;
            
            if (node->if_stmt.else_stmt) {
                return semantic_check_node(node->if_stmt.else_stmt, scope);
            }
            
            return true;
        }
        
        case AST_FOR: {
            SymbolTable *for_scope = symbol_table_create(scope);
            
            if (node->for_stmt.init && !semantic_check_node(node->for_stmt.init, for_scope)) {
                symbol_table_destroy(for_scope);
                return false;
            }
            
            if (node->for_stmt.condition) {
                Type *cond_type = semantic_check_expression(node->for_stmt.condition, for_scope);
                if (!cond_type) {
                    symbol_table_destroy(for_scope);
                    return false;
                }
            }
            
            if (node->for_stmt.update) {
                Type *update_type = semantic_check_expression(node->for_stmt.update, for_scope);
                if (!update_type) {
                    symbol_table_destroy(for_scope);
                    return false;
                }
            }
            
            bool result = semantic_check_node(node->for_stmt.body, for_scope);
            symbol_table_destroy(for_scope);
            return result;
        }
        
        case AST_PARALLEL_FOR: {
            // 并行for循环的语义检查类似普通for循环
            SymbolTable *parallel_scope = symbol_table_create(scope);
            
            if (!semantic_check_node(node->parallel_for.init, parallel_scope)) {
                symbol_table_destroy(parallel_scope);
                return false;
            }
            
            if (node->parallel_for.condition) {
                Type *cond_type = semantic_check_expression(node->parallel_for.condition, parallel_scope);
                if (!cond_type) {
                    symbol_table_destroy(parallel_scope);
                    return false;
                }
            }
            
            if (node->parallel_for.update) {
                Type *update_type = semantic_check_expression(node->parallel_for.update, parallel_scope);
                if (!update_type) {
                    symbol_table_destroy(parallel_scope);
                    return false;
                }
            }
            
            bool result = semantic_check_node(node->parallel_for.body, parallel_scope);
            symbol_table_destroy(parallel_scope);
            return result;
        }
        
        case AST_WHILE: {
            Type *cond_type = semantic_check_expression(node->while_stmt.condition, scope);
            if (!cond_type) return false;
            
            return semantic_check_node(node->while_stmt.body, scope);
        }
        
        case AST_RETURN: {
            if (node->return_stmt.expression) {
                Type *expr_type = semantic_check_expression(node->return_stmt.expression, scope);
                if (!expr_type) return false;
                
                // TODO: 检查返回类型是否与函数声明匹配
            }
            
            return true;
        }
        
        case AST_EXPRESSION_STMT: {
            Type *expr_type = semantic_check_expression(node->expr_stmt.expression, scope);
            return expr_type != NULL;
        }
        
        case AST_ASSIGN: {
            return semantic_check_assignment(node, scope);
        }
        
        case AST_CALL: {
            return semantic_check_function_call(node, scope);
        }
        
        case AST_CRITICAL: {
            return semantic_check_node(node->critical.body, scope);
        }
        
        case AST_BARRIER: {
            // barrier() 语句不需要特殊检查
            return true;
        }
        
        default: {
            // 其他表达式节点
            Type *type = semantic_check_expression(node, scope);
            return type != NULL;
        }
    }
}
