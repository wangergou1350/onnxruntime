#include "riscv_cc.h"

// =============================================================================
// 语义分析器实现
// =============================================================================

// 符号表哈希函数
static unsigned int hash_symbol(const char *name) {
    unsigned int hash = 5381;
    for (const char *p = name; *p; p++) {
        hash = ((hash << 5) + hash) + *p;
    }
    return hash;
}

// =============================================================================
// 符号表管理
// =============================================================================

SymbolTable *create_symbol_table(int scope_level) {
    SymbolTable *table = malloc(sizeof(SymbolTable));
    if (!table) {
        error("内存分配失败：无法创建符号表");
        return NULL;
    }
    
    table->bucket_count = 127; // 质数
    table->buckets = calloc(table->bucket_count, sizeof(Symbol*));
    table->scope_level = scope_level;
    table->parent = NULL;
    table->next = NULL;
    
    return table;
}

static void destroy_symbol_table(SymbolTable *table) {
    if (!table) return;
    
    // 释放所有符号
    for (int i = 0; i < table->bucket_count; i++) {
        Symbol *symbol = table->buckets[i];
        while (symbol) {
            Symbol *next = symbol->next;
            
            free(symbol->name);
            
            // 释放函数特定信息
            if (symbol->symbol_type == SYMBOL_FUNCTION) {
                free(symbol->function_info.param_types);
            }
            
            // 释放复合类型特定信息
            if (symbol->symbol_type == SYMBOL_TYPE && 
                (symbol->data_type == TYPE_STRUCT || symbol->data_type == TYPE_UNION)) {
                for (int j = 0; j < symbol->composite_info.member_count; j++) {
                    // 成员符号会在各自的作用域中被释放
                }
                free(symbol->composite_info.members);
            }
            
            free(symbol);
            symbol = next;
        }
    }
    
    free(table->buckets);
    free(table);
}

Symbol *create_symbol(const char *name, SymbolType type, DataType data_type) {
    Symbol *symbol = malloc(sizeof(Symbol));
    if (!symbol) {
        error("内存分配失败：无法创建符号");
        return NULL;
    }
    
    memset(symbol, 0, sizeof(Symbol));
    symbol->name = strdup(name);
    symbol->symbol_type = type;
    symbol->data_type = data_type;
    symbol->storage_class = STORAGE_AUTO;
    symbol->offset = 0;
    symbol->size = get_type_size(data_type);
    symbol->is_defined = false;
    symbol->is_used = false;
    symbol->is_const = false;
    symbol->is_volatile = false;
    symbol->is_atomic = false;
    symbol->scope_level = 0;
    symbol->next = NULL;
    
    return symbol;
}

void insert_symbol(SymbolTable *table, Symbol *symbol) {
    if (!table || !symbol) return;
    
    unsigned int index = hash_symbol(symbol->name) % table->bucket_count;
    symbol->scope_level = table->scope_level;
    
    // 插入到链表头部
    symbol->next = table->buckets[index];
    table->buckets[index] = symbol;
}

Symbol *lookup_symbol(SymbolTable *table, const char *name) {
    if (!table || !name) return NULL;
    
    unsigned int index = hash_symbol(name) % table->bucket_count;
    Symbol *symbol = table->buckets[index];
    
    while (symbol) {
        if (strcmp(symbol->name, name) == 0) {
            return symbol;
        }
        symbol = symbol->next;
    }
    
    return NULL;
}

static Symbol *lookup_symbol_recursive(SymbolTable *table, const char *name) {
    Symbol *symbol = lookup_symbol(table, name);
    if (symbol) {
        return symbol;
    }
    
    if (table->parent) {
        return lookup_symbol_recursive(table->parent, name);
    }
    
    return NULL;
}

// =============================================================================
// 语义分析器核心函数
// =============================================================================

SemanticAnalyzer *create_semantic_analyzer(void) {
    SemanticAnalyzer *analyzer = malloc(sizeof(SemanticAnalyzer));
    if (!analyzer) {
        error("内存分配失败：无法创建语义分析器");
        return NULL;
    }
    
    analyzer->global_scope = create_symbol_table(0);
    analyzer->current_scope = analyzer->global_scope;
    analyzer->current_scope_level = 0;
    analyzer->has_errors = false;
    analyzer->error_message = NULL;
    
    // 添加内置类型和函数
    insert_builtin_symbols(analyzer);
    
    return analyzer;
}

void destroy_semantic_analyzer(SemanticAnalyzer *analyzer) {
    if (!analyzer) return;
    
    // 销毁所有符号表
    SymbolTable *table = analyzer->global_scope;
    while (table) {
        SymbolTable *next = table->next;
        destroy_symbol_table(table);
        table = next;
    }
    
    free(analyzer->error_message);
    free(analyzer);
}

void enter_scope(SemanticAnalyzer *analyzer) {
    if (!analyzer) return;
    
    SymbolTable *new_scope = create_symbol_table(analyzer->current_scope_level + 1);
    new_scope->parent = analyzer->current_scope;
    analyzer->current_scope = new_scope;
    analyzer->current_scope_level++;
}

void exit_scope(SemanticAnalyzer *analyzer) {
    if (!analyzer || !analyzer->current_scope->parent) return;
    
    SymbolTable *old_scope = analyzer->current_scope;
    analyzer->current_scope = analyzer->current_scope->parent;
    analyzer->current_scope_level--;
    
    // 销毁离开的作用域
    destroy_symbol_table(old_scope);
}

static void semantic_error(SemanticAnalyzer *analyzer, ASTNode *node, const char *format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    
    error("语义错误：%s 在第 %d 行第 %d 列", buffer, node->line, node->column);
    
    analyzer->has_errors = true;
    va_end(args);
}

static void semantic_warning(SemanticAnalyzer *analyzer, ASTNode *node, const char *format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    
    warning("语义警告：%s 在第 %d 行第 %d 列", buffer, node->line, node->column);
    
    va_end(args);
}

// =============================================================================
// 内置符号注册
// =============================================================================

static void insert_builtin_symbols(SemanticAnalyzer *analyzer) {
    // 添加内置函数
    
    // printf函数
    Symbol *printf_sym = create_symbol("printf", SYMBOL_FUNCTION, TYPE_INT);
    printf_sym->function_info.return_type = TYPE_INT;
    printf_sym->function_info.param_count = -1; // 可变参数
    printf_sym->function_info.is_variadic = true;
    printf_sym->is_defined = true;
    insert_symbol(analyzer->global_scope, printf_sym);
    
    // malloc函数
    Symbol *malloc_sym = create_symbol("malloc", SYMBOL_FUNCTION, TYPE_POINTER);
    malloc_sym->function_info.return_type = TYPE_POINTER;
    malloc_sym->function_info.param_count = 1;
    malloc_sym->function_info.param_types = malloc(sizeof(DataType));
    malloc_sym->function_info.param_types[0] = TYPE_LONG; // size_t
    malloc_sym->is_defined = true;
    insert_symbol(analyzer->global_scope, malloc_sym);
    
    // free函数
    Symbol *free_sym = create_symbol("free", SYMBOL_FUNCTION, TYPE_VOID);
    free_sym->function_info.return_type = TYPE_VOID;
    free_sym->function_info.param_count = 1;
    free_sym->function_info.param_types = malloc(sizeof(DataType));
    free_sym->function_info.param_types[0] = TYPE_POINTER;
    free_sym->is_defined = true;
    insert_symbol(analyzer->global_scope, free_sym);
    
    // strlen函数
    Symbol *strlen_sym = create_symbol("strlen", SYMBOL_FUNCTION, TYPE_LONG);
    strlen_sym->function_info.return_type = TYPE_LONG;
    strlen_sym->function_info.param_count = 1;
    strlen_sym->function_info.param_types = malloc(sizeof(DataType));
    strlen_sym->function_info.param_types[0] = TYPE_POINTER;
    strlen_sym->is_defined = true;
    insert_symbol(analyzer->global_scope, strlen_sym);
    
    // strcpy函数
    Symbol *strcpy_sym = create_symbol("strcpy", SYMBOL_FUNCTION, TYPE_POINTER);
    strcpy_sym->function_info.return_type = TYPE_POINTER;
    strcpy_sym->function_info.param_count = 2;
    strcpy_sym->function_info.param_types = malloc(2 * sizeof(DataType));
    strcpy_sym->function_info.param_types[0] = TYPE_POINTER;
    strcpy_sym->function_info.param_types[1] = TYPE_POINTER;
    strcpy_sym->is_defined = true;
    insert_symbol(analyzer->global_scope, strcpy_sym);
}

// =============================================================================
// 类型检查
// =============================================================================

static bool is_compatible_type(DataType type1, DataType type2) {
    if (type1 == type2) return true;
    
    // 整数类型之间可以隐式转换
    if ((type1 == TYPE_CHAR || type1 == TYPE_SHORT || type1 == TYPE_INT || type1 == TYPE_LONG) &&
        (type2 == TYPE_CHAR || type2 == TYPE_SHORT || type2 == TYPE_INT || type2 == TYPE_LONG)) {
        return true;
    }
    
    // 浮点类型之间可以隐式转换
    if ((type1 == TYPE_FLOAT || type1 == TYPE_DOUBLE) &&
        (type2 == TYPE_FLOAT || type2 == TYPE_DOUBLE)) {
        return true;
    }
    
    // 整数和浮点之间可以隐式转换
    if ((type1 == TYPE_CHAR || type1 == TYPE_SHORT || type1 == TYPE_INT || type1 == TYPE_LONG) &&
        (type2 == TYPE_FLOAT || type2 == TYPE_DOUBLE)) {
        return true;
    }
    
    if ((type1 == TYPE_FLOAT || type1 == TYPE_DOUBLE) &&
        (type2 == TYPE_CHAR || type2 == TYPE_SHORT || type2 == TYPE_INT || type2 == TYPE_LONG)) {
        return true;
    }
    
    return false;
}

static DataType get_promoted_type(DataType type1, DataType type2) {
    // 类型提升规则
    if (type1 == TYPE_DOUBLE || type2 == TYPE_DOUBLE) return TYPE_DOUBLE;
    if (type1 == TYPE_FLOAT || type2 == TYPE_FLOAT) return TYPE_FLOAT;
    if (type1 == TYPE_LONG || type2 == TYPE_LONG) return TYPE_LONG;
    if (type1 == TYPE_INT || type2 == TYPE_INT) return TYPE_INT;
    if (type1 == TYPE_SHORT || type2 == TYPE_SHORT) return TYPE_SHORT;
    return TYPE_CHAR;
}

static bool is_lvalue(ASTNode *node) {
    switch (node->type) {
        case AST_IDENTIFIER:
            return true;
        case AST_INDEX_EXPR:
            return true;
        case AST_MEMBER_EXPR:
            return true;
        case AST_UNARY_EXPR:
            return node->unary_expr.operator == TOKEN_MULTIPLY; // 解引用
        default:
            return false;
    }
}

// =============================================================================
// 表达式类型检查
// =============================================================================

static DataType analyze_expression(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node) return TYPE_UNKNOWN;
    
    switch (node->type) {
        case AST_NUMBER_LITERAL:
            node->literal.literal_type = TYPE_INT;
            return TYPE_INT;
            
        case AST_FLOAT_LITERAL:
            node->literal.literal_type = TYPE_FLOAT;
            return TYPE_FLOAT;
            
        case AST_CHAR_LITERAL:
            node->literal.literal_type = TYPE_CHAR;
            return TYPE_CHAR;
            
        case AST_STRING_LITERAL:
            node->literal.literal_type = TYPE_POINTER;
            return TYPE_POINTER;
            
        case AST_IDENTIFIER: {
            Symbol *symbol = lookup_symbol_recursive(analyzer->current_scope, node->identifier.name);
            if (!symbol) {
                semantic_error(analyzer, node, "未声明的标识符 '%s'", node->identifier.name);
                return TYPE_UNKNOWN;
            }
            symbol->is_used = true;
            node->identifier.data_type = symbol->data_type;
            return symbol->data_type;
        }
        
        case AST_BINARY_EXPR: {
            DataType left_type = analyze_expression(analyzer, node->binary_expr.left);
            DataType right_type = analyze_expression(analyzer, node->binary_expr.right);
            
            switch (node->binary_expr.operator) {
                case TOKEN_PLUS:
                case TOKEN_MINUS:
                case TOKEN_MULTIPLY:
                case TOKEN_DIVIDE:
                case TOKEN_MODULO:
                    if (!is_compatible_type(left_type, right_type)) {
                        semantic_warning(analyzer, node, "类型不兼容的算术运算");
                    }
                    node->binary_expr.result_type = get_promoted_type(left_type, right_type);
                    return node->binary_expr.result_type;
                    
                case TOKEN_LESS:
                case TOKEN_GREATER:
                case TOKEN_LESS_EQUAL:
                case TOKEN_GREATER_EQUAL:
                case TOKEN_EQUAL:
                case TOKEN_NOT_EQUAL:
                    if (!is_compatible_type(left_type, right_type)) {
                        semantic_warning(analyzer, node, "类型不兼容的比较运算");
                    }
                    node->binary_expr.result_type = TYPE_INT;
                    return TYPE_INT;
                    
                case TOKEN_LOGICAL_AND:
                case TOKEN_LOGICAL_OR:
                    node->binary_expr.result_type = TYPE_INT;
                    return TYPE_INT;
                    
                case TOKEN_BITWISE_AND:
                case TOKEN_BITWISE_OR:
                case TOKEN_BITWISE_XOR:
                case TOKEN_LSHIFT:
                case TOKEN_RSHIFT:
                    if (!is_signed_type(left_type) || !is_signed_type(right_type)) {
                        semantic_warning(analyzer, node, "位运算应用于非整数类型");
                    }
                    node->binary_expr.result_type = get_promoted_type(left_type, right_type);
                    return node->binary_expr.result_type;
                    
                default:
                    semantic_error(analyzer, node, "不支持的二元运算符");
                    return TYPE_UNKNOWN;
            }
        }
        
        case AST_UNARY_EXPR: {
            DataType operand_type = analyze_expression(analyzer, node->unary_expr.operand);
            
            switch (node->unary_expr.operator) {
                case TOKEN_PLUS:
                case TOKEN_MINUS:
                    if (!is_signed_type(operand_type) && !is_float_type(operand_type)) {
                        semantic_warning(analyzer, node, "一元算术运算符应用于非数值类型");
                    }
                    node->unary_expr.result_type = operand_type;
                    return operand_type;
                    
                case TOKEN_LOGICAL_NOT:
                    node->unary_expr.result_type = TYPE_INT;
                    return TYPE_INT;
                    
                case TOKEN_BITWISE_NOT:
                    if (!is_signed_type(operand_type)) {
                        semantic_warning(analyzer, node, "位取反运算符应用于非整数类型");
                    }
                    node->unary_expr.result_type = operand_type;
                    return operand_type;
                    
                case TOKEN_INCREMENT:
                case TOKEN_DECREMENT:
                    if (!is_lvalue(node->unary_expr.operand)) {
                        semantic_error(analyzer, node, "递增/递减运算符的操作数必须是左值");
                    }
                    node->unary_expr.result_type = operand_type;
                    return operand_type;
                    
                case TOKEN_MULTIPLY: // 解引用
                    if (operand_type != TYPE_POINTER) {
                        semantic_error(analyzer, node, "解引用运算符应用于非指针类型");
                    }
                    // 简化：假设指向int
                    node->unary_expr.result_type = TYPE_INT;
                    return TYPE_INT;
                    
                case TOKEN_BITWISE_AND: // 取地址
                    if (!is_lvalue(node->unary_expr.operand)) {
                        semantic_error(analyzer, node, "取地址运算符的操作数必须是左值");
                    }
                    node->unary_expr.result_type = TYPE_POINTER;
                    return TYPE_POINTER;
                    
                default:
                    semantic_error(analyzer, node, "不支持的一元运算符");
                    return TYPE_UNKNOWN;
            }
        }
        
        case AST_ASSIGN_EXPR: {
            if (!is_lvalue(node->assign_expr.left)) {
                semantic_error(analyzer, node, "赋值运算符的左操作数必须是左值");
            }
            
            DataType left_type = analyze_expression(analyzer, node->assign_expr.left);
            DataType right_type = analyze_expression(analyzer, node->assign_expr.right);
            
            if (!is_compatible_type(left_type, right_type)) {
                semantic_warning(analyzer, node, "赋值中的类型不兼容");
            }
            
            return left_type;
        }
        
        case AST_CALL_EXPR: {
            DataType func_type = analyze_expression(analyzer, node->call_expr.function);
            
            // 查找函数符号
            if (node->call_expr.function->type == AST_IDENTIFIER) {
                Symbol *func_symbol = lookup_symbol_recursive(analyzer->current_scope, 
                                                             node->call_expr.function->identifier.name);
                if (!func_symbol) {
                    semantic_error(analyzer, node, "未声明的函数 '%s'", 
                                 node->call_expr.function->identifier.name);
                    return TYPE_UNKNOWN;
                }
                
                if (func_symbol->symbol_type != SYMBOL_FUNCTION) {
                    semantic_error(analyzer, node, "'%s' 不是函数", 
                                 node->call_expr.function->identifier.name);
                    return TYPE_UNKNOWN;
                }
                
                // 检查参数数量
                if (func_symbol->function_info.param_count >= 0 && 
                    node->call_expr.argument_count != func_symbol->function_info.param_count) {
                    semantic_error(analyzer, node, "函数 '%s' 参数数量不匹配：期望 %d，实际 %d",
                                 node->call_expr.function->identifier.name,
                                 func_symbol->function_info.param_count,
                                 node->call_expr.argument_count);
                }
                
                // 检查参数类型
                for (int i = 0; i < node->call_expr.argument_count && 
                     i < func_symbol->function_info.param_count; i++) {
                    DataType arg_type = analyze_expression(analyzer, node->call_expr.arguments[i]);
                    if (func_symbol->function_info.param_types && 
                        !is_compatible_type(arg_type, func_symbol->function_info.param_types[i])) {
                        semantic_warning(analyzer, node, "函数 '%s' 第 %d 个参数类型不匹配",
                                       node->call_expr.function->identifier.name, i + 1);
                    }
                }
                
                node->call_expr.return_type = func_symbol->function_info.return_type;
                return func_symbol->function_info.return_type;
            }
            
            return TYPE_UNKNOWN;
        }
        
        case AST_INDEX_EXPR: {
            DataType array_type = analyze_expression(analyzer, node->index_expr.array);
            DataType index_type = analyze_expression(analyzer, node->index_expr.index);
            
            if (array_type != TYPE_POINTER && array_type != TYPE_ARRAY) {
                semantic_error(analyzer, node, "数组下标运算符应用于非数组类型");
            }
            
            if (!is_signed_type(index_type)) {
                semantic_warning(analyzer, node, "数组下标应该是整数类型");
            }
            
            // 简化：假设数组元素类型为int
            node->index_expr.element_type = TYPE_INT;
            return TYPE_INT;
        }
        
        case AST_MEMBER_EXPR: {
            DataType object_type = analyze_expression(analyzer, node->member_expr.object);
            
            if (node->member_expr.is_pointer_access) {
                if (object_type != TYPE_POINTER) {
                    semantic_error(analyzer, node, "'->' 运算符应用于非指针类型");
                }
            } else {
                if (object_type != TYPE_STRUCT && object_type != TYPE_UNION) {
                    semantic_error(analyzer, node, "'.' 运算符应用于非结构体/联合体类型");
                }
            }
            
            // 简化：假设成员类型为int
            node->member_expr.member_type = TYPE_INT;
            return TYPE_INT;
        }
        
        case AST_SIZEOF_EXPR: {
            if (node->sizeof_expr.expression) {
                analyze_expression(analyzer, node->sizeof_expr.expression);
            }
            return TYPE_LONG; // sizeof返回size_t，这里简化为long
        }
        
        case AST_TERNARY_EXPR: {
            DataType condition_type = analyze_expression(analyzer, node->ternary_expr.condition);
            DataType true_type = analyze_expression(analyzer, node->ternary_expr.true_expr);
            DataType false_type = analyze_expression(analyzer, node->ternary_expr.false_expr);
            
            if (!is_compatible_type(true_type, false_type)) {
                semantic_warning(analyzer, node, "三元运算符的两个分支类型不兼容");
            }
            
            node->ternary_expr.result_type = get_promoted_type(true_type, false_type);
            return node->ternary_expr.result_type;
        }
        
        default:
            semantic_error(analyzer, node, "不支持的表达式类型");
            return TYPE_UNKNOWN;
    }
}

// =============================================================================
// 语句分析
// =============================================================================

static void analyze_statement(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_COMPOUND_STMT:
            enter_scope(analyzer);
            for (int i = 0; i < node->compound_stmt.statement_count; i++) {
                analyze_statement(analyzer, node->compound_stmt.statements[i]);
            }
            exit_scope(analyzer);
            break;
            
        case AST_EXPRESSION_STMT:
            analyze_expression(analyzer, node);
            break;
            
        case AST_IF_STMT:
            analyze_expression(analyzer, node->if_stmt.condition);
            analyze_statement(analyzer, node->if_stmt.then_stmt);
            if (node->if_stmt.else_stmt) {
                analyze_statement(analyzer, node->if_stmt.else_stmt);
            }
            break;
            
        case AST_WHILE_STMT:
            analyze_expression(analyzer, node->while_stmt.condition);
            analyze_statement(analyzer, node->while_stmt.body);
            break;
            
        case AST_FOR_STMT:
            if (node->for_stmt.init) {
                analyze_expression(analyzer, node->for_stmt.init);
            }
            if (node->for_stmt.condition) {
                analyze_expression(analyzer, node->for_stmt.condition);
            }
            if (node->for_stmt.increment) {
                analyze_expression(analyzer, node->for_stmt.increment);
            }
            analyze_statement(analyzer, node->for_stmt.body);
            break;
            
        case AST_PARALLEL_FOR_STMT:
            if (node->parallel_for_stmt.init) {
                analyze_expression(analyzer, node->parallel_for_stmt.init);
            }
            if (node->parallel_for_stmt.condition) {
                analyze_expression(analyzer, node->parallel_for_stmt.condition);
            }
            if (node->parallel_for_stmt.increment) {
                analyze_expression(analyzer, node->parallel_for_stmt.increment);
            }
            analyze_statement(analyzer, node->parallel_for_stmt.body);
            break;
            
        case AST_RETURN_STMT:
            if (node->return_stmt.value) {
                analyze_expression(analyzer, node->return_stmt.value);
            }
            break;
            
        case AST_BREAK_STMT:
        case AST_CONTINUE_STMT:
        case AST_BARRIER_STMT:
            // 这些语句不需要特殊处理
            break;
            
        default:
            // 其他语句类型按表达式处理
            analyze_expression(analyzer, node);
            break;
    }
}

// =============================================================================
// 声明分析
// =============================================================================

static void analyze_variable_declaration(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_VARIABLE_DECL) return;
    
    // 检查是否重复声明
    Symbol *existing = lookup_symbol(analyzer->current_scope, node->var_decl.name);
    if (existing) {
        semantic_error(analyzer, node, "变量 '%s' 重复声明", node->var_decl.name);
        return;
    }
    
    // 创建符号
    Symbol *symbol = create_symbol(node->var_decl.name, SYMBOL_VARIABLE, node->var_decl.data_type);
    symbol->storage_class = node->var_decl.storage_class;
    symbol->is_const = node->var_decl.is_const;
    symbol->is_volatile = node->var_decl.is_volatile;
    symbol->is_atomic = node->var_decl.is_atomic;
    symbol->is_defined = true;
    
    // 计算大小
    if (node->var_decl.array_size > 0) {
        symbol->size = get_type_size(node->var_decl.data_type) * node->var_decl.array_size;
    } else if (node->var_decl.pointer_level > 0) {
        symbol->size = 8; // 64位指针
    } else {
        symbol->size = get_type_size(node->var_decl.data_type);
    }
    
    // 分析初始化表达式
    if (node->var_decl.init_value) {
        DataType init_type = analyze_expression(analyzer, node->var_decl.init_value);
        if (!is_compatible_type(node->var_decl.data_type, init_type)) {
            semantic_warning(analyzer, node, "初始化表达式类型与变量类型不兼容");
        }
    }
    
    // 插入符号表
    insert_symbol(analyzer->current_scope, symbol);
}

static void analyze_function_definition(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION_DEF) return;
    
    // 检查是否重复定义
    Symbol *existing = lookup_symbol(analyzer->global_scope, node->function_def.name);
    if (existing && existing->is_defined) {
        semantic_error(analyzer, node, "函数 '%s' 重复定义", node->function_def.name);
        return;
    }
    
    // 创建函数符号
    Symbol *func_symbol = create_symbol(node->function_def.name, SYMBOL_FUNCTION, 
                                      node->function_def.return_type);
    func_symbol->function_info.return_type = node->function_def.return_type;
    func_symbol->function_info.param_count = node->function_def.parameter_count;
    func_symbol->function_info.is_parallel = node->function_def.is_parallel;
    func_symbol->is_defined = (node->function_def.body != NULL);
    
    // 分配参数类型数组
    if (node->function_def.parameter_count > 0) {
        func_symbol->function_info.param_types = 
            malloc(node->function_def.parameter_count * sizeof(DataType));
        
        for (int i = 0; i < node->function_def.parameter_count; i++) {
            ASTNode *param = node->function_def.parameters[i];
            if (param->type == AST_VARIABLE_DECL) {
                func_symbol->function_info.param_types[i] = param->var_decl.data_type;
            }
        }
    }
    
    // 插入函数符号到全局作用域
    insert_symbol(analyzer->global_scope, func_symbol);
    
    // 分析函数体
    if (node->function_def.body) {
        enter_scope(analyzer);
        
        // 添加参数到局部作用域
        for (int i = 0; i < node->function_def.parameter_count; i++) {
            analyze_variable_declaration(analyzer, node->function_def.parameters[i]);
        }
        
        analyze_statement(analyzer, node->function_def.body);
        exit_scope(analyzer);
    }
}

static void analyze_declaration(SemanticAnalyzer *analyzer, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_VARIABLE_DECL:
            analyze_variable_declaration(analyzer, node);
            break;
        case AST_FUNCTION_DEF:
            analyze_function_definition(analyzer, node);
            break;
        default:
            semantic_error(analyzer, node, "不支持的声明类型");
            break;
    }
}

// =============================================================================
// 主分析函数
// =============================================================================

bool analyze_semantics(SemanticAnalyzer *analyzer, ASTNode *ast) {
    if (!analyzer || !ast) {
        return false;
    }
    
    if (ast->type != AST_PROGRAM) {
        semantic_error(analyzer, ast, "根节点必须是程序节点");
        return false;
    }
    
    // 分析所有顶层声明
    for (int i = 0; i < ast->program.declaration_count; i++) {
        analyze_declaration(analyzer, ast->program.declarations[i]);
    }
    
    // 检查是否有main函数
    Symbol *main_func = lookup_symbol(analyzer->global_scope, "main");
    if (!main_func) {
        warning("程序中没有找到main函数");
    } else if (main_func->symbol_type != SYMBOL_FUNCTION) {
        semantic_error(analyzer, ast, "'main' 不是函数");
    }
    
    return !analyzer->has_errors;
}
