/*
 * X86/X64 并行 C 编译器 - 语法分析器
 * 负责构建抽象语法树 (AST)
 */

#include "x86_cc.h"

// =============================================================================
// 语法分析器实现
// =============================================================================

Parser *parser_create(Lexer *lexer) {
    Parser *parser = (Parser *)safe_malloc(sizeof(Parser));
    parser->lexer = lexer;
    parser->global_scope = symbol_table_create(NULL);
    parser->current_scope = parser->global_scope;
    parser->current_scope_level = 0;
    parser->has_error = false;
    
    // 初始化内置类型
    parser->builtin_types[0] = type_create_basic(TYPE_VOID);
    parser->builtin_types[1] = type_create_basic(TYPE_CHAR);
    parser->builtin_types[2] = type_create_basic(TYPE_INT);
    parser->builtin_types[3] = type_create_basic(TYPE_LONG);
    parser->builtin_types[4] = type_create_basic(TYPE_FLOAT);
    parser->builtin_types[5] = type_create_basic(TYPE_DOUBLE);
    parser->builtin_type_count = 6;
    
    return parser;
}

void parser_destroy(Parser *parser) {
    if (parser) {
        symbol_table_destroy(parser->global_scope);
        for (int i = 0; i < parser->builtin_type_count; i++) {
            type_destroy(parser->builtin_types[i]);
        }
        free(parser);
    }
}

void parser_error(Parser *parser, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(parser->error_msg, MAX_ERROR_MSG, format, args);
    va_end(args);
    parser->has_error = true;
}

static Token *parser_current_token(Parser *parser) {
    return &parser->lexer->current_token;
}

static bool parser_match(Parser *parser, TokenType type) {
    return parser_current_token(parser)->type == type;
}

static bool parser_consume(Parser *parser, TokenType type) {
    if (parser_match(parser, type)) {
        lexer_next_token(parser->lexer);
        return true;
    }
    return false;
}

static bool parser_expect(Parser *parser, TokenType type) {
    if (parser_consume(parser, type)) {
        return true;
    }
    parser_error(parser, "Expected token type %d, got %d", type, parser_current_token(parser)->type);
    return false;
}

// 前向声明
static ASTNode *parse_expression(Parser *parser);
static ASTNode *parse_statement(Parser *parser);
static ASTNode *parse_declaration(Parser *parser);
static Type *parse_type(Parser *parser);

// =============================================================================
// 表达式解析 (递归下降，按优先级)
// =============================================================================

static ASTNode *parse_primary(Parser *parser) {
    Token *token = parser_current_token(parser);
    
    switch (token->type) {
        case TOKEN_NUMBER: {
            ASTNode *node = ast_create_number(token->int_val);
            lexer_next_token(parser->lexer);
            return node;
        }
        
        case TOKEN_FLOAT: {
            ASTNode *node = ast_create_float(token->float_val);
            lexer_next_token(parser->lexer);
            return node;
        }
        
        case TOKEN_STRING: {
            ASTNode *node = ast_create_string(token->value);
            lexer_next_token(parser->lexer);
            return node;
        }
        
        case TOKEN_CHAR: {
            ASTNode *node = ast_create_number((long)token->char_val);
            node->type = AST_CHAR;
            node->char_literal.char_value = token->char_val;
            lexer_next_token(parser->lexer);
            return node;
        }
        
        case TOKEN_IDENTIFIER: {
            ASTNode *node = ast_create_identifier(token->value);
            lexer_next_token(parser->lexer);
            return node;
        }
        
        case TOKEN_LPAREN: {
            lexer_next_token(parser->lexer); // 消费 (
            ASTNode *expr = parse_expression(parser);
            if (!parser_expect(parser, TOKEN_RPAREN)) {
                ast_destroy(expr);
                return NULL;
            }
            return expr;
        }
        
        case TOKEN_SIZEOF: {
            lexer_next_token(parser->lexer); // 消费 sizeof
            if (!parser_expect(parser, TOKEN_LPAREN)) {
                return NULL;
            }
            
            ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
            node->type = AST_SIZEOF;
            node->line = token->line;
            node->column = token->column;
            
            // 尝试解析类型或表达式
            if (parser_match(parser, TOKEN_INT) || parser_match(parser, TOKEN_FLOAT_KW) ||
                parser_match(parser, TOKEN_DOUBLE) || parser_match(parser, TOKEN_CHAR_KW) ||
                parser_match(parser, TOKEN_VOID) || parser_match(parser, TOKEN_STRUCT)) {
                node->sizeof_expr.type = parse_type(parser);
                node->sizeof_expr.expression = NULL;
            } else {
                node->sizeof_expr.type = NULL;
                node->sizeof_expr.expression = parse_expression(parser);
            }
            
            if (!parser_expect(parser, TOKEN_RPAREN)) {
                ast_destroy(node);
                return NULL;
            }
            
            return node;
        }
        
        default:
            parser_error(parser, "Unexpected token in primary expression: %d", token->type);
            return NULL;
    }
}

static ASTNode *parse_postfix(Parser *parser) {
    ASTNode *expr = parse_primary(parser);
    if (!expr) return NULL;
    
    while (true) {
        Token *token = parser_current_token(parser);
        
        switch (token->type) {
            case TOKEN_LBRACKET: {
                // 数组访问: expr[index]
                lexer_next_token(parser->lexer); // 消费 [
                ASTNode *index = parse_expression(parser);
                if (!index || !parser_expect(parser, TOKEN_RBRACKET)) {
                    ast_destroy(expr);
                    ast_destroy(index);
                    return NULL;
                }
                
                ASTNode *array_access = (ASTNode *)safe_malloc(sizeof(ASTNode));
                array_access->type = AST_ARRAY_ACCESS;
                array_access->line = token->line;
                array_access->column = token->column;
                array_access->array_access.array = expr;
                array_access->array_access.index = index;
                
                expr = array_access;
                break;
            }
            
            case TOKEN_LPAREN: {
                // 函数调用: expr(args...)
                lexer_next_token(parser->lexer); // 消费 (
                
                ASTNode **args = NULL;
                int arg_count = 0;
                
                if (!parser_match(parser, TOKEN_RPAREN)) {
                    // 解析参数列表
                    args = (ASTNode **)safe_malloc(sizeof(ASTNode*) * MAX_PARAMS);
                    
                    do {
                        if (arg_count >= MAX_PARAMS) {
                            parser_error(parser, "Too many function arguments");
                            break;
                        }
                        
                        args[arg_count] = parse_expression(parser);
                        if (!args[arg_count]) {
                            for (int i = 0; i < arg_count; i++) {
                                ast_destroy(args[i]);
                            }
                            free(args);
                            ast_destroy(expr);
                            return NULL;
                        }
                        arg_count++;
                    } while (parser_consume(parser, TOKEN_COMMA));
                }
                
                if (!parser_expect(parser, TOKEN_RPAREN)) {
                    for (int i = 0; i < arg_count; i++) {
                        ast_destroy(args[i]);
                    }
                    free(args);
                    ast_destroy(expr);
                    return NULL;
                }
                
                expr = ast_create_call(expr, args, arg_count);
                break;
            }
            
            case TOKEN_DOT: {
                // 成员访问: expr.member
                lexer_next_token(parser->lexer); // 消费 .
                
                if (!parser_match(parser, TOKEN_IDENTIFIER)) {
                    parser_error(parser, "Expected member name after '.'");
                    ast_destroy(expr);
                    return NULL;
                }
                
                ASTNode *member_access = (ASTNode *)safe_malloc(sizeof(ASTNode));
                member_access->type = AST_MEMBER_ACCESS;
                member_access->line = token->line;
                member_access->column = token->column;
                member_access->member_access.object = expr;
                member_access->member_access.member = safe_strdup(parser_current_token(parser)->value);
                member_access->member_access.is_pointer = false;
                
                lexer_next_token(parser->lexer);
                expr = member_access;
                break;
            }
            
            case TOKEN_ARROW: {
                // 指针成员访问: expr->member
                lexer_next_token(parser->lexer); // 消费 ->
                
                if (!parser_match(parser, TOKEN_IDENTIFIER)) {
                    parser_error(parser, "Expected member name after '->'");
                    ast_destroy(expr);
                    return NULL;
                }
                
                ASTNode *member_access = (ASTNode *)safe_malloc(sizeof(ASTNode));
                member_access->type = AST_MEMBER_ACCESS;
                member_access->line = token->line;
                member_access->column = token->column;
                member_access->member_access.object = expr;
                member_access->member_access.member = safe_strdup(parser_current_token(parser)->value);
                member_access->member_access.is_pointer = true;
                
                lexer_next_token(parser->lexer);
                expr = member_access;
                break;
            }
            
            case TOKEN_INCREMENT:
            case TOKEN_DECREMENT: {
                // 后置自增/自减: expr++ / expr--
                ASTNode *unary = ast_create_unary_op(token->type, expr);
                lexer_next_token(parser->lexer);
                expr = unary;
                break;
            }
            
            default:
                return expr;
        }
    }
}

static ASTNode *parse_unary(Parser *parser) {
    Token *token = parser_current_token(parser);
    
    switch (token->type) {
        case TOKEN_PLUS:
        case TOKEN_MINUS:
        case TOKEN_NOT:
        case TOKEN_BIT_NOT:
        case TOKEN_INCREMENT:
        case TOKEN_DECREMENT: {
            TokenType op = token->type;
            lexer_next_token(parser->lexer);
            ASTNode *operand = parse_unary(parser);
            if (!operand) return NULL;
            return ast_create_unary_op(op, operand);
        }
        
        case TOKEN_BIT_AND: {
            // 取地址运算符 &
            lexer_next_token(parser->lexer);
            ASTNode *operand = parse_unary(parser);
            if (!operand) return NULL;
            return ast_create_unary_op(TOKEN_BIT_AND, operand);
        }
        
        case TOKEN_MULTIPLY: {
            // 解引用运算符 *
            lexer_next_token(parser->lexer);
            ASTNode *operand = parse_unary(parser);
            if (!operand) return NULL;
            return ast_create_unary_op(TOKEN_MULTIPLY, operand);
        }
        
        default:
            return parse_postfix(parser);
    }
}

static ASTNode *parse_binary_expr(Parser *parser, int min_precedence) {
    ASTNode *left = parse_unary(parser);
    if (!left) return NULL;
    
    while (true) {
        Token *token = parser_current_token(parser);
        int precedence = 0;
        
        // 设置运算符优先级
        switch (token->type) {
            case TOKEN_OR:
                precedence = 1; break;
            case TOKEN_AND:
                precedence = 2; break;
            case TOKEN_BIT_OR:
                precedence = 3; break;
            case TOKEN_BIT_XOR:
                precedence = 4; break;
            case TOKEN_BIT_AND:
                precedence = 5; break;
            case TOKEN_EQ:
            case TOKEN_NE:
                precedence = 6; break;
            case TOKEN_LT:
            case TOKEN_LE:
            case TOKEN_GT:
            case TOKEN_GE:
                precedence = 7; break;
            case TOKEN_LSHIFT:
            case TOKEN_RSHIFT:
                precedence = 8; break;
            case TOKEN_PLUS:
            case TOKEN_MINUS:
                precedence = 9; break;
            case TOKEN_MULTIPLY:
            case TOKEN_DIVIDE:
            case TOKEN_MODULO:
                precedence = 10; break;
            default:
                return left;
        }
        
        if (precedence < min_precedence) {
            return left;
        }
        
        TokenType op = token->type;
        lexer_next_token(parser->lexer);
        ASTNode *right = parse_binary_expr(parser, precedence + 1);
        if (!right) {
            ast_destroy(left);
            return NULL;
        }
        
        left = ast_create_binary_op(op, left, right);
    }
}

static ASTNode *parse_conditional(Parser *parser) {
    ASTNode *expr = parse_binary_expr(parser, 1);
    if (!expr) return NULL;
    
    if (parser_consume(parser, TOKEN_QUESTION)) {
        ASTNode *true_expr = parse_expression(parser);
        if (!true_expr || !parser_expect(parser, TOKEN_COLON)) {
            ast_destroy(expr);
            ast_destroy(true_expr);
            return NULL;
        }
        
        ASTNode *false_expr = parse_conditional(parser);
        if (!false_expr) {
            ast_destroy(expr);
            ast_destroy(true_expr);
            return NULL;
        }
        
        ASTNode *conditional = (ASTNode *)safe_malloc(sizeof(ASTNode));
        conditional->type = AST_CONDITIONAL;
        conditional->conditional.condition = expr;
        conditional->conditional.true_expr = true_expr;
        conditional->conditional.false_expr = false_expr;
        
        return conditional;
    }
    
    return expr;
}

static ASTNode *parse_assignment(Parser *parser) {
    ASTNode *expr = parse_conditional(parser);
    if (!expr) return NULL;
    
    Token *token = parser_current_token(parser);
    if (token->type == TOKEN_ASSIGN || token->type == TOKEN_PLUS_ASSIGN ||
        token->type == TOKEN_MINUS_ASSIGN || token->type == TOKEN_MUL_ASSIGN ||
        token->type == TOKEN_DIV_ASSIGN) {
        
        TokenType op = token->type;
        lexer_next_token(parser->lexer);
        ASTNode *right = parse_assignment(parser);
        if (!right) {
            ast_destroy(expr);
            return NULL;
        }
        
        return ast_create_assignment(expr, right, op);
    }
    
    return expr;
}

static ASTNode *parse_expression(Parser *parser) {
    return parse_assignment(parser);
}

// =============================================================================
// 类型解析
// =============================================================================

static Type *parse_type(Parser *parser) {
    Token *token = parser_current_token(parser);
    Type *base_type = NULL;
    
    // 解析原子类型修饰符
    bool is_atomic = false;
    bool is_thread_local = false;
    
    if (parser_consume(parser, TOKEN_ATOMIC)) {
        is_atomic = true;
    }
    
    if (parser_consume(parser, TOKEN_THREAD_LOCAL)) {
        is_thread_local = true;
    }
    
    // 解析基本类型
    switch (parser_current_token(parser)->type) {
        case TOKEN_VOID:
            base_type = type_create_basic(TYPE_VOID);
            lexer_next_token(parser->lexer);
            break;
        case TOKEN_CHAR_KW:
            base_type = type_create_basic(TYPE_CHAR);
            lexer_next_token(parser->lexer);
            break;
        case TOKEN_INT:
            base_type = type_create_basic(TYPE_INT);
            lexer_next_token(parser->lexer);
            break;
        case TOKEN_FLOAT_KW:
            base_type = type_create_basic(TYPE_FLOAT);
            lexer_next_token(parser->lexer);
            break;
        case TOKEN_DOUBLE:
            base_type = type_create_basic(TYPE_DOUBLE);
            lexer_next_token(parser->lexer);
            break;
        case TOKEN_STRUCT: {
            lexer_next_token(parser->lexer); // 消费 struct
            if (!parser_match(parser, TOKEN_IDENTIFIER)) {
                parser_error(parser, "Expected struct name");
                return NULL;
            }
            base_type = type_create_struct(parser_current_token(parser)->value);
            lexer_next_token(parser->lexer);
            break;
        }
        default:
            parser_error(parser, "Expected type name");
            return NULL;
    }
    
    // 应用修饰符
    if (is_atomic) {
        if (base_type->kind == TYPE_INT) {
            base_type->kind = TYPE_ATOMIC_INT;
        } else if (base_type->kind == TYPE_LONG) {
            base_type->kind = TYPE_ATOMIC_LONG;
        }
        base_type->is_atomic = true;
    }
    
    if (is_thread_local) {
        base_type->is_thread_local = true;
    }
    
    // 解析指针和数组
    Type *type = base_type;
    
    // 指针
    while (parser_consume(parser, TOKEN_MULTIPLY)) {
        type = type_create_pointer(type);
    }
    
    return type;
}

// =============================================================================
// 语句解析
// =============================================================================

static ASTNode *parse_block(Parser *parser) {
    if (!parser_expect(parser, TOKEN_LBRACE)) {
        return NULL;
    }
    
    ASTNode **statements = (ASTNode **)safe_malloc(sizeof(ASTNode*) * 1000);
    int stmt_count = 0;
    
    while (!parser_match(parser, TOKEN_RBRACE) && !parser_match(parser, TOKEN_EOF)) {
        ASTNode *stmt = parse_statement(parser);
        if (stmt) {
            statements[stmt_count++] = stmt;
        } else if (parser->has_error) {
            // 清理已分配的语句
            for (int i = 0; i < stmt_count; i++) {
                ast_destroy(statements[i]);
            }
            free(statements);
            return NULL;
        }
    }
    
    if (!parser_expect(parser, TOKEN_RBRACE)) {
        for (int i = 0; i < stmt_count; i++) {
            ast_destroy(statements[i]);
        }
        free(statements);
        return NULL;
    }
    
    return ast_create_block(statements, stmt_count);
}

static ASTNode *parse_if_statement(Parser *parser) {
    lexer_next_token(parser->lexer); // 消费 if
    
    if (!parser_expect(parser, TOKEN_LPAREN)) {
        return NULL;
    }
    
    ASTNode *condition = parse_expression(parser);
    if (!condition || !parser_expect(parser, TOKEN_RPAREN)) {
        ast_destroy(condition);
        return NULL;
    }
    
    ASTNode *then_stmt = parse_statement(parser);
    if (!then_stmt) {
        ast_destroy(condition);
        return NULL;
    }
    
    ASTNode *else_stmt = NULL;
    if (parser_consume(parser, TOKEN_ELSE)) {
        else_stmt = parse_statement(parser);
        if (!else_stmt) {
            ast_destroy(condition);
            ast_destroy(then_stmt);
            return NULL;
        }
    }
    
    return ast_create_if(condition, then_stmt, else_stmt);
}

static ASTNode *parse_for_statement(Parser *parser) {
    lexer_next_token(parser->lexer); // 消费 for
    
    if (!parser_expect(parser, TOKEN_LPAREN)) {
        return NULL;
    }
    
    // 初始化部分 (可选)
    ASTNode *init = NULL;
    if (!parser_match(parser, TOKEN_SEMICOLON)) {
        init = parse_expression(parser);
        if (!init) return NULL;
    }
    if (!parser_expect(parser, TOKEN_SEMICOLON)) {
        ast_destroy(init);
        return NULL;
    }
    
    // 条件部分 (可选)
    ASTNode *condition = NULL;
    if (!parser_match(parser, TOKEN_SEMICOLON)) {
        condition = parse_expression(parser);
        if (!condition) {
            ast_destroy(init);
            return NULL;
        }
    }
    if (!parser_expect(parser, TOKEN_SEMICOLON)) {
        ast_destroy(init);
        ast_destroy(condition);
        return NULL;
    }
    
    // 更新部分 (可选)
    ASTNode *update = NULL;
    if (!parser_match(parser, TOKEN_RPAREN)) {
        update = parse_expression(parser);
        if (!update) {
            ast_destroy(init);
            ast_destroy(condition);
            return NULL;
        }
    }
    if (!parser_expect(parser, TOKEN_RPAREN)) {
        ast_destroy(init);
        ast_destroy(condition);
        ast_destroy(update);
        return NULL;
    }
    
    ASTNode *body = parse_statement(parser);
    if (!body) {
        ast_destroy(init);
        ast_destroy(condition);
        ast_destroy(update);
        return NULL;
    }
    
    return ast_create_for(init, condition, update, body);
}

static ASTNode *parse_parallel_for_statement(Parser *parser) {
    lexer_next_token(parser->lexer); // 消费 parallel_for
    
    if (!parser_expect(parser, TOKEN_LPAREN)) {
        return NULL;
    }
    
    // 解析循环变量声明: int i = 0
    Type *var_type = parse_type(parser);
    if (!var_type) return NULL;
    
    if (!parser_match(parser, TOKEN_IDENTIFIER)) {
        parser_error(parser, "Expected loop variable name");
        return NULL;
    }
    
    char *var_name = safe_strdup(parser_current_token(parser)->value);
    lexer_next_token(parser->lexer);
    
    if (!parser_expect(parser, TOKEN_ASSIGN)) {
        free(var_name);
        return NULL;
    }
    
    ASTNode *init_expr = parse_expression(parser);
    if (!init_expr) {
        free(var_name);
        return NULL;
    }
    
    // 创建初始化节点
    ASTNode *init = ast_create_var_decl(var_type, var_name, init_expr);
    
    if (!parser_expect(parser, TOKEN_SEMICOLON)) {
        ast_destroy(init);
        return NULL;
    }
    
    // 条件部分
    ASTNode *condition = parse_expression(parser);
    if (!condition || !parser_expect(parser, TOKEN_SEMICOLON)) {
        ast_destroy(init);
        ast_destroy(condition);
        return NULL;
    }
    
    // 更新部分
    ASTNode *update = parse_expression(parser);
    if (!update || !parser_expect(parser, TOKEN_RPAREN)) {
        ast_destroy(init);
        ast_destroy(condition);
        ast_destroy(update);
        return NULL;
    }
    
    ASTNode *body = parse_statement(parser);
    if (!body) {
        ast_destroy(init);
        ast_destroy(condition);
        ast_destroy(update);
        return NULL;
    }
    
    return ast_create_parallel_for(init, condition, update, body, 0); // 线程数自动确定
}

static ASTNode *parse_while_statement(Parser *parser) {
    lexer_next_token(parser->lexer); // 消费 while
    
    if (!parser_expect(parser, TOKEN_LPAREN)) {
        return NULL;
    }
    
    ASTNode *condition = parse_expression(parser);
    if (!condition || !parser_expect(parser, TOKEN_RPAREN)) {
        ast_destroy(condition);
        return NULL;
    }
    
    ASTNode *body = parse_statement(parser);
    if (!body) {
        ast_destroy(condition);
        return NULL;
    }
    
    return ast_create_while(condition, body);
}

static ASTNode *parse_return_statement(Parser *parser) {
    lexer_next_token(parser->lexer); // 消费 return
    
    ASTNode *expression = NULL;
    if (!parser_match(parser, TOKEN_SEMICOLON)) {
        expression = parse_expression(parser);
        if (!expression) return NULL;
    }
    
    if (!parser_expect(parser, TOKEN_SEMICOLON)) {
        ast_destroy(expression);
        return NULL;
    }
    
    return ast_create_return(expression);
}

static ASTNode *parse_critical_statement(Parser *parser) {
    lexer_next_token(parser->lexer); // 消费 critical
    
    ASTNode *body = parse_statement(parser);
    if (!body) return NULL;
    
    ASTNode *critical = (ASTNode *)safe_malloc(sizeof(ASTNode));
    critical->type = AST_CRITICAL;
    critical->critical.body = body;
    
    return critical;
}

static ASTNode *parse_barrier_statement(Parser *parser) {
    lexer_next_token(parser->lexer); // 消费 barrier
    
    if (!parser_expect(parser, TOKEN_LPAREN) ||
        !parser_expect(parser, TOKEN_RPAREN) ||
        !parser_expect(parser, TOKEN_SEMICOLON)) {
        return NULL;
    }
    
    ASTNode *barrier = (ASTNode *)safe_malloc(sizeof(ASTNode));
    barrier->type = AST_BARRIER;
    
    return barrier;
}

static ASTNode *parse_statement(Parser *parser) {
    Token *token = parser_current_token(parser);
    
    switch (token->type) {
        case TOKEN_LBRACE:
            return parse_block(parser);
            
        case TOKEN_IF:
            return parse_if_statement(parser);
            
        case TOKEN_FOR:
            return parse_for_statement(parser);
            
        case TOKEN_PARALLEL_FOR:
            return parse_parallel_for_statement(parser);
            
        case TOKEN_WHILE:
            return parse_while_statement(parser);
            
        case TOKEN_RETURN:
            return parse_return_statement(parser);
            
        case TOKEN_CRITICAL:
            return parse_critical_statement(parser);
            
        case TOKEN_BARRIER:
            return parse_barrier_statement(parser);
            
        case TOKEN_SEMICOLON:
            lexer_next_token(parser->lexer); // 空语句
            return NULL;
            
        // 类型关键字表示声明
        case TOKEN_INT:
        case TOKEN_FLOAT_KW:
        case TOKEN_DOUBLE:
        case TOKEN_CHAR_KW:
        case TOKEN_VOID:
        case TOKEN_STRUCT:
        case TOKEN_ATOMIC:
        case TOKEN_THREAD_LOCAL:
            return parse_declaration(parser);
            
        default: {
            // 表达式语句
            ASTNode *expr = parse_expression(parser);
            if (!expr) return NULL;
            
            if (!parser_expect(parser, TOKEN_SEMICOLON)) {
                ast_destroy(expr);
                return NULL;
            }
            
            ASTNode *stmt = (ASTNode *)safe_malloc(sizeof(ASTNode));
            stmt->type = AST_EXPRESSION_STMT;
            stmt->expr_stmt.expression = expr;
            
            return stmt;
        }
    }
}

// =============================================================================
// 声明解析
// =============================================================================

static ASTNode *parse_variable_declaration(Parser *parser) {
    Type *type = parse_type(parser);
    if (!type) return NULL;
    
    if (!parser_match(parser, TOKEN_IDENTIFIER)) {
        parser_error(parser, "Expected variable name");
        return NULL;
    }
    
    char *name = safe_strdup(parser_current_token(parser)->value);
    lexer_next_token(parser->lexer);
    
    // 处理数组声明
    while (parser_consume(parser, TOKEN_LBRACKET)) {
        ASTNode *size_expr = NULL;
        if (!parser_match(parser, TOKEN_RBRACKET)) {
            size_expr = parse_expression(parser);
            if (!size_expr) {
                free(name);
                return NULL;
            }
        }
        
        if (!parser_expect(parser, TOKEN_RBRACKET)) {
            free(name);
            ast_destroy(size_expr);
            return NULL;
        }
        
        // 这里简化处理，假设大小是编译时常量
        int size = 0;
        if (size_expr && size_expr->type == AST_NUMBER) {
            size = (int)size_expr->number.int_value;
        }
        
        type = type_create_array(type, size);
        ast_destroy(size_expr);
    }
    
    ASTNode *initializer = NULL;
    if (parser_consume(parser, TOKEN_ASSIGN)) {
        initializer = parse_expression(parser);
        if (!initializer) {
            free(name);
            return NULL;
        }
    }
    
    if (!parser_expect(parser, TOKEN_SEMICOLON)) {
        free(name);
        ast_destroy(initializer);
        return NULL;
    }
    
    return ast_create_var_decl(type, name, initializer);
}

static ASTNode *parse_function_declaration(Parser *parser) {
    Type *return_type = parse_type(parser);
    if (!return_type) return NULL;
    
    if (!parser_match(parser, TOKEN_IDENTIFIER)) {
        parser_error(parser, "Expected function name");
        return NULL;
    }
    
    char *name = safe_strdup(parser_current_token(parser)->value);
    lexer_next_token(parser->lexer);
    
    if (!parser_expect(parser, TOKEN_LPAREN)) {
        free(name);
        return NULL;
    }
    
    // 解析参数列表
    ASTNode **parameters = NULL;
    int param_count = 0;
    
    if (!parser_match(parser, TOKEN_RPAREN)) {
        parameters = (ASTNode **)safe_malloc(sizeof(ASTNode*) * MAX_PARAMS);
        
        do {
            if (param_count >= MAX_PARAMS) {
                parser_error(parser, "Too many function parameters");
                break;
            }
            
            Type *param_type = parse_type(parser);
            if (!param_type) break;
            
            char *param_name = NULL;
            if (parser_match(parser, TOKEN_IDENTIFIER)) {
                param_name = safe_strdup(parser_current_token(parser)->value);
                lexer_next_token(parser->lexer);
            }
            
            ASTNode *param = (ASTNode *)safe_malloc(sizeof(ASTNode));
            param->type = AST_PARAM_DECL;
            param->param_decl.type = param_type;
            param->param_decl.name = param_name;
            
            parameters[param_count++] = param;
            
        } while (parser_consume(parser, TOKEN_COMMA));
    }
    
    if (!parser_expect(parser, TOKEN_RPAREN)) {
        free(name);
        for (int i = 0; i < param_count; i++) {
            ast_destroy(parameters[i]);
        }
        free(parameters);
        return NULL;
    }
    
    ASTNode *body = NULL;
    if (parser_match(parser, TOKEN_LBRACE)) {
        body = parse_block(parser);
        if (!body) {
            free(name);
            for (int i = 0; i < param_count; i++) {
                ast_destroy(parameters[i]);
            }
            free(parameters);
            return NULL;
        }
    } else {
        // 函数声明，没有实现
        if (!parser_expect(parser, TOKEN_SEMICOLON)) {
            free(name);
            for (int i = 0; i < param_count; i++) {
                ast_destroy(parameters[i]);
            }
            free(parameters);
            return NULL;
        }
    }
    
    return ast_create_function_decl(return_type, name, parameters, param_count, body);
}

static ASTNode *parse_declaration(Parser *parser) {
    // 保存当前位置，用于区分变量声明和函数声明
    char *saved_pos = parser->lexer->current;
    int saved_line = parser->lexer->line;
    int saved_column = parser->lexer->column;
    
    Type *type = parse_type(parser);
    if (!type) return NULL;
    
    if (!parser_match(parser, TOKEN_IDENTIFIER)) {
        parser_error(parser, "Expected identifier in declaration");
        return NULL;
    }
    
    // 向前看，判断是函数还是变量
    lexer_next_token(parser->lexer); // 跳过标识符
    
    if (parser_match(parser, TOKEN_LPAREN)) {
        // 函数声明，重置位置重新解析
        parser->lexer->current = saved_pos;
        parser->lexer->line = saved_line;
        parser->lexer->column = saved_column;
        lexer_next_token(parser->lexer); // 重新读取当前标记
        return parse_function_declaration(parser);
    } else {
        // 变量声明，重置位置重新解析
        parser->lexer->current = saved_pos;
        parser->lexer->line = saved_line;
        parser->lexer->column = saved_column;
        lexer_next_token(parser->lexer); // 重新读取当前标记
        return parse_variable_declaration(parser);
    }
}

// =============================================================================
// 主解析函数
// =============================================================================

ASTNode *parser_parse(Parser *parser) {
    ASTNode **declarations = (ASTNode **)safe_malloc(sizeof(ASTNode*) * 1000);
    int decl_count = 0;
    
    while (!parser_match(parser, TOKEN_EOF)) {
        if (parser_match(parser, TOKEN_NEWLINE)) {
            lexer_next_token(parser->lexer);
            continue;
        }
        
        ASTNode *decl = parse_declaration(parser);
        if (decl) {
            declarations[decl_count++] = decl;
        } else if (parser->has_error) {
            // 清理已分配的声明
            for (int i = 0; i < decl_count; i++) {
                ast_destroy(declarations[i]);
            }
            free(declarations);
            return NULL;
        }
    }
    
    return ast_create_block(declarations, decl_count);
}
