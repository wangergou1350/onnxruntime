#include "riscv_cc.h"

// =============================================================================
// 语法分析器实现
// =============================================================================

typedef struct Parser {
    Token *tokens;
    int token_count;
    int current_position;
    bool has_errors;
    char *error_message;
} Parser;

// 前向声明
static ASTNode *parse_expression(Parser *parser);
static ASTNode *parse_statement(Parser *parser);
static ASTNode *parse_declaration(Parser *parser);
static ASTNode *parse_function_definition(Parser *parser);

// =============================================================================
// 解析器工具函数
// =============================================================================

static Parser *create_parser(Token *tokens, int token_count) {
    Parser *parser = malloc(sizeof(Parser));
    if (!parser) {
        error("内存分配失败：无法创建语法分析器");
        return NULL;
    }
    
    parser->tokens = tokens;
    parser->token_count = token_count;
    parser->current_position = 0;
    parser->has_errors = false;
    parser->error_message = NULL;
    
    return parser;
}

static void destroy_parser(Parser *parser) {
    if (parser) {
        free(parser->error_message);
        free(parser);
    }
}

static Token *current_token(Parser *parser) {
    if (parser->current_position >= parser->token_count) {
        return &parser->tokens[parser->token_count - 1]; // 返回EOF标记
    }
    return &parser->tokens[parser->current_position];
}

static Token *peek_token(Parser *parser, int offset) {
    int pos = parser->current_position + offset;
    if (pos >= parser->token_count) {
        return &parser->tokens[parser->token_count - 1]; // 返回EOF标记
    }
    return &parser->tokens[pos];
}

static void advance_token(Parser *parser) {
    if (parser->current_position < parser->token_count - 1) {
        parser->current_position++;
    }
}

static bool match_token(Parser *parser, TokenType type) {
    Token *token = current_token(parser);
    return token->type == type;
}

static bool consume_token(Parser *parser, TokenType type) {
    if (match_token(parser, type)) {
        advance_token(parser);
        return true;
    }
    return false;
}

static bool expect_token(Parser *parser, TokenType type) {
    if (match_token(parser, type)) {
        advance_token(parser);
        return true;
    }
    
    Token *token = current_token(parser);
    error("语法错误：期望 %s，但得到 %s 在第 %d 行第 %d 列",
          token_type_to_string(type),
          token_type_to_string(token->type),
          token->line,
          token->column);
    parser->has_errors = true;
    return false;
}

static void parser_error(Parser *parser, const char *format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    
    Token *token = current_token(parser);
    error("语法错误：%s 在第 %d 行第 %d 列", buffer, token->line, token->column);
    
    parser->has_errors = true;
    va_end(args);
}

// =============================================================================
// AST节点创建函数
// =============================================================================

ASTNode *create_ast_node(ASTNodeType type) {
    ASTNode *node = malloc(sizeof(ASTNode));
    if (!node) {
        error("内存分配失败：无法创建AST节点");
        return NULL;
    }
    
    memset(node, 0, sizeof(ASTNode));
    node->type = type;
    
    return node;
}

void destroy_ast(ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->program.declaration_count; i++) {
                destroy_ast(node->program.declarations[i]);
            }
            free(node->program.declarations);
            break;
            
        case AST_FUNCTION_DEF:
            free(node->function_def.name);
            for (int i = 0; i < node->function_def.parameter_count; i++) {
                destroy_ast(node->function_def.parameters[i]);
            }
            free(node->function_def.parameters);
            destroy_ast(node->function_def.body);
            break;
            
        case AST_VARIABLE_DECL:
            free(node->var_decl.name);
            destroy_ast(node->var_decl.init_value);
            break;
            
        case AST_COMPOUND_STMT:
            for (int i = 0; i < node->compound_stmt.statement_count; i++) {
                destroy_ast(node->compound_stmt.statements[i]);
            }
            free(node->compound_stmt.statements);
            break;
            
        case AST_BINARY_EXPR:
            destroy_ast(node->binary_expr.left);
            destroy_ast(node->binary_expr.right);
            break;
            
        case AST_UNARY_EXPR:
            destroy_ast(node->unary_expr.operand);
            break;
            
        case AST_CALL_EXPR:
            destroy_ast(node->call_expr.function);
            for (int i = 0; i < node->call_expr.argument_count; i++) {
                destroy_ast(node->call_expr.arguments[i]);
            }
            free(node->call_expr.arguments);
            break;
            
        case AST_IDENTIFIER:
            free(node->identifier.name);
            break;
            
        case AST_STRING_LITERAL:
            free(node->literal.string_value);
            break;
            
        default:
            // 其他节点类型的清理
            break;
    }
    
    free(node);
}

// =============================================================================
// 类型解析
// =============================================================================

static DataType parse_type_specifier(Parser *parser) {
    Token *token = current_token(parser);
    
    switch (token->type) {
        case TOKEN_VOID:
            advance_token(parser);
            return TYPE_VOID;
        case TOKEN_CHAR:
            advance_token(parser);
            return TYPE_CHAR;
        case TOKEN_SHORT:
            advance_token(parser);
            return TYPE_SHORT;
        case TOKEN_INT:
            advance_token(parser);
            return TYPE_INT;
        case TOKEN_LONG:
            advance_token(parser);
            return TYPE_LONG;
        case TOKEN_FLOAT_KW:
            advance_token(parser);
            return TYPE_FLOAT;
        case TOKEN_DOUBLE:
            advance_token(parser);
            return TYPE_DOUBLE;
        default:
            parser_error(parser, "期望类型说明符");
            return TYPE_UNKNOWN;
    }
}

// =============================================================================
// 表达式解析
// =============================================================================

static ASTNode *parse_primary_expression(Parser *parser) {
    Token *token = current_token(parser);
    ASTNode *node = NULL;
    
    switch (token->type) {
        case TOKEN_NUMBER:
            node = create_ast_node(AST_NUMBER_LITERAL);
            node->literal.int_value = token->int_value;
            node->literal.literal_type = TYPE_INT;
            advance_token(parser);
            break;
            
        case TOKEN_FLOAT:
            node = create_ast_node(AST_FLOAT_LITERAL);
            node->literal.float_value = token->float_value;
            node->literal.literal_type = TYPE_FLOAT;
            advance_token(parser);
            break;
            
        case TOKEN_CHARACTER:
            node = create_ast_node(AST_CHAR_LITERAL);
            node->literal.char_value = token->char_value;
            node->literal.literal_type = TYPE_CHAR;
            advance_token(parser);
            break;
            
        case TOKEN_STRING:
            node = create_ast_node(AST_STRING_LITERAL);
            node->literal.string_value = strdup(token->value);
            node->literal.literal_type = TYPE_POINTER;
            advance_token(parser);
            break;
            
        case TOKEN_IDENTIFIER:
            node = create_ast_node(AST_IDENTIFIER);
            node->identifier.name = strdup(token->value);
            advance_token(parser);
            break;
            
        case TOKEN_LPAREN:
            advance_token(parser);
            node = parse_expression(parser);
            expect_token(parser, TOKEN_RPAREN);
            break;
            
        default:
            parser_error(parser, "期望主表达式");
            break;
    }
    
    return node;
}

static ASTNode *parse_postfix_expression(Parser *parser) {
    ASTNode *node = parse_primary_expression(parser);
    
    while (true) {
        Token *token = current_token(parser);
        
        if (token->type == TOKEN_LBRACKET) {
            // 数组下标
            advance_token(parser);
            ASTNode *index_expr = create_ast_node(AST_INDEX_EXPR);
            index_expr->index_expr.array = node;
            index_expr->index_expr.index = parse_expression(parser);
            expect_token(parser, TOKEN_RBRACKET);
            node = index_expr;
        }
        else if (token->type == TOKEN_LPAREN) {
            // 函数调用
            advance_token(parser);
            ASTNode *call_expr = create_ast_node(AST_CALL_EXPR);
            call_expr->call_expr.function = node;
            call_expr->call_expr.arguments = malloc(16 * sizeof(ASTNode*));
            call_expr->call_expr.argument_count = 0;
            
            if (!match_token(parser, TOKEN_RPAREN)) {
                do {
                    call_expr->call_expr.arguments[call_expr->call_expr.argument_count++] = 
                        parse_expression(parser);
                } while (consume_token(parser, TOKEN_COMMA));
            }
            
            expect_token(parser, TOKEN_RPAREN);
            node = call_expr;
        }
        else if (token->type == TOKEN_DOT) {
            // 结构体成员访问
            advance_token(parser);
            ASTNode *member_expr = create_ast_node(AST_MEMBER_EXPR);
            member_expr->member_expr.object = node;
            member_expr->member_expr.is_pointer_access = false;
            
            Token *member_token = current_token(parser);
            if (member_token->type == TOKEN_IDENTIFIER) {
                member_expr->member_expr.member = strdup(member_token->value);
                advance_token(parser);
            } else {
                parser_error(parser, "期望成员名称");
            }
            node = member_expr;
        }
        else if (token->type == TOKEN_ARROW) {
            // 指针成员访问
            advance_token(parser);
            ASTNode *member_expr = create_ast_node(AST_MEMBER_EXPR);
            member_expr->member_expr.object = node;
            member_expr->member_expr.is_pointer_access = true;
            
            Token *member_token = current_token(parser);
            if (member_token->type == TOKEN_IDENTIFIER) {
                member_expr->member_expr.member = strdup(member_token->value);
                advance_token(parser);
            } else {
                parser_error(parser, "期望成员名称");
            }
            node = member_expr;
        }
        else if (token->type == TOKEN_INCREMENT || token->type == TOKEN_DECREMENT) {
            // 后缀递增/递减
            ASTNode *unary_expr = create_ast_node(AST_UNARY_EXPR);
            unary_expr->unary_expr.operator = token->type;
            unary_expr->unary_expr.operand = node;
            advance_token(parser);
            node = unary_expr;
        }
        else {
            break;
        }
    }
    
    return node;
}

static ASTNode *parse_unary_expression(Parser *parser) {
    Token *token = current_token(parser);
    
    if (token->type == TOKEN_INCREMENT || token->type == TOKEN_DECREMENT ||
        token->type == TOKEN_PLUS || token->type == TOKEN_MINUS ||
        token->type == TOKEN_LOGICAL_NOT || token->type == TOKEN_BITWISE_NOT ||
        token->type == TOKEN_MULTIPLY || token->type == TOKEN_BITWISE_AND) {
        
        ASTNode *unary_expr = create_ast_node(AST_UNARY_EXPR);
        unary_expr->unary_expr.operator = token->type;
        advance_token(parser);
        unary_expr->unary_expr.operand = parse_unary_expression(parser);
        return unary_expr;
    }
    else if (token->type == TOKEN_SIZEOF) {
        advance_token(parser);
        ASTNode *sizeof_expr = create_ast_node(AST_SIZEOF_EXPR);
        
        if (match_token(parser, TOKEN_LPAREN)) {
            advance_token(parser);
            // 可能是类型或表达式
            sizeof_expr->sizeof_expr.target_type = parse_type_specifier(parser);
            expect_token(parser, TOKEN_RPAREN);
        } else {
            sizeof_expr->sizeof_expr.expression = parse_unary_expression(parser);
        }
        return sizeof_expr;
    }
    else {
        return parse_postfix_expression(parser);
    }
}

static ASTNode *parse_cast_expression(Parser *parser) {
    // 简化版：先不处理类型转换
    return parse_unary_expression(parser);
}

static ASTNode *parse_multiplicative_expression(Parser *parser) {
    ASTNode *left = parse_cast_expression(parser);
    
    while (true) {
        Token *token = current_token(parser);
        if (token->type == TOKEN_MULTIPLY || token->type == TOKEN_DIVIDE || 
            token->type == TOKEN_MODULO) {
            
            ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
            binary_expr->binary_expr.operator = token->type;
            binary_expr->binary_expr.left = left;
            advance_token(parser);
            binary_expr->binary_expr.right = parse_cast_expression(parser);
            left = binary_expr;
        } else {
            break;
        }
    }
    
    return left;
}

static ASTNode *parse_additive_expression(Parser *parser) {
    ASTNode *left = parse_multiplicative_expression(parser);
    
    while (true) {
        Token *token = current_token(parser);
        if (token->type == TOKEN_PLUS || token->type == TOKEN_MINUS) {
            ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
            binary_expr->binary_expr.operator = token->type;
            binary_expr->binary_expr.left = left;
            advance_token(parser);
            binary_expr->binary_expr.right = parse_multiplicative_expression(parser);
            left = binary_expr;
        } else {
            break;
        }
    }
    
    return left;
}

static ASTNode *parse_shift_expression(Parser *parser) {
    ASTNode *left = parse_additive_expression(parser);
    
    while (true) {
        Token *token = current_token(parser);
        if (token->type == TOKEN_LSHIFT || token->type == TOKEN_RSHIFT) {
            ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
            binary_expr->binary_expr.operator = token->type;
            binary_expr->binary_expr.left = left;
            advance_token(parser);
            binary_expr->binary_expr.right = parse_additive_expression(parser);
            left = binary_expr;
        } else {
            break;
        }
    }
    
    return left;
}

static ASTNode *parse_relational_expression(Parser *parser) {
    ASTNode *left = parse_shift_expression(parser);
    
    while (true) {
        Token *token = current_token(parser);
        if (token->type == TOKEN_LESS || token->type == TOKEN_GREATER ||
            token->type == TOKEN_LESS_EQUAL || token->type == TOKEN_GREATER_EQUAL) {
            
            ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
            binary_expr->binary_expr.operator = token->type;
            binary_expr->binary_expr.left = left;
            advance_token(parser);
            binary_expr->binary_expr.right = parse_shift_expression(parser);
            left = binary_expr;
        } else {
            break;
        }
    }
    
    return left;
}

static ASTNode *parse_equality_expression(Parser *parser) {
    ASTNode *left = parse_relational_expression(parser);
    
    while (true) {
        Token *token = current_token(parser);
        if (token->type == TOKEN_EQUAL || token->type == TOKEN_NOT_EQUAL) {
            ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
            binary_expr->binary_expr.operator = token->type;
            binary_expr->binary_expr.left = left;
            advance_token(parser);
            binary_expr->binary_expr.right = parse_relational_expression(parser);
            left = binary_expr;
        } else {
            break;
        }
    }
    
    return left;
}

static ASTNode *parse_and_expression(Parser *parser) {
    ASTNode *left = parse_equality_expression(parser);
    
    while (match_token(parser, TOKEN_BITWISE_AND)) {
        ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
        binary_expr->binary_expr.operator = TOKEN_BITWISE_AND;
        binary_expr->binary_expr.left = left;
        advance_token(parser);
        binary_expr->binary_expr.right = parse_equality_expression(parser);
        left = binary_expr;
    }
    
    return left;
}

static ASTNode *parse_xor_expression(Parser *parser) {
    ASTNode *left = parse_and_expression(parser);
    
    while (match_token(parser, TOKEN_BITWISE_XOR)) {
        ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
        binary_expr->binary_expr.operator = TOKEN_BITWISE_XOR;
        binary_expr->binary_expr.left = left;
        advance_token(parser);
        binary_expr->binary_expr.right = parse_and_expression(parser);
        left = binary_expr;
    }
    
    return left;
}

static ASTNode *parse_or_expression(Parser *parser) {
    ASTNode *left = parse_xor_expression(parser);
    
    while (match_token(parser, TOKEN_BITWISE_OR)) {
        ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
        binary_expr->binary_expr.operator = TOKEN_BITWISE_OR;
        binary_expr->binary_expr.left = left;
        advance_token(parser);
        binary_expr->binary_expr.right = parse_xor_expression(parser);
        left = binary_expr;
    }
    
    return left;
}

static ASTNode *parse_logical_and_expression(Parser *parser) {
    ASTNode *left = parse_or_expression(parser);
    
    while (match_token(parser, TOKEN_LOGICAL_AND)) {
        ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
        binary_expr->binary_expr.operator = TOKEN_LOGICAL_AND;
        binary_expr->binary_expr.left = left;
        advance_token(parser);
        binary_expr->binary_expr.right = parse_or_expression(parser);
        left = binary_expr;
    }
    
    return left;
}

static ASTNode *parse_logical_or_expression(Parser *parser) {
    ASTNode *left = parse_logical_and_expression(parser);
    
    while (match_token(parser, TOKEN_LOGICAL_OR)) {
        ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
        binary_expr->binary_expr.operator = TOKEN_LOGICAL_OR;
        binary_expr->binary_expr.left = left;
        advance_token(parser);
        binary_expr->binary_expr.right = parse_logical_and_expression(parser);
        left = binary_expr;
    }
    
    return left;
}

static ASTNode *parse_conditional_expression(Parser *parser) {
    ASTNode *condition = parse_logical_or_expression(parser);
    
    if (match_token(parser, TOKEN_QUESTION)) {
        advance_token(parser);
        ASTNode *ternary_expr = create_ast_node(AST_TERNARY_EXPR);
        ternary_expr->ternary_expr.condition = condition;
        ternary_expr->ternary_expr.true_expr = parse_expression(parser);
        expect_token(parser, TOKEN_COLON);
        ternary_expr->ternary_expr.false_expr = parse_conditional_expression(parser);
        return ternary_expr;
    }
    
    return condition;
}

static ASTNode *parse_assignment_expression(Parser *parser) {
    ASTNode *left = parse_conditional_expression(parser);
    
    Token *token = current_token(parser);
    if (token->type == TOKEN_ASSIGN || token->type == TOKEN_PLUS_ASSIGN ||
        token->type == TOKEN_MINUS_ASSIGN || token->type == TOKEN_MUL_ASSIGN ||
        token->type == TOKEN_DIV_ASSIGN || token->type == TOKEN_MOD_ASSIGN ||
        token->type == TOKEN_AND_ASSIGN || token->type == TOKEN_OR_ASSIGN ||
        token->type == TOKEN_XOR_ASSIGN || token->type == TOKEN_LSHIFT_ASSIGN ||
        token->type == TOKEN_RSHIFT_ASSIGN) {
        
        ASTNode *assign_expr = create_ast_node(AST_ASSIGN_EXPR);
        assign_expr->assign_expr.operator = token->type;
        assign_expr->assign_expr.left = left;
        advance_token(parser);
        assign_expr->assign_expr.right = parse_assignment_expression(parser);
        return assign_expr;
    }
    
    return left;
}

static ASTNode *parse_expression(Parser *parser) {
    ASTNode *left = parse_assignment_expression(parser);
    
    while (match_token(parser, TOKEN_COMMA)) {
        advance_token(parser);
        ASTNode *binary_expr = create_ast_node(AST_BINARY_EXPR);
        binary_expr->binary_expr.operator = TOKEN_COMMA;
        binary_expr->binary_expr.left = left;
        binary_expr->binary_expr.right = parse_assignment_expression(parser);
        left = binary_expr;
    }
    
    return left;
}

// =============================================================================
// 语句解析
// =============================================================================

static ASTNode *parse_compound_statement(Parser *parser) {
    expect_token(parser, TOKEN_LBRACE);
    
    ASTNode *compound = create_ast_node(AST_COMPOUND_STMT);
    compound->compound_stmt.statements = malloc(64 * sizeof(ASTNode*));
    compound->compound_stmt.statement_count = 0;
    compound->compound_stmt.statement_capacity = 64;
    
    while (!match_token(parser, TOKEN_RBRACE) && !match_token(parser, TOKEN_EOF)) {
        ASTNode *stmt = NULL;
        
        // 尝试解析声明或语句
        if (match_token(parser, TOKEN_INT) || match_token(parser, TOKEN_CHAR) ||
            match_token(parser, TOKEN_FLOAT_KW) || match_token(parser, TOKEN_DOUBLE) ||
            match_token(parser, TOKEN_VOID) || match_token(parser, TOKEN_LONG) ||
            match_token(parser, TOKEN_SHORT) || match_token(parser, TOKEN_SIGNED) ||
            match_token(parser, TOKEN_UNSIGNED) || match_token(parser, TOKEN_CONST) ||
            match_token(parser, TOKEN_VOLATILE) || match_token(parser, TOKEN_STATIC) ||
            match_token(parser, TOKEN_EXTERN) || match_token(parser, TOKEN_REGISTER) ||
            match_token(parser, TOKEN_AUTO) || match_token(parser, TOKEN_ATOMIC)) {
            stmt = parse_declaration(parser);
        } else {
            stmt = parse_statement(parser);
        }
        
        if (stmt) {
            // 动态扩展数组
            if (compound->compound_stmt.statement_count >= compound->compound_stmt.statement_capacity) {
                compound->compound_stmt.statement_capacity *= 2;
                compound->compound_stmt.statements = realloc(
                    compound->compound_stmt.statements,
                    compound->compound_stmt.statement_capacity * sizeof(ASTNode*)
                );
            }
            compound->compound_stmt.statements[compound->compound_stmt.statement_count++] = stmt;
        }
    }
    
    expect_token(parser, TOKEN_RBRACE);
    return compound;
}

static ASTNode *parse_if_statement(Parser *parser) {
    expect_token(parser, TOKEN_IF);
    expect_token(parser, TOKEN_LPAREN);
    
    ASTNode *if_stmt = create_ast_node(AST_IF_STMT);
    if_stmt->if_stmt.condition = parse_expression(parser);
    
    expect_token(parser, TOKEN_RPAREN);
    if_stmt->if_stmt.then_stmt = parse_statement(parser);
    
    if (match_token(parser, TOKEN_ELSE)) {
        advance_token(parser);
        if_stmt->if_stmt.else_stmt = parse_statement(parser);
    }
    
    return if_stmt;
}

static ASTNode *parse_while_statement(Parser *parser) {
    expect_token(parser, TOKEN_WHILE);
    expect_token(parser, TOKEN_LPAREN);
    
    ASTNode *while_stmt = create_ast_node(AST_WHILE_STMT);
    while_stmt->while_stmt.condition = parse_expression(parser);
    
    expect_token(parser, TOKEN_RPAREN);
    while_stmt->while_stmt.body = parse_statement(parser);
    
    return while_stmt;
}

static ASTNode *parse_for_statement(Parser *parser) {
    expect_token(parser, TOKEN_FOR);
    expect_token(parser, TOKEN_LPAREN);
    
    ASTNode *for_stmt = create_ast_node(AST_FOR_STMT);
    
    // 初始化表达式
    if (!match_token(parser, TOKEN_SEMICOLON)) {
        for_stmt->for_stmt.init = parse_expression(parser);
    }
    expect_token(parser, TOKEN_SEMICOLON);
    
    // 条件表达式
    if (!match_token(parser, TOKEN_SEMICOLON)) {
        for_stmt->for_stmt.condition = parse_expression(parser);
    }
    expect_token(parser, TOKEN_SEMICOLON);
    
    // 递增表达式
    if (!match_token(parser, TOKEN_RPAREN)) {
        for_stmt->for_stmt.increment = parse_expression(parser);
    }
    expect_token(parser, TOKEN_RPAREN);
    
    for_stmt->for_stmt.body = parse_statement(parser);
    
    return for_stmt;
}

static ASTNode *parse_parallel_for_statement(Parser *parser) {
    expect_token(parser, TOKEN_PARALLEL_FOR);
    expect_token(parser, TOKEN_LPAREN);
    
    ASTNode *parallel_for = create_ast_node(AST_PARALLEL_FOR_STMT);
    
    // 初始化表达式
    if (!match_token(parser, TOKEN_SEMICOLON)) {
        parallel_for->parallel_for_stmt.init = parse_expression(parser);
    }
    expect_token(parser, TOKEN_SEMICOLON);
    
    // 条件表达式
    if (!match_token(parser, TOKEN_SEMICOLON)) {
        parallel_for->parallel_for_stmt.condition = parse_expression(parser);
    }
    expect_token(parser, TOKEN_SEMICOLON);
    
    // 递增表达式
    if (!match_token(parser, TOKEN_RPAREN)) {
        parallel_for->parallel_for_stmt.increment = parse_expression(parser);
    }
    expect_token(parser, TOKEN_RPAREN);
    
    parallel_for->parallel_for_stmt.body = parse_statement(parser);
    parallel_for->parallel_for_stmt.num_threads = 0; // 自动检测
    
    return parallel_for;
}

static ASTNode *parse_return_statement(Parser *parser) {
    expect_token(parser, TOKEN_RETURN);
    
    ASTNode *return_stmt = create_ast_node(AST_RETURN_STMT);
    
    if (!match_token(parser, TOKEN_SEMICOLON)) {
        return_stmt->return_stmt.value = parse_expression(parser);
    }
    
    expect_token(parser, TOKEN_SEMICOLON);
    return return_stmt;
}

static ASTNode *parse_break_statement(Parser *parser) {
    expect_token(parser, TOKEN_BREAK);
    expect_token(parser, TOKEN_SEMICOLON);
    
    return create_ast_node(AST_BREAK_STMT);
}

static ASTNode *parse_continue_statement(Parser *parser) {
    expect_token(parser, TOKEN_CONTINUE);
    expect_token(parser, TOKEN_SEMICOLON);
    
    return create_ast_node(AST_CONTINUE_STMT);
}

static ASTNode *parse_expression_statement(Parser *parser) {
    ASTNode *expr_stmt = create_ast_node(AST_EXPRESSION_STMT);
    
    if (!match_token(parser, TOKEN_SEMICOLON)) {
        expr_stmt = parse_expression(parser);
    }
    
    expect_token(parser, TOKEN_SEMICOLON);
    return expr_stmt;
}

static ASTNode *parse_statement(Parser *parser) {
    Token *token = current_token(parser);
    
    switch (token->type) {
        case TOKEN_LBRACE:
            return parse_compound_statement(parser);
        case TOKEN_IF:
            return parse_if_statement(parser);
        case TOKEN_WHILE:
            return parse_while_statement(parser);
        case TOKEN_FOR:
            return parse_for_statement(parser);
        case TOKEN_PARALLEL_FOR:
            return parse_parallel_for_statement(parser);
        case TOKEN_RETURN:
            return parse_return_statement(parser);
        case TOKEN_BREAK:
            return parse_break_statement(parser);
        case TOKEN_CONTINUE:
            return parse_continue_statement(parser);
        case TOKEN_BARRIER:
            advance_token(parser);
            expect_token(parser, TOKEN_SEMICOLON);
            return create_ast_node(AST_BARRIER_STMT);
        default:
            return parse_expression_statement(parser);
    }
}

// =============================================================================
// 声明解析
// =============================================================================

static ASTNode *parse_variable_declaration(Parser *parser) {
    ASTNode *var_decl = create_ast_node(AST_VARIABLE_DECL);
    
    // 解析存储类说明符
    var_decl->var_decl.storage_class = STORAGE_AUTO;
    if (match_token(parser, TOKEN_STATIC)) {
        var_decl->var_decl.storage_class = STORAGE_STATIC;
        advance_token(parser);
    } else if (match_token(parser, TOKEN_EXTERN)) {
        var_decl->var_decl.storage_class = STORAGE_EXTERN;
        advance_token(parser);
    } else if (match_token(parser, TOKEN_REGISTER)) {
        var_decl->var_decl.storage_class = STORAGE_REGISTER;
        advance_token(parser);
    }
    
    // 解析类型限定符
    if (match_token(parser, TOKEN_CONST)) {
        var_decl->var_decl.is_const = true;
        advance_token(parser);
    }
    if (match_token(parser, TOKEN_VOLATILE)) {
        var_decl->var_decl.is_volatile = true;
        advance_token(parser);
    }
    if (match_token(parser, TOKEN_ATOMIC)) {
        var_decl->var_decl.is_atomic = true;
        advance_token(parser);
    }
    
    // 解析基本类型
    var_decl->var_decl.data_type = parse_type_specifier(parser);
    
    // 解析声明符
    var_decl->var_decl.pointer_level = 0;
    while (match_token(parser, TOKEN_MULTIPLY)) {
        var_decl->var_decl.pointer_level++;
        advance_token(parser);
    }
    
    Token *name_token = current_token(parser);
    if (name_token->type == TOKEN_IDENTIFIER) {
        var_decl->var_decl.name = strdup(name_token->value);
        advance_token(parser);
    } else {
        parser_error(parser, "期望变量名");
        return var_decl;
    }
    
    // 数组声明
    var_decl->var_decl.array_size = -1;
    if (match_token(parser, TOKEN_LBRACKET)) {
        advance_token(parser);
        if (!match_token(parser, TOKEN_RBRACKET)) {
            ASTNode *size_expr = parse_expression(parser);
            // 简化：假设是常量表达式
            if (size_expr->type == AST_NUMBER_LITERAL) {
                var_decl->var_decl.array_size = size_expr->literal.int_value;
            }
            destroy_ast(size_expr);
        }
        expect_token(parser, TOKEN_RBRACKET);
    }
    
    // 初始化
    if (match_token(parser, TOKEN_ASSIGN)) {
        advance_token(parser);
        var_decl->var_decl.init_value = parse_assignment_expression(parser);
    }
    
    return var_decl;
}

static ASTNode *parse_function_definition(Parser *parser) {
    ASTNode *func_def = create_ast_node(AST_FUNCTION_DEF);
    
    // 解析返回类型
    func_def->function_def.return_type = parse_type_specifier(parser);
    
    // 解析函数名
    Token *name_token = current_token(parser);
    if (name_token->type == TOKEN_IDENTIFIER) {
        func_def->function_def.name = strdup(name_token->value);
        advance_token(parser);
    } else {
        parser_error(parser, "期望函数名");
        return func_def;
    }
    
    // 解析参数列表
    expect_token(parser, TOKEN_LPAREN);
    
    func_def->function_def.parameters = malloc(16 * sizeof(ASTNode*));
    func_def->function_def.parameter_count = 0;
    
    if (!match_token(parser, TOKEN_RPAREN)) {
        do {
            ASTNode *param = parse_variable_declaration(parser);
            func_def->function_def.parameters[func_def->function_def.parameter_count++] = param;
        } while (consume_token(parser, TOKEN_COMMA));
    }
    
    expect_token(parser, TOKEN_RPAREN);
    
    // 解析函数体
    if (match_token(parser, TOKEN_LBRACE)) {
        func_def->function_def.body = parse_compound_statement(parser);
    } else {
        // 函数声明（没有函数体）
        expect_token(parser, TOKEN_SEMICOLON);
    }
    
    return func_def;
}

static ASTNode *parse_declaration(Parser *parser) {
    // 简化版：判断是函数定义还是变量声明
    // 向前查看以确定类型
    int saved_pos = parser->current_position;
    
    // 跳过存储类和类型限定符
    while (match_token(parser, TOKEN_STATIC) || match_token(parser, TOKEN_EXTERN) ||
           match_token(parser, TOKEN_REGISTER) || match_token(parser, TOKEN_CONST) ||
           match_token(parser, TOKEN_VOLATILE) || match_token(parser, TOKEN_ATOMIC)) {
        advance_token(parser);
    }
    
    // 跳过类型说明符
    if (match_token(parser, TOKEN_VOID) || match_token(parser, TOKEN_CHAR) ||
        match_token(parser, TOKEN_SHORT) || match_token(parser, TOKEN_INT) ||
        match_token(parser, TOKEN_LONG) || match_token(parser, TOKEN_FLOAT_KW) ||
        match_token(parser, TOKEN_DOUBLE) || match_token(parser, TOKEN_SIGNED) ||
        match_token(parser, TOKEN_UNSIGNED)) {
        advance_token(parser);
    }
    
    // 跳过指针
    while (match_token(parser, TOKEN_MULTIPLY)) {
        advance_token(parser);
    }
    
    // 跳过标识符
    if (match_token(parser, TOKEN_IDENTIFIER)) {
        advance_token(parser);
    }
    
    bool is_function = match_token(parser, TOKEN_LPAREN);
    
    // 恢复位置
    parser->current_position = saved_pos;
    
    if (is_function) {
        return parse_function_definition(parser);
    } else {
        ASTNode *var_decl = parse_variable_declaration(parser);
        expect_token(parser, TOKEN_SEMICOLON);
        return var_decl;
    }
}

// =============================================================================
// 程序解析
// =============================================================================

ASTNode *parse(Token *tokens, int token_count) {
    Parser *parser = create_parser(tokens, token_count);
    if (!parser) {
        return NULL;
    }
    
    ASTNode *program = create_ast_node(AST_PROGRAM);
    program->program.declarations = malloc(64 * sizeof(ASTNode*));
    program->program.declaration_count = 0;
    program->program.declaration_capacity = 64;
    
    while (!match_token(parser, TOKEN_EOF)) {
        ASTNode *decl = parse_declaration(parser);
        if (decl) {
            // 动态扩展数组
            if (program->program.declaration_count >= program->program.declaration_capacity) {
                program->program.declaration_capacity *= 2;
                program->program.declarations = realloc(
                    program->program.declarations,
                    program->program.declaration_capacity * sizeof(ASTNode*)
                );
            }
            program->program.declarations[program->program.declaration_count++] = decl;
        }
        
        if (parser->has_errors) {
            break;
        }
    }
    
    bool success = !parser->has_errors;
    destroy_parser(parser);
    
    if (!success) {
        destroy_ast(program);
        return NULL;
    }
    
    return program;
}

// =============================================================================
// AST打印函数
// =============================================================================

void print_ast(ASTNode *node, int indent) {
    if (!node) return;
    
    for (int i = 0; i < indent; i++) printf("  ");
    
    switch (node->type) {
        case AST_PROGRAM:
            printf("Program (%d declarations)\n", node->program.declaration_count);
            for (int i = 0; i < node->program.declaration_count; i++) {
                print_ast(node->program.declarations[i], indent + 1);
            }
            break;
            
        case AST_FUNCTION_DEF:
            printf("FunctionDef: %s -> %s (%d params)\n",
                   node->function_def.name,
                   data_type_to_string(node->function_def.return_type),
                   node->function_def.parameter_count);
            for (int i = 0; i < node->function_def.parameter_count; i++) {
                print_ast(node->function_def.parameters[i], indent + 1);
            }
            if (node->function_def.body) {
                print_ast(node->function_def.body, indent + 1);
            }
            break;
            
        case AST_VARIABLE_DECL:
            printf("VarDecl: %s : %s",
                   node->var_decl.name,
                   data_type_to_string(node->var_decl.data_type));
            if (node->var_decl.pointer_level > 0) {
                printf(" *%d", node->var_decl.pointer_level);
            }
            if (node->var_decl.array_size >= 0) {
                printf(" [%d]", node->var_decl.array_size);
            }
            printf("\n");
            if (node->var_decl.init_value) {
                print_ast(node->var_decl.init_value, indent + 1);
            }
            break;
            
        case AST_COMPOUND_STMT:
            printf("CompoundStmt (%d statements)\n", node->compound_stmt.statement_count);
            for (int i = 0; i < node->compound_stmt.statement_count; i++) {
                print_ast(node->compound_stmt.statements[i], indent + 1);
            }
            break;
            
        case AST_IF_STMT:
            printf("IfStmt\n");
            for (int i = 0; i < indent + 1; i++) printf("  ");
            printf("condition:\n");
            print_ast(node->if_stmt.condition, indent + 2);
            for (int i = 0; i < indent + 1; i++) printf("  ");
            printf("then:\n");
            print_ast(node->if_stmt.then_stmt, indent + 2);
            if (node->if_stmt.else_stmt) {
                for (int i = 0; i < indent + 1; i++) printf("  ");
                printf("else:\n");
                print_ast(node->if_stmt.else_stmt, indent + 2);
            }
            break;
            
        case AST_WHILE_STMT:
            printf("WhileStmt\n");
            print_ast(node->while_stmt.condition, indent + 1);
            print_ast(node->while_stmt.body, indent + 1);
            break;
            
        case AST_FOR_STMT:
            printf("ForStmt\n");
            if (node->for_stmt.init) print_ast(node->for_stmt.init, indent + 1);
            if (node->for_stmt.condition) print_ast(node->for_stmt.condition, indent + 1);
            if (node->for_stmt.increment) print_ast(node->for_stmt.increment, indent + 1);
            print_ast(node->for_stmt.body, indent + 1);
            break;
            
        case AST_PARALLEL_FOR_STMT:
            printf("ParallelForStmt\n");
            if (node->parallel_for_stmt.init) print_ast(node->parallel_for_stmt.init, indent + 1);
            if (node->parallel_for_stmt.condition) print_ast(node->parallel_for_stmt.condition, indent + 1);
            if (node->parallel_for_stmt.increment) print_ast(node->parallel_for_stmt.increment, indent + 1);
            print_ast(node->parallel_for_stmt.body, indent + 1);
            break;
            
        case AST_RETURN_STMT:
            printf("ReturnStmt\n");
            if (node->return_stmt.value) {
                print_ast(node->return_stmt.value, indent + 1);
            }
            break;
            
        case AST_BINARY_EXPR:
            printf("BinaryExpr: %s\n", token_type_to_string(node->binary_expr.operator));
            print_ast(node->binary_expr.left, indent + 1);
            print_ast(node->binary_expr.right, indent + 1);
            break;
            
        case AST_UNARY_EXPR:
            printf("UnaryExpr: %s\n", token_type_to_string(node->unary_expr.operator));
            print_ast(node->unary_expr.operand, indent + 1);
            break;
            
        case AST_ASSIGN_EXPR:
            printf("AssignExpr: %s\n", token_type_to_string(node->assign_expr.operator));
            print_ast(node->assign_expr.left, indent + 1);
            print_ast(node->assign_expr.right, indent + 1);
            break;
            
        case AST_CALL_EXPR:
            printf("CallExpr (%d args)\n", node->call_expr.argument_count);
            print_ast(node->call_expr.function, indent + 1);
            for (int i = 0; i < node->call_expr.argument_count; i++) {
                print_ast(node->call_expr.arguments[i], indent + 1);
            }
            break;
            
        case AST_IDENTIFIER:
            printf("Identifier: %s\n", node->identifier.name);
            break;
            
        case AST_NUMBER_LITERAL:
            printf("NumberLiteral: %d\n", node->literal.int_value);
            break;
            
        case AST_FLOAT_LITERAL:
            printf("FloatLiteral: %f\n", node->literal.float_value);
            break;
            
        case AST_CHAR_LITERAL:
            printf("CharLiteral: '%c'\n", node->literal.char_value);
            break;
            
        case AST_STRING_LITERAL:
            printf("StringLiteral: %s\n", node->literal.string_value);
            break;
            
        default:
            printf("UnknownNode: %d\n", node->type);
            break;
    }
}
