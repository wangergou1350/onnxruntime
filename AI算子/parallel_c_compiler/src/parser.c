/*
 * ParallelC Compiler - Syntax Analyzer (Parser)
 * 
 * This module implements the syntax analysis phase using recursive descent parsing.
 * It builds an Abstract Syntax Tree (AST) from the token stream.
 */

#include "pcc.h"

typedef struct {
    Token *tokens;
    int current;
    int count;
} Parser;

static ASTNode *parse_expression(Parser *parser);
static ASTNode *parse_statement(Parser *parser);
static ASTNode *parse_block(Parser *parser);

static Token *current_token(Parser *parser) {
    if (parser->current >= parser->count) {
        return &parser->tokens[parser->count - 1]; // EOF token
    }
    return &parser->tokens[parser->current];
}

static Token *peek_token(Parser *parser, int offset) {
    int pos = parser->current + offset;
    if (pos >= parser->count) {
        return &parser->tokens[parser->count - 1]; // EOF token
    }
    return &parser->tokens[pos];
}

static bool match(Parser *parser, TokenType type) {
    if (current_token(parser)->type == type) {
        parser->current++;
        return true;
    }
    return false;
}

static bool expect(Parser *parser, TokenType type) {
    if (current_token(parser)->type == type) {
        parser->current++;
        return true;
    }
    
    error("Expected token type %d but got %d at line %d", 
          type, current_token(parser)->type, current_token(parser)->line);
    return false;
}

static ASTNode *create_ast_node(ASTNodeType type) {
    ASTNode *node = calloc(1, sizeof(ASTNode));
    if (!node) {
        error("Memory allocation failed");
        return NULL;
    }
    node->type = type;
    return node;
}

// Forward declarations for recursive parsing
static ASTNode *parse_primary(Parser *parser);
static ASTNode *parse_unary(Parser *parser);
static ASTNode *parse_multiplicative(Parser *parser);
static ASTNode *parse_additive(Parser *parser);
static ASTNode *parse_relational(Parser *parser);
static ASTNode *parse_equality(Parser *parser);
static ASTNode *parse_assignment(Parser *parser);

static DataType token_to_data_type(TokenType token) {
    switch (token) {
        case TOKEN_INT: return TYPE_INT;
        case TOKEN_FLOAT: return TYPE_FLOAT;
        case TOKEN_CHAR: return TYPE_CHAR;
        case TOKEN_VOID: return TYPE_VOID;
        default: return TYPE_INT; // Default fallback
    }
}

static ASTNode *parse_primary(Parser *parser) {
    Token *token = current_token(parser);
    
    if (token->type == TOKEN_NUMBER) {
        ASTNode *node = create_ast_node(AST_NUMBER);
        if (!node) return NULL;
        
        // Determine if it's int or float
        if (strchr(token->value, '.')) {
            node->data_type = TYPE_FLOAT;
            node->number.float_value = atof(token->value);
        } else {
            node->data_type = TYPE_INT;
            node->number.int_value = atoi(token->value);
        }
        
        parser->current++;
        return node;
    }
    
    if (token->type == TOKEN_STRING) {
        ASTNode *node = create_ast_node(AST_STRING);
        if (!node) return NULL;
        
        node->data_type = TYPE_POINTER; // String is char*
        node->string.value = strdup(token->value);
        
        parser->current++;
        return node;
    }
    
    if (token->type == TOKEN_IDENTIFIER) {
        ASTNode *node = create_ast_node(AST_IDENTIFIER);
        if (!node) return NULL;
        
        node->identifier.name = strdup(token->value);
        parser->current++;
        
        // Check for function call
        if (current_token(parser)->type == TOKEN_LPAREN) {
            // Convert to function call
            node->type = AST_FUNCTION_CALL;
            node->function_call.name = node->identifier.name;
            node->function_call.args = NULL;
            node->function_call.arg_count = 0;
            
            parser->current++; // consume '('
            
            // Parse arguments
            if (current_token(parser)->type != TOKEN_RPAREN) {
                int capacity = 10;
                node->function_call.args = malloc(capacity * sizeof(ASTNode*));
                
                do {
                    if (node->function_call.arg_count >= capacity) {
                        capacity *= 2;
                        node->function_call.args = realloc(node->function_call.args, 
                                                         capacity * sizeof(ASTNode*));
                    }
                    
                    node->function_call.args[node->function_call.arg_count] = parse_expression(parser);
                    node->function_call.arg_count++;
                    
                } while (match(parser, TOKEN_COMMA));
            }
            
            expect(parser, TOKEN_RPAREN);
        }
        // Check for array access
        else if (current_token(parser)->type == TOKEN_LBRACKET) {
            // Convert to array access
            ASTNode *array_node = node;
            node = create_ast_node(AST_ARRAY_ACCESS);
            node->array_access.array = array_node;
            
            parser->current++; // consume '['
            node->array_access.index = parse_expression(parser);
            expect(parser, TOKEN_RBRACKET);
        }
        
        return node;
    }
    
    if (match(parser, TOKEN_LPAREN)) {
        ASTNode *expr = parse_expression(parser);
        expect(parser, TOKEN_RPAREN);
        return expr;
    }
    
    // Parallel constructs
    if (token->type == TOKEN_THREAD_ID) {
        ASTNode *node = create_ast_node(AST_FUNCTION_CALL);
        node->function_call.name = strdup("thread_id");
        node->function_call.args = NULL;
        node->function_call.arg_count = 0;
        parser->current++;
        
        expect(parser, TOKEN_LPAREN);
        expect(parser, TOKEN_RPAREN);
        
        return node;
    }
    
    error("Unexpected token in primary expression at line %d", token->line);
    return NULL;
}

static ASTNode *parse_unary(Parser *parser) {
    Token *token = current_token(parser);
    
    if (token->type == TOKEN_MINUS || token->type == TOKEN_PLUS) {
        ASTNode *node = create_ast_node(AST_UNARY_OP);
        if (!node) return NULL;
        
        node->unary_op.operator = token->type;
        parser->current++;
        node->unary_op.operand = parse_unary(parser);
        
        return node;
    }
    
    return parse_primary(parser);
}

static ASTNode *parse_multiplicative(Parser *parser) {
    ASTNode *left = parse_unary(parser);
    if (!left) return NULL;
    
    while (current_token(parser)->type == TOKEN_MULTIPLY || 
           current_token(parser)->type == TOKEN_DIVIDE) {
        
        ASTNode *node = create_ast_node(AST_BINARY_OP);
        if (!node) return NULL;
        
        node->binary_op.left = left;
        node->binary_op.operator = current_token(parser)->type;
        parser->current++;
        node->binary_op.right = parse_unary(parser);
        
        left = node;
    }
    
    return left;
}

static ASTNode *parse_additive(Parser *parser) {
    ASTNode *left = parse_multiplicative(parser);
    if (!left) return NULL;
    
    while (current_token(parser)->type == TOKEN_PLUS || 
           current_token(parser)->type == TOKEN_MINUS) {
        
        ASTNode *node = create_ast_node(AST_BINARY_OP);
        if (!node) return NULL;
        
        node->binary_op.left = left;
        node->binary_op.operator = current_token(parser)->type;
        parser->current++;
        node->binary_op.right = parse_multiplicative(parser);
        
        left = node;
    }
    
    return left;
}

static ASTNode *parse_relational(Parser *parser) {
    ASTNode *left = parse_additive(parser);
    if (!left) return NULL;
    
    while (current_token(parser)->type == TOKEN_LESS || 
           current_token(parser)->type == TOKEN_GREATER ||
           current_token(parser)->type == TOKEN_LESS_EQUAL ||
           current_token(parser)->type == TOKEN_GREATER_EQUAL) {
        
        ASTNode *node = create_ast_node(AST_BINARY_OP);
        if (!node) return NULL;
        
        node->binary_op.left = left;
        node->binary_op.operator = current_token(parser)->type;
        parser->current++;
        node->binary_op.right = parse_additive(parser);
        
        left = node;
    }
    
    return left;
}

static ASTNode *parse_equality(Parser *parser) {
    ASTNode *left = parse_relational(parser);
    if (!left) return NULL;
    
    while (current_token(parser)->type == TOKEN_EQUAL || 
           current_token(parser)->type == TOKEN_NOT_EQUAL) {
        
        ASTNode *node = create_ast_node(AST_BINARY_OP);
        if (!node) return NULL;
        
        node->binary_op.left = left;
        node->binary_op.operator = current_token(parser)->type;
        parser->current++;
        node->binary_op.right = parse_relational(parser);
        
        left = node;
    }
    
    return left;
}

static ASTNode *parse_expression(Parser *parser) {
    return parse_equality(parser);
}

static ASTNode *parse_assignment(Parser *parser) {
    ASTNode *left = parse_expression(parser);
    if (!left) return NULL;
    
    if (current_token(parser)->type == TOKEN_ASSIGN) {
        ASTNode *node = create_ast_node(AST_ASSIGNMENT);
        if (!node) return NULL;
        
        node->assignment.target = left;
        parser->current++;
        node->assignment.value = parse_assignment(parser);
        
        return node;
    }
    
    return left;
}

static ASTNode *parse_variable_declaration(Parser *parser) {
    DataType type = token_to_data_type(current_token(parser)->type);
    parser->current++; // consume type token
    
    if (current_token(parser)->type != TOKEN_IDENTIFIER) {
        error("Expected identifier in variable declaration at line %d", 
              current_token(parser)->line);
        return NULL;
    }
    
    ASTNode *node = create_ast_node(AST_VARIABLE_DECL);
    if (!node) return NULL;
    
    node->var_decl.var_type = type;
    node->var_decl.name = strdup(current_token(parser)->value);
    node->var_decl.array_size = -1; // Not an array by default
    parser->current++;
    
    // Check for array declaration
    if (current_token(parser)->type == TOKEN_LBRACKET) {
        parser->current++;
        if (current_token(parser)->type == TOKEN_NUMBER) {
            node->var_decl.array_size = atoi(current_token(parser)->value);
            parser->current++;
        }
        expect(parser, TOKEN_RBRACKET);
    }
    
    // Check for initialization
    if (current_token(parser)->type == TOKEN_ASSIGN) {
        parser->current++;
        node->var_decl.init_value = parse_expression(parser);
    } else {
        node->var_decl.init_value = NULL;
    }
    
    expect(parser, TOKEN_SEMICOLON);
    return node;
}

static ASTNode *parse_if_statement(Parser *parser) {
    parser->current++; // consume 'if'
    
    ASTNode *node = create_ast_node(AST_IF_STMT);
    if (!node) return NULL;
    
    expect(parser, TOKEN_LPAREN);
    node->if_stmt.condition = parse_expression(parser);
    expect(parser, TOKEN_RPAREN);
    
    node->if_stmt.then_stmt = parse_statement(parser);
    
    if (current_token(parser)->type == TOKEN_ELSE) {
        parser->current++;
        node->if_stmt.else_stmt = parse_statement(parser);
    } else {
        node->if_stmt.else_stmt = NULL;
    }
    
    return node;
}

static ASTNode *parse_while_statement(Parser *parser) {
    parser->current++; // consume 'while'
    
    ASTNode *node = create_ast_node(AST_WHILE_STMT);
    if (!node) return NULL;
    
    expect(parser, TOKEN_LPAREN);
    node->while_stmt.condition = parse_expression(parser);
    expect(parser, TOKEN_RPAREN);
    
    node->while_stmt.body = parse_statement(parser);
    
    return node;
}

static ASTNode *parse_for_statement(Parser *parser) {
    parser->current++; // consume 'for'
    
    ASTNode *node = create_ast_node(AST_FOR_STMT);
    if (!node) return NULL;
    
    expect(parser, TOKEN_LPAREN);
    
    // Init
    if (current_token(parser)->type != TOKEN_SEMICOLON) {
        node->for_stmt.init = parse_assignment(parser);
    }
    expect(parser, TOKEN_SEMICOLON);
    
    // Condition
    if (current_token(parser)->type != TOKEN_SEMICOLON) {
        node->for_stmt.condition = parse_expression(parser);
    }
    expect(parser, TOKEN_SEMICOLON);
    
    // Update
    if (current_token(parser)->type != TOKEN_RPAREN) {
        node->for_stmt.update = parse_assignment(parser);
    }
    expect(parser, TOKEN_RPAREN);
    
    node->for_stmt.body = parse_statement(parser);
    
    return node;
}

static ASTNode *parse_parallel_for(Parser *parser) {
    parser->current++; // consume 'parallel_for'
    
    ASTNode *node = create_ast_node(AST_PARALLEL_FOR);
    if (!node) return NULL;
    
    expect(parser, TOKEN_LPAREN);
    node->parallel_for.start = parse_expression(parser);
    expect(parser, TOKEN_COMMA);
    node->parallel_for.end = parse_expression(parser);
    expect(parser, TOKEN_COMMA);
    node->parallel_for.body = parse_statement(parser);
    expect(parser, TOKEN_RPAREN);
    expect(parser, TOKEN_SEMICOLON);
    
    return node;
}

static ASTNode *parse_return_statement(Parser *parser) {
    parser->current++; // consume 'return'
    
    ASTNode *node = create_ast_node(AST_RETURN_STMT);
    if (!node) return NULL;
    
    if (current_token(parser)->type != TOKEN_SEMICOLON) {
        node->return_stmt.value = parse_expression(parser);
    } else {
        node->return_stmt.value = NULL;
    }
    
    expect(parser, TOKEN_SEMICOLON);
    return node;
}

static ASTNode *parse_block(Parser *parser) {
    expect(parser, TOKEN_LBRACE);
    
    ASTNode *node = create_ast_node(AST_BLOCK);
    if (!node) return NULL;
    
    int capacity = 10;
    node->block.statements = malloc(capacity * sizeof(ASTNode*));
    node->block.statement_count = 0;
    
    while (current_token(parser)->type != TOKEN_RBRACE && 
           current_token(parser)->type != TOKEN_EOF) {
        
        if (node->block.statement_count >= capacity) {
            capacity *= 2;
            node->block.statements = realloc(node->block.statements, 
                                           capacity * sizeof(ASTNode*));
        }
        
        ASTNode *stmt = parse_statement(parser);
        if (stmt) {
            node->block.statements[node->block.statement_count++] = stmt;
        }
    }
    
    expect(parser, TOKEN_RBRACE);
    return node;
}

static ASTNode *parse_statement(Parser *parser) {
    Token *token = current_token(parser);
    
    switch (token->type) {
        case TOKEN_INT:
        case TOKEN_FLOAT:
        case TOKEN_CHAR:
        case TOKEN_VOID:
            return parse_variable_declaration(parser);
            
        case TOKEN_IF:
            return parse_if_statement(parser);
            
        case TOKEN_WHILE:
            return parse_while_statement(parser);
            
        case TOKEN_FOR:
            return parse_for_statement(parser);
            
        case TOKEN_PARALLEL_FOR:
            return parse_parallel_for(parser);
            
        case TOKEN_RETURN:
            return parse_return_statement(parser);
            
        case TOKEN_LBRACE:
            return parse_block(parser);
            
        default: {
            // Expression statement
            ASTNode *expr = parse_assignment(parser);
            expect(parser, TOKEN_SEMICOLON);
            return expr;
        }
    }
}

static ASTNode *parse_function(Parser *parser) {
    // Return type
    DataType return_type = token_to_data_type(current_token(parser)->type);
    parser->current++;
    
    // Function name
    if (current_token(parser)->type != TOKEN_IDENTIFIER) {
        error("Expected function name at line %d", current_token(parser)->line);
        return NULL;
    }
    
    ASTNode *node = create_ast_node(AST_FUNCTION_DEF);
    if (!node) return NULL;
    
    node->function_def.return_type = return_type;
    node->function_def.name = strdup(current_token(parser)->value);
    parser->current++;
    
    // Parameters
    expect(parser, TOKEN_LPAREN);
    
    int capacity = 10;
    node->function_def.params = malloc(capacity * sizeof(ASTNode*));
    node->function_def.param_count = 0;
    
    while (current_token(parser)->type != TOKEN_RPAREN && 
           current_token(parser)->type != TOKEN_EOF) {
        
        if (node->function_def.param_count >= capacity) {
            capacity *= 2;
            node->function_def.params = realloc(node->function_def.params, 
                                              capacity * sizeof(ASTNode*));
        }
        
        // Parameter type
        DataType param_type = token_to_data_type(current_token(parser)->type);
        parser->current++;
        
        // Parameter name
        if (current_token(parser)->type != TOKEN_IDENTIFIER) {
            error("Expected parameter name at line %d", current_token(parser)->line);
            return NULL;
        }
        
        ASTNode *param = create_ast_node(AST_VARIABLE_DECL);
        param->var_decl.var_type = param_type;
        param->var_decl.name = strdup(current_token(parser)->value);
        param->var_decl.init_value = NULL;
        param->var_decl.array_size = -1;
        parser->current++;
        
        node->function_def.params[node->function_def.param_count++] = param;
        
        if (current_token(parser)->type == TOKEN_COMMA) {
            parser->current++;
        }
    }
    
    expect(parser, TOKEN_RPAREN);
    
    // Function body
    node->function_def.body = parse_block(parser);
    
    return node;
}

static ASTNode *parse_program(Parser *parser) {
    ASTNode *node = create_ast_node(AST_PROGRAM);
    if (!node) return NULL;
    
    int capacity = 10;
    node->program.functions = malloc(capacity * sizeof(ASTNode*));
    node->program.function_count = 0;
    
    while (current_token(parser)->type != TOKEN_EOF) {
        if (node->program.function_count >= capacity) {
            capacity *= 2;
            node->program.functions = realloc(node->program.functions, 
                                            capacity * sizeof(ASTNode*));
        }
        
        ASTNode *func = parse_function(parser);
        if (func) {
            node->program.functions[node->program.function_count++] = func;
        }
    }
    
    return node;
}

ASTNode *parse(Token *tokens, int token_count) {
    if (!tokens || token_count == 0) {
        return NULL;
    }
    
    Parser parser = {
        .tokens = tokens,
        .current = 0,
        .count = token_count
    };
    
    return parse_program(&parser);
}

void free_ast(ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->program.function_count; i++) {
                free_ast(node->program.functions[i]);
            }
            free(node->program.functions);
            break;
            
        case AST_FUNCTION_DEF:
            free(node->function_def.name);
            for (int i = 0; i < node->function_def.param_count; i++) {
                free_ast(node->function_def.params[i]);
            }
            free(node->function_def.params);
            free_ast(node->function_def.body);
            break;
            
        case AST_VARIABLE_DECL:
            free(node->var_decl.name);
            free_ast(node->var_decl.init_value);
            break;
            
        case AST_ASSIGNMENT:
            free_ast(node->assignment.target);
            free_ast(node->assignment.value);
            break;
            
        case AST_BINARY_OP:
            free_ast(node->binary_op.left);
            free_ast(node->binary_op.right);
            break;
            
        case AST_UNARY_OP:
            free_ast(node->unary_op.operand);
            break;
            
        case AST_FUNCTION_CALL:
            free(node->function_call.name);
            for (int i = 0; i < node->function_call.arg_count; i++) {
                free_ast(node->function_call.args[i]);
            }
            free(node->function_call.args);
            break;
            
        case AST_IF_STMT:
            free_ast(node->if_stmt.condition);
            free_ast(node->if_stmt.then_stmt);
            free_ast(node->if_stmt.else_stmt);
            break;
            
        case AST_WHILE_STMT:
            free_ast(node->while_stmt.condition);
            free_ast(node->while_stmt.body);
            break;
            
        case AST_FOR_STMT:
            free_ast(node->for_stmt.init);
            free_ast(node->for_stmt.condition);
            free_ast(node->for_stmt.update);
            free_ast(node->for_stmt.body);
            break;
            
        case AST_PARALLEL_FOR:
            free_ast(node->parallel_for.start);
            free_ast(node->parallel_for.end);
            free_ast(node->parallel_for.body);
            break;
            
        case AST_RETURN_STMT:
            free_ast(node->return_stmt.value);
            break;
            
        case AST_BLOCK:
            for (int i = 0; i < node->block.statement_count; i++) {
                free_ast(node->block.statements[i]);
            }
            free(node->block.statements);
            break;
            
        case AST_IDENTIFIER:
            free(node->identifier.name);
            break;
            
        case AST_STRING:
            free(node->string.value);
            break;
            
        case AST_ARRAY_ACCESS:
            free_ast(node->array_access.array);
            free_ast(node->array_access.index);
            break;
            
        default:
            break;
    }
    
    free(node);
}

void print_ast(ASTNode *node, int indent) {
    if (!node) return;
    
    for (int i = 0; i < indent; i++) printf("  ");
    
    switch (node->type) {
        case AST_PROGRAM:
            printf("PROGRAM\n");
            for (int i = 0; i < node->program.function_count; i++) {
                print_ast(node->program.functions[i], indent + 1);
            }
            break;
            
        case AST_FUNCTION_DEF:
            printf("FUNCTION: %s\n", node->function_def.name);
            for (int i = 0; i < node->function_def.param_count; i++) {
                print_ast(node->function_def.params[i], indent + 1);
            }
            print_ast(node->function_def.body, indent + 1);
            break;
            
        case AST_VARIABLE_DECL:
            printf("VAR_DECL: %s\n", node->var_decl.name);
            if (node->var_decl.init_value) {
                print_ast(node->var_decl.init_value, indent + 1);
            }
            break;
            
        case AST_ASSIGNMENT:
            printf("ASSIGNMENT\n");
            print_ast(node->assignment.target, indent + 1);
            print_ast(node->assignment.value, indent + 1);
            break;
            
        case AST_BINARY_OP:
            printf("BINARY_OP: %d\n", node->binary_op.operator);
            print_ast(node->binary_op.left, indent + 1);
            print_ast(node->binary_op.right, indent + 1);
            break;
            
        case AST_IDENTIFIER:
            printf("IDENTIFIER: %s\n", node->identifier.name);
            break;
            
        case AST_NUMBER:
            if (node->data_type == TYPE_FLOAT) {
                printf("NUMBER: %f\n", node->number.float_value);
            } else {
                printf("NUMBER: %d\n", node->number.int_value);
            }
            break;
            
        default:
            printf("AST_NODE: %d\n", node->type);
            break;
    }
}
