/*
 * ParallelC++ Compiler - Parser (Syntax Analyzer)
 * 
 * This module implements recursive descent parsing for C++ with parallel extensions.
 * It builds an Abstract Syntax Tree (AST) from the token stream.
 */

#include "pcpp.h"

typedef struct {
    Token *tokens;
    int current;
    int count;
} Parser;

// Forward declarations
static ASTNode *parse_expression(Parser *parser);
static ASTNode *parse_statement(Parser *parser);
static ASTNode *parse_class_definition(Parser *parser);
static ASTNode *parse_function_definition(Parser *parser);
static ASTNode *parse_method_definition(Parser *parser, const char *class_name);

static Token *current_token(Parser *parser) {
    if (parser->current >= parser->count) {
        return &parser->tokens[parser->count - 1]; // EOF token
    }
    return &parser->tokens[parser->current];
}

static Token *peek_token(Parser *parser, int offset) {
    int pos = parser->current + offset;
    if (pos >= parser->count) {
        return &parser->tokens[parser->count - 1];
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

static DataType token_to_data_type(TokenType token) {
    switch (token) {
        case TOKEN_INT: return TYPE_INT;
        case TOKEN_FLOAT: return TYPE_FLOAT;
        case TOKEN_DOUBLE: return TYPE_DOUBLE;
        case TOKEN_CHAR_TYPE: return TYPE_CHAR;
        case TOKEN_BOOL: return TYPE_BOOL;
        case TOKEN_VOID: return TYPE_VOID;
        default: return TYPE_INT;
    }
}

static AccessModifier token_to_access_modifier(TokenType token) {
    switch (token) {
        case TOKEN_PUBLIC: return ACCESS_PUBLIC;
        case TOKEN_PRIVATE: return ACCESS_PRIVATE;
        case TOKEN_PROTECTED: return ACCESS_PROTECTED;
        default: return ACCESS_PRIVATE; // Default in C++
    }
}

// Parse primary expressions
static ASTNode *parse_primary(Parser *parser) {
    Token *token = current_token(parser);
    
    if (token->type == TOKEN_NUMBER) {
        ASTNode *node = create_ast_node(AST_NUMBER);
        if (!node) return NULL;
        
        if (strchr(token->value, '.') || strchr(token->value, 'f') || strchr(token->value, 'F')) {
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
        
        node->data_type = TYPE_POINTER;
        node->string.value = strdup(token->value);
        
        parser->current++;
        return node;
    }
    
    if (token->type == TOKEN_CHAR) {
        ASTNode *node = create_ast_node(AST_CHAR_LITERAL);
        if (!node) return NULL;
        
        node->data_type = TYPE_CHAR;
        node->char_literal.value = token->value[0];
        
        parser->current++;
        return node;
    }
    
    if (token->type == TOKEN_THIS) {
        ASTNode *node = create_ast_node(AST_THIS_POINTER);
        if (!node) return NULL;
        
        node->data_type = TYPE_POINTER;
        parser->current++;
        return node;
    }
    
    if (token->type == TOKEN_IDENTIFIER) {
        ASTNode *node = create_ast_node(AST_IDENTIFIER);
        if (!node) return NULL;
        
        node->identifier.name = strdup(token->value);
        
        // Check for scope resolution
        if (peek_token(parser, 1)->type == TOKEN_SCOPE_RESOLUTION) {
            parser->current += 2; // Skip identifier and ::
            if (current_token(parser)->type == TOKEN_IDENTIFIER) {
                node->identifier.class_scope = node->identifier.name;
                node->identifier.name = strdup(current_token(parser)->value);
                parser->current++;
            }
        } else {
            parser->current++;
        }
        
        // Check for function call
        if (current_token(parser)->type == TOKEN_LPAREN) {
            node->type = AST_FUNCTION_CALL;
            node->function_call.name = node->identifier.name;
            node->function_call.class_scope = node->identifier.class_scope;
            node->function_call.args = NULL;
            node->function_call.arg_count = 0;
            
            parser->current++; // consume '('
            
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
            ASTNode *array_node = node;
            node = create_ast_node(AST_ARRAY_ACCESS);
            node->array_access.array = array_node;
            
            parser->current++; // consume '['
            node->array_access.index = parse_expression(parser);
            expect(parser, TOKEN_RBRACKET);
        }
        // Check for member access
        else if (current_token(parser)->type == TOKEN_DOT || 
                 current_token(parser)->type == TOKEN_ARROW) {
            ASTNode *object_node = node;
            node = create_ast_node(AST_MEMBER_ACCESS);
            node->member_access.object = object_node;
            node->member_access.is_arrow = (current_token(parser)->type == TOKEN_ARROW);
            
            parser->current++; // consume '.' or '->'
            
            if (current_token(parser)->type == TOKEN_IDENTIFIER) {
                node->member_access.member_name = strdup(current_token(parser)->value);
                parser->current++;
                
                // Check if it's a method call
                if (current_token(parser)->type == TOKEN_LPAREN) {
                    ASTNode *method_node = create_ast_node(AST_METHOD_CALL);
                    method_node->method_call.object = node->member_access.object;
                    method_node->method_call.method_name = node->member_access.member_name;
                    method_node->method_call.is_arrow = node->member_access.is_arrow;
                    method_node->method_call.args = NULL;
                    method_node->method_call.arg_count = 0;
                    
                    parser->current++; // consume '('
                    
                    if (current_token(parser)->type != TOKEN_RPAREN) {
                        int capacity = 10;
                        method_node->method_call.args = malloc(capacity * sizeof(ASTNode*));
                        
                        do {
                            if (method_node->method_call.arg_count >= capacity) {
                                capacity *= 2;
                                method_node->method_call.args = realloc(method_node->method_call.args, 
                                                                      capacity * sizeof(ASTNode*));
                            }
                            
                            method_node->method_call.args[method_node->method_call.arg_count] = parse_expression(parser);
                            method_node->method_call.arg_count++;
                            
                        } while (match(parser, TOKEN_COMMA));
                    }
                    
                    expect(parser, TOKEN_RPAREN);
                    free(node); // Free the member access node
                    node = method_node;
                }
            }
        }
        
        return node;
    }
    
    // new expression
    if (token->type == TOKEN_NEW) {
        ASTNode *node = create_ast_node(AST_NEW_EXPR);
        if (!node) return NULL;
        
        parser->current++; // consume 'new'
        
        // Parse type
        if (current_token(parser)->type == TOKEN_IDENTIFIER) {
            node->new_expr.class_name = strdup(current_token(parser)->value);
            node->new_expr.type = TYPE_CLASS;
            parser->current++;
        } else {
            node->new_expr.type = token_to_data_type(current_token(parser)->type);
            parser->current++;
        }
        
        // Check for array allocation
        if (current_token(parser)->type == TOKEN_LBRACKET) {
            node->new_expr.is_array = true;
            parser->current++; // consume '['
            node->new_expr.array_size = parse_expression(parser);
            expect(parser, TOKEN_RBRACKET);
        } else {
            node->new_expr.is_array = false;
            
            // Parse constructor arguments
            if (current_token(parser)->type == TOKEN_LPAREN) {
                parser->current++; // consume '('
                
                int capacity = 10;
                node->new_expr.args = malloc(capacity * sizeof(ASTNode*));
                node->new_expr.arg_count = 0;
                
                if (current_token(parser)->type != TOKEN_RPAREN) {
                    do {
                        if (node->new_expr.arg_count >= capacity) {
                            capacity *= 2;
                            node->new_expr.args = realloc(node->new_expr.args, 
                                                        capacity * sizeof(ASTNode*));
                        }
                        
                        node->new_expr.args[node->new_expr.arg_count] = parse_expression(parser);
                        node->new_expr.arg_count++;
                        
                    } while (match(parser, TOKEN_COMMA));
                }
                
                expect(parser, TOKEN_RPAREN);
            }
        }
        
        return node;
    }
    
    // delete expression
    if (token->type == TOKEN_DELETE) {
        ASTNode *node = create_ast_node(AST_DELETE_EXPR);
        if (!node) return NULL;
        
        parser->current++; // consume 'delete'
        
        // Check for array delete
        if (current_token(parser)->type == TOKEN_LBRACKET) {
            node->delete_expr.is_array = true;
            parser->current++; // consume '['
            expect(parser, TOKEN_RBRACKET);
        } else {
            node->delete_expr.is_array = false;
        }
        
        node->delete_expr.operand = parse_expression(parser);
        return node;
    }
    
    if (match(parser, TOKEN_LPAREN)) {
        ASTNode *expr = parse_expression(parser);
        expect(parser, TOKEN_RPAREN);
        return expr;
    }
    
    // Parallel extensions
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

// Parse unary expressions
static ASTNode *parse_unary(Parser *parser) {
    Token *token = current_token(parser);
    
    if (token->type == TOKEN_MINUS || token->type == TOKEN_PLUS || 
        token->type == TOKEN_NOT || token->type == TOKEN_BITWISE_NOT ||
        token->type == TOKEN_INCREMENT || token->type == TOKEN_DECREMENT) {
        
        ASTNode *node = create_ast_node(AST_UNARY_OP);
        if (!node) return NULL;
        
        node->unary_op.operator = token->type;
        parser->current++;
        node->unary_op.operand = parse_unary(parser);
        
        return node;
    }
    
    // Address-of operator
    if (token->type == TOKEN_BITWISE_AND) {
        ASTNode *node = create_ast_node(AST_ADDRESS_OF);
        if (!node) return NULL;
        
        parser->current++;
        node->unary_op.operand = parse_unary(parser);
        
        return node;
    }
    
    // Pointer dereference
    if (token->type == TOKEN_MULTIPLY) {
        ASTNode *node = create_ast_node(AST_POINTER_DEREF);
        if (!node) return NULL;
        
        parser->current++;
        node->unary_op.operand = parse_unary(parser);
        
        return node;
    }
    
    return parse_primary(parser);
}

// Parse binary expressions with operator precedence
static ASTNode *parse_multiplicative(Parser *parser) {
    ASTNode *left = parse_unary(parser);
    if (!left) return NULL;
    
    while (current_token(parser)->type == TOKEN_MULTIPLY || 
           current_token(parser)->type == TOKEN_DIVIDE ||
           current_token(parser)->type == TOKEN_MODULO) {
        
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

static ASTNode *parse_logical_and(Parser *parser) {
    ASTNode *left = parse_equality(parser);
    if (!left) return NULL;
    
    while (current_token(parser)->type == TOKEN_LOGICAL_AND) {
        ASTNode *node = create_ast_node(AST_BINARY_OP);
        if (!node) return NULL;
        
        node->binary_op.left = left;
        node->binary_op.operator = current_token(parser)->type;
        parser->current++;
        node->binary_op.right = parse_equality(parser);
        
        left = node;
    }
    
    return left;
}

static ASTNode *parse_logical_or(Parser *parser) {
    ASTNode *left = parse_logical_and(parser);
    if (!left) return NULL;
    
    while (current_token(parser)->type == TOKEN_LOGICAL_OR) {
        ASTNode *node = create_ast_node(AST_BINARY_OP);
        if (!node) return NULL;
        
        node->binary_op.left = left;
        node->binary_op.operator = current_token(parser)->type;
        parser->current++;
        node->binary_op.right = parse_logical_and(parser);
        
        left = node;
    }
    
    return left;
}

static ASTNode *parse_expression(Parser *parser) {
    return parse_logical_or(parser);
}

static ASTNode *parse_assignment(Parser *parser) {
    ASTNode *left = parse_expression(parser);
    if (!left) return NULL;
    
    if (current_token(parser)->type == TOKEN_ASSIGN ||
        current_token(parser)->type == TOKEN_PLUS_ASSIGN ||
        current_token(parser)->type == TOKEN_MINUS_ASSIGN ||
        current_token(parser)->type == TOKEN_MULTIPLY_ASSIGN ||
        current_token(parser)->type == TOKEN_DIVIDE_ASSIGN) {
        
        ASTNode *node = create_ast_node(AST_ASSIGNMENT);
        if (!node) return NULL;
        
        node->assignment.target = left;
        node->assignment.operator = current_token(parser)->type;
        parser->current++;
        node->assignment.value = parse_assignment(parser);
        
        return node;
    }
    
    return left;
}

// Parse variable declarations
static ASTNode *parse_variable_declaration(Parser *parser) {
    DataType type = TYPE_INT;
    char *class_type = NULL;
    bool is_static = false;
    bool is_const = false;
    
    // Handle modifiers
    while (current_token(parser)->type == TOKEN_STATIC || 
           current_token(parser)->type == TOKEN_CONST) {
        if (current_token(parser)->type == TOKEN_STATIC) {
            is_static = true;
        } else if (current_token(parser)->type == TOKEN_CONST) {
            is_const = true;
        }
        parser->current++;
    }
    
    // Parse type
    if (current_token(parser)->type == TOKEN_IDENTIFIER) {
        // Class type
        type = TYPE_CLASS;
        class_type = strdup(current_token(parser)->value);
        parser->current++;
    } else {
        type = token_to_data_type(current_token(parser)->type);
        parser->current++;
    }
    
    if (current_token(parser)->type != TOKEN_IDENTIFIER) {
        error("Expected identifier in variable declaration at line %d", 
              current_token(parser)->line);
        return NULL;
    }
    
    ASTNode *node = create_ast_node(AST_VARIABLE_DECL);
    if (!node) return NULL;
    
    node->var_decl.var_type = type;
    node->var_decl.class_type = class_type;
    node->var_decl.name = strdup(current_token(parser)->value);
    node->var_decl.is_static = is_static;
    node->var_decl.is_const = is_const;
    node->var_decl.array_size = -1;
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

// Parse control flow statements
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

// Parse class members
static ASTNode *parse_constructor(Parser *parser, const char *class_name) {
    ASTNode *node = create_ast_node(AST_CONSTRUCTOR);
    if (!node) return NULL;
    
    node->constructor.class_name = strdup(class_name);
    
    // Skip constructor name (already parsed)
    parser->current++;
    
    // Parse parameters
    expect(parser, TOKEN_LPAREN);
    
    int capacity = 10;
    node->constructor.params = malloc(capacity * sizeof(ASTNode*));
    node->constructor.param_count = 0;
    
    while (current_token(parser)->type != TOKEN_RPAREN && 
           current_token(parser)->type != TOKEN_EOF) {
        
        if (node->constructor.param_count >= capacity) {
            capacity *= 2;
            node->constructor.params = realloc(node->constructor.params, 
                                             capacity * sizeof(ASTNode*));
        }
        
        // Parse parameter
        DataType param_type = token_to_data_type(current_token(parser)->type);
        parser->current++;
        
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
        
        node->constructor.params[node->constructor.param_count++] = param;
        
        if (current_token(parser)->type == TOKEN_COMMA) {
            parser->current++;
        }
    }
    
    expect(parser, TOKEN_RPAREN);
    
    // Parse initializer list (optional)
    if (current_token(parser)->type == TOKEN_COLON) {
        parser->current++;
        // TODO: Parse member initializer list
        node->constructor.initializer_list = NULL;
        node->constructor.initializer_count = 0;
    }
    
    // Parse body
    node->constructor.body = parse_block(parser);
    
    return node;
}

static ASTNode *parse_destructor(Parser *parser, const char *class_name) {
    ASTNode *node = create_ast_node(AST_DESTRUCTOR);
    if (!node) return NULL;
    
    node->destructor.class_name = strdup(class_name);
    node->destructor.is_virtual = false; // TODO: Check for virtual
    
    // Skip destructor name
    parser->current++;
    
    expect(parser, TOKEN_LPAREN);
    expect(parser, TOKEN_RPAREN);
    
    node->destructor.body = parse_block(parser);
    
    return node;
}

static ASTNode *parse_class_definition(Parser *parser) {
    bool is_parallel_class = false;
    
    // Check for parallel_class
    if (current_token(parser)->type == TOKEN_PARALLEL_CLASS) {
        is_parallel_class = true;
        parser->current++;
    }
    
    expect(parser, TOKEN_CLASS);
    
    if (current_token(parser)->type != TOKEN_IDENTIFIER) {
        error("Expected class name at line %d", current_token(parser)->line);
        return NULL;
    }
    
    ASTNode *node = create_ast_node(AST_CLASS_DEF);
    if (!node) return NULL;
    
    node->class_def.name = strdup(current_token(parser)->value);
    node->class_def.is_parallel_class = is_parallel_class;
    node->class_def.has_base_class = false;
    parser->current++;
    
    // Check for inheritance
    if (current_token(parser)->type == TOKEN_COLON) {
        parser->current++;
        
        // Parse access modifier (optional)
        AccessModifier access = ACCESS_PRIVATE; // Default for class
        if (current_token(parser)->type == TOKEN_PUBLIC ||
            current_token(parser)->type == TOKEN_PRIVATE ||
            current_token(parser)->type == TOKEN_PROTECTED) {
            access = token_to_access_modifier(current_token(parser)->type);
            parser->current++;
        }
        
        if (current_token(parser)->type == TOKEN_IDENTIFIER) {
            node->class_def.base_class = strdup(current_token(parser)->value);
            node->class_def.has_base_class = true;
            parser->current++;
        }
    }
    
    expect(parser, TOKEN_LBRACE);
    
    // Parse class members
    int capacity = 20;
    node->class_def.members = malloc(capacity * sizeof(ASTNode*));
    node->class_def.member_count = 0;
    
    AccessModifier current_access = ACCESS_PRIVATE; // Default for class
    
    while (current_token(parser)->type != TOKEN_RBRACE && 
           current_token(parser)->type != TOKEN_EOF) {
        
        // Check for access modifiers
        if (current_token(parser)->type == TOKEN_PUBLIC ||
            current_token(parser)->type == TOKEN_PRIVATE ||
            current_token(parser)->type == TOKEN_PROTECTED) {
            current_access = token_to_access_modifier(current_token(parser)->type);
            parser->current++;
            expect(parser, TOKEN_COLON);
            continue;
        }
        
        if (node->class_def.member_count >= capacity) {
            capacity *= 2;
            node->class_def.members = realloc(node->class_def.members, 
                                            capacity * sizeof(ASTNode*));
        }
        
        ASTNode *member = NULL;
        
        // Check for constructor (same name as class)
        if (current_token(parser)->type == TOKEN_IDENTIFIER &&
            strcmp(current_token(parser)->value, node->class_def.name) == 0) {
            member = parse_constructor(parser, node->class_def.name);
        }
        // Check for destructor
        else if (current_token(parser)->type == TOKEN_BITWISE_NOT &&
                 peek_token(parser, 1)->type == TOKEN_IDENTIFIER &&
                 strcmp(peek_token(parser, 1)->value, node->class_def.name) == 0) {
            parser->current++; // consume ~
            member = parse_destructor(parser, node->class_def.name);
        }
        // Method or variable declaration
        else {
            // Look ahead to determine if it's a method or variable
            int lookahead = 0;
            while (peek_token(parser, lookahead)->type != TOKEN_SEMICOLON &&
                   peek_token(parser, lookahead)->type != TOKEN_LPAREN &&
                   peek_token(parser, lookahead)->type != TOKEN_LBRACE &&
                   peek_token(parser, lookahead)->type != TOKEN_EOF) {
                lookahead++;
            }
            
            if (peek_token(parser, lookahead)->type == TOKEN_LPAREN) {
                // Method
                member = parse_method_definition(parser, node->class_def.name);
            } else {
                // Variable
                member = parse_variable_declaration(parser);
            }
        }
        
        if (member) {
            node->class_def.members[node->class_def.member_count++] = member;
        }
    }
    
    expect(parser, TOKEN_RBRACE);
    expect(parser, TOKEN_SEMICOLON);
    
    return node;
}

static ASTNode *parse_method_definition(Parser *parser, const char *class_name) {
    bool is_virtual = false;
    bool is_static = false;
    bool is_const = false;
    bool is_thread_safe = false;
    
    // Handle modifiers
    while (current_token(parser)->type == TOKEN_VIRTUAL ||
           current_token(parser)->type == TOKEN_STATIC ||
           current_token(parser)->type == TOKEN_THREAD_SAFE) {
        if (current_token(parser)->type == TOKEN_VIRTUAL) {
            is_virtual = true;
        } else if (current_token(parser)->type == TOKEN_STATIC) {
            is_static = true;
        } else if (current_token(parser)->type == TOKEN_THREAD_SAFE) {
            is_thread_safe = true;
        }
        parser->current++;
    }
    
    // Parse return type
    DataType return_type = token_to_data_type(current_token(parser)->type);
    parser->current++;
    
    // Parse method name
    if (current_token(parser)->type != TOKEN_IDENTIFIER) {
        error("Expected method name at line %d", current_token(parser)->line);
        return NULL;
    }
    
    ASTNode *node = create_ast_node(AST_METHOD);
    if (!node) return NULL;
    
    node->function_def.name = strdup(current_token(parser)->value);
    node->function_def.return_type = return_type;
    node->function_def.is_virtual = is_virtual;
    node->function_def.is_static = is_static;
    node->function_def.is_thread_safe = is_thread_safe;
    node->function_def.class_name = strdup(class_name);
    parser->current++;
    
    // Parse parameters
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
        
        // Parse parameter
        DataType param_type = token_to_data_type(current_token(parser)->type);
        parser->current++;
        
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
    
    // Check for const method
    if (current_token(parser)->type == TOKEN_CONST) {
        is_const = true;
        node->function_def.is_const = is_const;
        parser->current++;
    }
    
    // Parse body
    node->function_def.body = parse_block(parser);
    
    return node;
}

static ASTNode *parse_function_definition(Parser *parser) {
    // Parse return type
    DataType return_type = token_to_data_type(current_token(parser)->type);
    parser->current++;
    
    // Parse function name
    if (current_token(parser)->type != TOKEN_IDENTIFIER) {
        error("Expected function name at line %d", current_token(parser)->line);
        return NULL;
    }
    
    ASTNode *node = create_ast_node(AST_FUNCTION_DEF);
    if (!node) return NULL;
    
    node->function_def.name = strdup(current_token(parser)->value);
    node->function_def.return_type = return_type;
    node->function_def.is_virtual = false;
    node->function_def.is_static = false;
    node->function_def.is_const = false;
    node->function_def.class_name = NULL;
    parser->current++;
    
    // Parse parameters
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
        
        // Parse parameter
        DataType param_type = token_to_data_type(current_token(parser)->type);
        parser->current++;
        
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
    
    // Parse body
    node->function_def.body = parse_block(parser);
    
    return node;
}

static ASTNode *parse_statement(Parser *parser) {
    Token *token = current_token(parser);
    
    switch (token->type) {
        case TOKEN_INT:
        case TOKEN_FLOAT:
        case TOKEN_DOUBLE:
        case TOKEN_CHAR_TYPE:
        case TOKEN_BOOL:
        case TOKEN_VOID:
        case TOKEN_CONST:
        case TOKEN_STATIC:
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
            
        case TOKEN_BREAK:
        case TOKEN_CONTINUE:
            parser->current++;
            expect(parser, TOKEN_SEMICOLON);
            return create_ast_node(token->type == TOKEN_BREAK ? AST_BREAK_STMT : AST_CONTINUE_STMT);
            
        default: {
            // Expression statement
            ASTNode *expr = parse_assignment(parser);
            expect(parser, TOKEN_SEMICOLON);
            return expr;
        }
    }
}

static ASTNode *parse_program(Parser *parser) {
    ASTNode *node = create_ast_node(AST_PROGRAM);
    if (!node) return NULL;
    
    int capacity = 10;
    node->program.declarations = malloc(capacity * sizeof(ASTNode*));
    node->program.declaration_count = 0;
    
    while (current_token(parser)->type != TOKEN_EOF) {
        if (node->program.declaration_count >= capacity) {
            capacity *= 2;
            node->program.declarations = realloc(node->program.declarations, 
                                               capacity * sizeof(ASTNode*));
        }
        
        ASTNode *decl = NULL;
        
        // Check for class definition
        if (current_token(parser)->type == TOKEN_CLASS ||
            current_token(parser)->type == TOKEN_PARALLEL_CLASS) {
            decl = parse_class_definition(parser);
        }
        // Check for function definition
        else {
            // Look ahead to determine if it's a function
            int lookahead = 0;
            while (peek_token(parser, lookahead)->type != TOKEN_SEMICOLON &&
                   peek_token(parser, lookahead)->type != TOKEN_LPAREN &&
                   peek_token(parser, lookahead)->type != TOKEN_EOF) {
                lookahead++;
            }
            
            if (peek_token(parser, lookahead)->type == TOKEN_LPAREN) {
                decl = parse_function_definition(parser);
            } else {
                decl = parse_variable_declaration(parser);
            }
        }
        
        if (decl) {
            node->program.declarations[node->program.declaration_count++] = decl;
        }
    }
    
    return node;
}

ASTNode *parse_cpp(Token *tokens, int token_count) {
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
    
    // TODO: Implement proper AST cleanup for all node types
    // This is a simplified version
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->program.declaration_count; i++) {
                free_ast(node->program.declarations[i]);
            }
            free(node->program.declarations);
            break;
            
        case AST_CLASS_DEF:
            free(node->class_def.name);
            free(node->class_def.base_class);
            for (int i = 0; i < node->class_def.member_count; i++) {
                free_ast(node->class_def.members[i]);
            }
            free(node->class_def.members);
            break;
            
        default:
            // TODO: Add cases for all node types
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
            for (int i = 0; i < node->program.declaration_count; i++) {
                print_ast(node->program.declarations[i], indent + 1);
            }
            break;
            
        case AST_CLASS_DEF:
            printf("CLASS: %s", node->class_def.name);
            if (node->class_def.has_base_class) {
                printf(" : %s", node->class_def.base_class);
            }
            if (node->class_def.is_parallel_class) {
                printf(" [PARALLEL]");
            }
            printf("\n");
            for (int i = 0; i < node->class_def.member_count; i++) {
                print_ast(node->class_def.members[i], indent + 1);
            }
            break;
            
        case AST_FUNCTION_DEF:
        case AST_METHOD:
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
            
        case AST_IDENTIFIER:
            printf("IDENTIFIER: %s\n", node->identifier.name);
            break;
            
        case AST_NUMBER:
            printf("NUMBER: %d\n", node->number.int_value);
            break;
            
        default:
            printf("AST_NODE: %d\n", node->type);
            break;
    }
}
