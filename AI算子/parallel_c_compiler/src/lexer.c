/*
 * ParallelC Compiler - Lexical Analyzer
 * 
 * This module implements the lexical analysis phase of the compiler.
 * It converts the input source code into a stream of tokens.
 */

#include "pcc.h"
#include <ctype.h>

// Keywords table
static struct {
    const char *keyword;
    TokenType token_type;
} keywords[] = {
    {"int", TOKEN_INT},
    {"float", TOKEN_FLOAT},
    {"char", TOKEN_CHAR},
    {"void", TOKEN_VOID},
    {"if", TOKEN_IF},
    {"else", TOKEN_ELSE},
    {"while", TOKEN_WHILE},
    {"for", TOKEN_FOR},
    {"return", TOKEN_RETURN},
    {"parallel_for", TOKEN_PARALLEL_FOR},
    {"parallel_reduce", TOKEN_PARALLEL_REDUCE},
    {"atomic_add", TOKEN_ATOMIC_ADD},
    {"barrier", TOKEN_BARRIER},
    {"thread_id", TOKEN_THREAD_ID},
    {NULL, TOKEN_ERROR}
};

static TokenType lookup_keyword(const char *identifier) {
    for (int i = 0; keywords[i].keyword != NULL; i++) {
        if (strcmp(identifier, keywords[i].keyword) == 0) {
            return keywords[i].token_type;
        }
    }
    return TOKEN_IDENTIFIER;
}

static bool is_alpha(char c) {
    return isalpha(c) || c == '_';
}

static bool is_alnum(char c) {
    return isalnum(c) || c == '_';
}

static Token create_token(TokenType type, const char *value, int line, int column) {
    Token token;
    token.type = type;
    token.value = value ? strdup(value) : NULL;
    token.line = line;
    token.column = column;
    return token;
}

Token *tokenize(const char *source, int *token_count) {
    if (!source || !token_count) {
        return NULL;
    }
    
    int capacity = 1000;
    Token *tokens = malloc(capacity * sizeof(Token));
    if (!tokens) {
        return NULL;
    }
    
    *token_count = 0;
    int line = 1;
    int column = 1;
    int pos = 0;
    int len = strlen(source);
    
    while (pos < len) {
        // Resize token array if needed
        if (*token_count >= capacity - 1) {
            capacity *= 2;
            Token *new_tokens = realloc(tokens, capacity * sizeof(Token));
            if (!new_tokens) {
                free_tokens(tokens, *token_count);
                return NULL;
            }
            tokens = new_tokens;
        }
        
        char current = source[pos];
        
        // Skip whitespace
        if (isspace(current)) {
            if (current == '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
            pos++;
            continue;
        }
        
        // Skip comments
        if (current == '/' && pos + 1 < len && source[pos + 1] == '/') {
            // Single line comment
            while (pos < len && source[pos] != '\n') {
                pos++;
            }
            continue;
        }
        
        if (current == '/' && pos + 1 < len && source[pos + 1] == '*') {
            // Multi-line comment
            pos += 2;
            while (pos + 1 < len && !(source[pos] == '*' && source[pos + 1] == '/')) {
                if (source[pos] == '\n') {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
                pos++;
            }
            if (pos + 1 < len) {
                pos += 2; // Skip */
                column += 2;
            }
            continue;
        }
        
        int start_column = column;
        
        // Numbers
        if (isdigit(current)) {
            int start = pos;
            bool has_dot = false;
            
            while (pos < len && (isdigit(source[pos]) || (!has_dot && source[pos] == '.'))) {
                if (source[pos] == '.') {
                    has_dot = true;
                }
                pos++;
                column++;
            }
            
            char *number_str = malloc(pos - start + 1);
            strncpy(number_str, source + start, pos - start);
            number_str[pos - start] = '\0';
            
            tokens[*token_count] = create_token(TOKEN_NUMBER, number_str, line, start_column);
            free(number_str);
            (*token_count)++;
            continue;
        }
        
        // Identifiers and keywords
        if (is_alpha(current)) {
            int start = pos;
            
            while (pos < len && is_alnum(source[pos])) {
                pos++;
                column++;
            }
            
            char *identifier = malloc(pos - start + 1);
            strncpy(identifier, source + start, pos - start);
            identifier[pos - start] = '\0';
            
            TokenType type = lookup_keyword(identifier);
            tokens[*token_count] = create_token(type, identifier, line, start_column);
            free(identifier);
            (*token_count)++;
            continue;
        }
        
        // String literals
        if (current == '"') {
            int start = pos + 1;
            pos++; // Skip opening quote
            column++;
            
            while (pos < len && source[pos] != '"') {
                if (source[pos] == '\\' && pos + 1 < len) {
                    pos += 2; // Skip escaped character
                    column += 2;
                } else {
                    pos++;
                    column++;
                }
            }
            
            if (pos >= len) {
                error("Unterminated string literal at line %d", line);
                free_tokens(tokens, *token_count);
                return NULL;
            }
            
            char *string_value = malloc(pos - start + 1);
            strncpy(string_value, source + start, pos - start);
            string_value[pos - start] = '\0';
            
            tokens[*token_count] = create_token(TOKEN_STRING, string_value, line, start_column);
            free(string_value);
            (*token_count)++;
            
            pos++; // Skip closing quote
            column++;
            continue;
        }
        
        // Two-character operators
        if (pos + 1 < len) {
            char next = source[pos + 1];
            TokenType two_char_token = TOKEN_ERROR;
            
            if (current == '=' && next == '=') two_char_token = TOKEN_EQUAL;
            else if (current == '!' && next == '=') two_char_token = TOKEN_NOT_EQUAL;
            else if (current == '<' && next == '=') two_char_token = TOKEN_LESS_EQUAL;
            else if (current == '>' && next == '=') two_char_token = TOKEN_GREATER_EQUAL;
            
            if (two_char_token != TOKEN_ERROR) {
                tokens[*token_count] = create_token(two_char_token, NULL, line, start_column);
                (*token_count)++;
                pos += 2;
                column += 2;
                continue;
            }
        }
        
        // Single-character tokens
        TokenType single_char_token = TOKEN_ERROR;
        switch (current) {
            case '+': single_char_token = TOKEN_PLUS; break;
            case '-': single_char_token = TOKEN_MINUS; break;
            case '*': single_char_token = TOKEN_MULTIPLY; break;
            case '/': single_char_token = TOKEN_DIVIDE; break;
            case '=': single_char_token = TOKEN_ASSIGN; break;
            case '<': single_char_token = TOKEN_LESS; break;
            case '>': single_char_token = TOKEN_GREATER; break;
            case '(': single_char_token = TOKEN_LPAREN; break;
            case ')': single_char_token = TOKEN_RPAREN; break;
            case '{': single_char_token = TOKEN_LBRACE; break;
            case '}': single_char_token = TOKEN_RBRACE; break;
            case '[': single_char_token = TOKEN_LBRACKET; break;
            case ']': single_char_token = TOKEN_RBRACKET; break;
            case ';': single_char_token = TOKEN_SEMICOLON; break;
            case ',': single_char_token = TOKEN_COMMA; break;
        }
        
        if (single_char_token != TOKEN_ERROR) {
            tokens[*token_count] = create_token(single_char_token, NULL, line, start_column);
            (*token_count)++;
            pos++;
            column++;
            continue;
        }
        
        // Unknown character
        error("Unknown character '%c' at line %d, column %d", current, line, column);
        free_tokens(tokens, *token_count);
        return NULL;
    }
    
    // Add EOF token
    tokens[*token_count] = create_token(TOKEN_EOF, NULL, line, column);
    (*token_count)++;
    
    return tokens;
}

void free_tokens(Token *tokens, int count) {
    if (!tokens) return;
    
    for (int i = 0; i < count; i++) {
        free(tokens[i].value);
    }
    free(tokens);
}

void print_token(Token token) {
    const char *type_names[] = {
        [TOKEN_NUMBER] = "NUMBER",
        [TOKEN_IDENTIFIER] = "IDENTIFIER",
        [TOKEN_STRING] = "STRING",
        [TOKEN_INT] = "INT",
        [TOKEN_FLOAT] = "FLOAT",
        [TOKEN_CHAR] = "CHAR",
        [TOKEN_IF] = "IF",
        [TOKEN_ELSE] = "ELSE",
        [TOKEN_WHILE] = "WHILE",
        [TOKEN_FOR] = "FOR",
        [TOKEN_RETURN] = "RETURN",
        [TOKEN_VOID] = "VOID",
        [TOKEN_PARALLEL_FOR] = "PARALLEL_FOR",
        [TOKEN_PARALLEL_REDUCE] = "PARALLEL_REDUCE",
        [TOKEN_ATOMIC_ADD] = "ATOMIC_ADD",
        [TOKEN_BARRIER] = "BARRIER",
        [TOKEN_THREAD_ID] = "THREAD_ID",
        [TOKEN_PLUS] = "PLUS",
        [TOKEN_MINUS] = "MINUS",
        [TOKEN_MULTIPLY] = "MULTIPLY",
        [TOKEN_DIVIDE] = "DIVIDE",
        [TOKEN_ASSIGN] = "ASSIGN",
        [TOKEN_EQUAL] = "EQUAL",
        [TOKEN_NOT_EQUAL] = "NOT_EQUAL",
        [TOKEN_LESS] = "LESS",
        [TOKEN_GREATER] = "GREATER",
        [TOKEN_LESS_EQUAL] = "LESS_EQUAL",
        [TOKEN_GREATER_EQUAL] = "GREATER_EQUAL",
        [TOKEN_LPAREN] = "LPAREN",
        [TOKEN_RPAREN] = "RPAREN",
        [TOKEN_LBRACE] = "LBRACE",
        [TOKEN_RBRACE] = "RBRACE",
        [TOKEN_LBRACKET] = "LBRACKET",
        [TOKEN_RBRACKET] = "RBRACKET",
        [TOKEN_SEMICOLON] = "SEMICOLON",
        [TOKEN_COMMA] = "COMMA",
        [TOKEN_EOF] = "EOF",
        [TOKEN_ERROR] = "ERROR"
    };
    
    printf("Token: %s", type_names[token.type]);
    if (token.value) {
        printf(" ('%s')", token.value);
    }
    printf(" at line %d, column %d\n", token.line, token.column);
}
