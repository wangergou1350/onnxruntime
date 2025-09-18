/*
 * ParallelC++ Compiler - Lexical Analyzer
 * 
 * This module implements lexical analysis for C++ with parallel extensions.
 * It recognizes C++ keywords, operators, and parallel computing primitives.
 */

#include "pcpp.h"

// C++ keyword mapping
typedef struct {
    const char *keyword;
    TokenType token_type;
} KeywordMap;

static KeywordMap cpp_keywords[] = {
    // C++ specific keywords
    {"class", TOKEN_CLASS},
    {"public", TOKEN_PUBLIC},
    {"private", TOKEN_PRIVATE},
    {"protected", TOKEN_PROTECTED},
    {"virtual", TOKEN_VIRTUAL},
    {"static", TOKEN_STATIC},
    {"const", TOKEN_CONST},
    {"inline", TOKEN_INLINE},
    {"friend", TOKEN_FRIEND},
    {"namespace", TOKEN_NAMESPACE},
    {"using", TOKEN_USING},
    {"template", TOKEN_TEMPLATE},
    {"typename", TOKEN_TYPENAME},
    {"this", TOKEN_THIS},
    {"new", TOKEN_NEW},
    {"delete", TOKEN_DELETE},
    {"operator", TOKEN_OPERATOR},
    {"override", TOKEN_OVERRIDE},
    {"final", TOKEN_FINAL},
    
    // C keywords
    {"int", TOKEN_INT},
    {"float", TOKEN_FLOAT},
    {"double", TOKEN_DOUBLE},
    {"char", TOKEN_CHAR_TYPE},
    {"void", TOKEN_VOID},
    {"bool", TOKEN_BOOL},
    {"if", TOKEN_IF},
    {"else", TOKEN_ELSE},
    {"while", TOKEN_WHILE},
    {"for", TOKEN_FOR},
    {"do", TOKEN_DO},
    {"switch", TOKEN_SWITCH},
    {"case", TOKEN_CASE},
    {"default", TOKEN_DEFAULT},
    {"break", TOKEN_BREAK},
    {"continue", TOKEN_CONTINUE},
    {"return", TOKEN_RETURN},
    {"sizeof", TOKEN_SIZEOF},
    {"typedef", TOKEN_TYPEDEF},
    {"struct", TOKEN_STRUCT},
    {"union", TOKEN_UNION},
    {"enum", TOKEN_ENUM},
    
    // Parallel computing extensions
    {"parallel_for", TOKEN_PARALLEL_FOR},
    {"parallel_invoke", TOKEN_PARALLEL_INVOKE},
    {"parallel_class", TOKEN_PARALLEL_CLASS},
    {"thread_safe", TOKEN_THREAD_SAFE},
    {"atomic_add", TOKEN_ATOMIC_ADD},
    {"atomic_sub", TOKEN_ATOMIC_SUB},
    {"barrier", TOKEN_BARRIER},
    {"thread_id", TOKEN_THREAD_ID},
    {"num_threads", TOKEN_NUM_THREADS},
    {"lambda", TOKEN_LAMBDA},
    
    {NULL, TOKEN_EOF} // Sentinel
};

static TokenType lookup_keyword(const char *word) {
    for (int i = 0; cpp_keywords[i].keyword != NULL; i++) {
        if (strcmp(word, cpp_keywords[i].keyword) == 0) {
            return cpp_keywords[i].token_type;
        }
    }
    return TOKEN_IDENTIFIER;
}

static bool is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

static bool is_alnum(char c) {
    return is_alpha(c) || is_digit(c);
}

static bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\r';
}

static Token create_token(TokenType type, const char *value, int line, int column) {
    Token token;
    token.type = type;
    strncpy(token.value, value, MAX_TOKEN_LENGTH - 1);
    token.value[MAX_TOKEN_LENGTH - 1] = '\0';
    token.line = line;
    token.column = column;
    return token;
}

static int scan_identifier(const char *source, int pos, char *buffer) {
    int start = pos;
    int buf_pos = 0;
    
    while (source[pos] && is_alnum(source[pos]) && buf_pos < MAX_TOKEN_LENGTH - 1) {
        buffer[buf_pos++] = source[pos++];
    }
    buffer[buf_pos] = '\0';
    
    return pos - start;
}

static int scan_number(const char *source, int pos, char *buffer) {
    int start = pos;
    int buf_pos = 0;
    bool has_dot = false;
    
    while (source[pos] && buf_pos < MAX_TOKEN_LENGTH - 1) {
        if (is_digit(source[pos])) {
            buffer[buf_pos++] = source[pos++];
        } else if (source[pos] == '.' && !has_dot) {
            has_dot = true;
            buffer[buf_pos++] = source[pos++];
        } else if (source[pos] == 'f' || source[pos] == 'F') {
            // Float suffix
            buffer[buf_pos++] = source[pos++];
            break;
        } else if (source[pos] == 'l' || source[pos] == 'L') {
            // Long suffix
            buffer[buf_pos++] = source[pos++];
            break;
        } else {
            break;
        }
    }
    
    buffer[buf_pos] = '\0';
    return pos - start;
}

static int scan_string(const char *source, int pos, char *buffer) {
    int start = pos;
    int buf_pos = 0;
    char quote = source[pos++]; // Skip opening quote
    
    while (source[pos] && source[pos] != quote && buf_pos < MAX_TOKEN_LENGTH - 1) {
        if (source[pos] == '\\' && source[pos + 1]) {
            // Handle escape sequences
            pos++; // Skip backslash
            switch (source[pos]) {
                case 'n': buffer[buf_pos++] = '\n'; break;
                case 't': buffer[buf_pos++] = '\t'; break;
                case 'r': buffer[buf_pos++] = '\r'; break;
                case '\\': buffer[buf_pos++] = '\\'; break;
                case '"': buffer[buf_pos++] = '"'; break;
                case '\'': buffer[buf_pos++] = '\''; break;
                case '0': buffer[buf_pos++] = '\0'; break;
                default: 
                    buffer[buf_pos++] = '\\';
                    buffer[buf_pos++] = source[pos];
                    break;
            }
            pos++;
        } else {
            buffer[buf_pos++] = source[pos++];
        }
    }
    
    if (source[pos] == quote) {
        pos++; // Skip closing quote
    }
    
    buffer[buf_pos] = '\0';
    return pos - start;
}

static int scan_char(const char *source, int pos, char *buffer) {
    int start = pos;
    int buf_pos = 0;
    
    pos++; // Skip opening quote
    
    if (source[pos] == '\\' && source[pos + 1]) {
        // Handle escape sequences
        pos++; // Skip backslash
        switch (source[pos]) {
            case 'n': buffer[buf_pos++] = '\n'; break;
            case 't': buffer[buf_pos++] = '\t'; break;
            case 'r': buffer[buf_pos++] = '\r'; break;
            case '\\': buffer[buf_pos++] = '\\'; break;
            case '\'': buffer[buf_pos++] = '\''; break;
            case '0': buffer[buf_pos++] = '\0'; break;
            default: 
                buffer[buf_pos++] = source[pos];
                break;
        }
        pos++;
    } else if (source[pos] != '\'') {
        buffer[buf_pos++] = source[pos++];
    }
    
    if (source[pos] == '\'') {
        pos++; // Skip closing quote
    }
    
    buffer[buf_pos] = '\0';
    return pos - start;
}

static int skip_comment(const char *source, int pos) {
    if (source[pos] == '/' && source[pos + 1] == '/') {
        // Single-line comment
        pos += 2;
        while (source[pos] && source[pos] != '\n') {
            pos++;
        }
        return pos;
    } else if (source[pos] == '/' && source[pos + 1] == '*') {
        // Multi-line comment
        pos += 2;
        while (source[pos] && !(source[pos] == '*' && source[pos + 1] == '/')) {
            pos++;
        }
        if (source[pos] == '*' && source[pos + 1] == '/') {
            pos += 2;
        }
        return pos;
    }
    return pos;
}

Token *tokenize(const char *source, int *token_count) {
    if (!source || !token_count) {
        return NULL;
    }
    
    int capacity = 1000;
    Token *tokens = malloc(capacity * sizeof(Token));
    if (!tokens) {
        error("Memory allocation failed");
        return NULL;
    }
    
    int pos = 0;
    int count = 0;
    int line = 1;
    int column = 1;
    int line_start = 0;
    
    while (source[pos]) {
        // Resize if needed
        if (count >= capacity - 1) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(Token));
            if (!tokens) {
                error("Memory reallocation failed");
                return NULL;
            }
        }
        
        char current = source[pos];
        column = pos - line_start + 1;
        
        // Skip whitespace
        if (is_whitespace(current)) {
            pos++;
            continue;
        }
        
        // Handle newlines
        if (current == '\n') {
            line++;
            line_start = pos + 1;
            pos++;
            continue;
        }
        
        // Handle comments
        if (current == '/' && (source[pos + 1] == '/' || source[pos + 1] == '*')) {
            int new_pos = skip_comment(source, pos);
            // Count newlines in multi-line comments
            for (int i = pos; i < new_pos; i++) {
                if (source[i] == '\n') {
                    line++;
                    line_start = i + 1;
                }
            }
            pos = new_pos;
            continue;
        }
        
        char buffer[MAX_TOKEN_LENGTH];
        int advance = 0;
        TokenType token_type = TOKEN_EOF;
        
        // Identifiers and keywords
        if (is_alpha(current)) {
            advance = scan_identifier(source, pos, buffer);
            token_type = lookup_keyword(buffer);
        }
        // Numbers
        else if (is_digit(current)) {
            advance = scan_number(source, pos, buffer);
            token_type = TOKEN_NUMBER;
        }
        // Strings
        else if (current == '"') {
            advance = scan_string(source, pos, buffer);
            token_type = TOKEN_STRING;
        }
        // Character literals
        else if (current == '\'') {
            advance = scan_char(source, pos, buffer);
            token_type = TOKEN_CHAR;
        }
        // Two-character operators
        else if (current == '+' && source[pos + 1] == '+') {
            strcpy(buffer, "++");
            advance = 2;
            token_type = TOKEN_INCREMENT;
        }
        else if (current == '-' && source[pos + 1] == '-') {
            strcpy(buffer, "--");
            advance = 2;
            token_type = TOKEN_DECREMENT;
        }
        else if (current == '+' && source[pos + 1] == '=') {
            strcpy(buffer, "+=");
            advance = 2;
            token_type = TOKEN_PLUS_ASSIGN;
        }
        else if (current == '-' && source[pos + 1] == '=') {
            strcpy(buffer, "-=");
            advance = 2;
            token_type = TOKEN_MINUS_ASSIGN;
        }
        else if (current == '*' && source[pos + 1] == '=') {
            strcpy(buffer, "*=");
            advance = 2;
            token_type = TOKEN_MULTIPLY_ASSIGN;
        }
        else if (current == '/' && source[pos + 1] == '=') {
            strcpy(buffer, "/=");
            advance = 2;
            token_type = TOKEN_DIVIDE_ASSIGN;
        }
        else if (current == '=' && source[pos + 1] == '=') {
            strcpy(buffer, "==");
            advance = 2;
            token_type = TOKEN_EQUAL;
        }
        else if (current == '!' && source[pos + 1] == '=') {
            strcpy(buffer, "!=");
            advance = 2;
            token_type = TOKEN_NOT_EQUAL;
        }
        else if (current == '<' && source[pos + 1] == '=') {
            strcpy(buffer, "<=");
            advance = 2;
            token_type = TOKEN_LESS_EQUAL;
        }
        else if (current == '>' && source[pos + 1] == '=') {
            strcpy(buffer, ">=");
            advance = 2;
            token_type = TOKEN_GREATER_EQUAL;
        }
        else if (current == '&' && source[pos + 1] == '&') {
            strcpy(buffer, "&&");
            advance = 2;
            token_type = TOKEN_LOGICAL_AND;
        }
        else if (current == '|' && source[pos + 1] == '|') {
            strcpy(buffer, "||");
            advance = 2;
            token_type = TOKEN_LOGICAL_OR;
        }
        else if (current == '<' && source[pos + 1] == '<') {
            strcpy(buffer, "<<");
            advance = 2;
            token_type = TOKEN_LEFT_SHIFT;
        }
        else if (current == '>' && source[pos + 1] == '>') {
            strcpy(buffer, ">>");
            advance = 2;
            token_type = TOKEN_RIGHT_SHIFT;
        }
        else if (current == '-' && source[pos + 1] == '>') {
            strcpy(buffer, "->");
            advance = 2;
            token_type = TOKEN_ARROW;
        }
        else if (current == ':' && source[pos + 1] == ':') {
            strcpy(buffer, "::");
            advance = 2;
            token_type = TOKEN_SCOPE_RESOLUTION;
        }
        // Single-character operators and punctuation
        else {
            buffer[0] = current;
            buffer[1] = '\0';
            advance = 1;
            
            switch (current) {
                case '+': token_type = TOKEN_PLUS; break;
                case '-': token_type = TOKEN_MINUS; break;
                case '*': token_type = TOKEN_MULTIPLY; break;
                case '/': token_type = TOKEN_DIVIDE; break;
                case '%': token_type = TOKEN_MODULO; break;
                case '=': token_type = TOKEN_ASSIGN; break;
                case '<': token_type = TOKEN_LESS; break;
                case '>': token_type = TOKEN_GREATER; break;
                case '!': token_type = TOKEN_NOT; break;
                case '&': token_type = TOKEN_BITWISE_AND; break;
                case '|': token_type = TOKEN_BITWISE_OR; break;
                case '^': token_type = TOKEN_BITWISE_XOR; break;
                case '~': token_type = TOKEN_BITWISE_NOT; break;
                case ';': token_type = TOKEN_SEMICOLON; break;
                case ',': token_type = TOKEN_COMMA; break;
                case '.': token_type = TOKEN_DOT; break;
                case '?': token_type = TOKEN_QUESTION; break;
                case ':': token_type = TOKEN_COLON; break;
                case '(': token_type = TOKEN_LPAREN; break;
                case ')': token_type = TOKEN_RPAREN; break;
                case '{': token_type = TOKEN_LBRACE; break;
                case '}': token_type = TOKEN_RBRACE; break;
                case '[': token_type = TOKEN_LBRACKET; break;
                case ']': token_type = TOKEN_RBRACKET; break;
                default:
                    error("Unknown character '%c' at line %d, column %d", current, line, column);
                    advance = 1; // Skip unknown character
                    continue;
            }
        }
        
        if (token_type != TOKEN_EOF) {
            tokens[count++] = create_token(token_type, buffer, line, column);
        }
        
        pos += advance;
    }
    
    // Add EOF token
    tokens[count++] = create_token(TOKEN_EOF, "", line, column);
    
    *token_count = count;
    return tokens;
}

void free_tokens(Token *tokens, int count) {
    if (tokens) {
        free(tokens);
    }
}

void print_tokens(Token *tokens, int count) {
    printf("=== TOKENS ===\n");
    for (int i = 0; i < count; i++) {
        printf("%-3d: %-20s '%s' (line %d, col %d)\n", 
               i, 
               // Convert token type to string (simplified)
               tokens[i].type == TOKEN_IDENTIFIER ? "IDENTIFIER" :
               tokens[i].type == TOKEN_NUMBER ? "NUMBER" :
               tokens[i].type == TOKEN_STRING ? "STRING" :
               tokens[i].type == TOKEN_CLASS ? "CLASS" :
               tokens[i].type == TOKEN_PUBLIC ? "PUBLIC" :
               tokens[i].type == TOKEN_PRIVATE ? "PRIVATE" :
               tokens[i].type == TOKEN_PARALLEL_FOR ? "PARALLEL_FOR" :
               tokens[i].type == TOKEN_EOF ? "EOF" : "OTHER",
               tokens[i].value,
               tokens[i].line,
               tokens[i].column);
    }
    printf("=== END TOKENS ===\n\n");
}
