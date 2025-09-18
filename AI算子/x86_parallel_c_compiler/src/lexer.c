/*
 * X86/X64 并行 C 编译器 - 词法分析器
 * 负责将源代码转换为标记流
 */

#include "x86_cc.h"

// =============================================================================
// 关键字映射表
// =============================================================================

typedef struct {
    char *keyword;
    TokenType token;
} KeywordMapping;

static KeywordMapping keywords[] = {
    // 基本类型
    {"int", TOKEN_INT},
    {"float", TOKEN_FLOAT_KW},
    {"double", TOKEN_DOUBLE},
    {"char", TOKEN_CHAR_KW},
    {"void", TOKEN_VOID},
    
    // 控制流
    {"if", TOKEN_IF},
    {"else", TOKEN_ELSE},
    {"for", TOKEN_FOR},
    {"while", TOKEN_WHILE},
    {"return", TOKEN_RETURN},
    
    // 结构体
    {"struct", TOKEN_STRUCT},
    {"sizeof", TOKEN_SIZEOF},
    
    // 并行计算扩展
    {"parallel_for", TOKEN_PARALLEL_FOR},
    {"atomic", TOKEN_ATOMIC},
    {"barrier", TOKEN_BARRIER},
    {"critical", TOKEN_CRITICAL},
    {"thread_local", TOKEN_THREAD_LOCAL},
    
    {NULL, TOKEN_ERROR}
};

// =============================================================================
// 词法分析器实现
// =============================================================================

Lexer *lexer_create(char *source) {
    Lexer *lexer = (Lexer *)safe_malloc(sizeof(Lexer));
    lexer->source = safe_strdup(source);
    lexer->current = lexer->source;
    lexer->line = 1;
    lexer->column = 1;
    lexer->has_error = false;
    
    // 读取第一个标记
    lexer_next_token(lexer);
    
    return lexer;
}

void lexer_destroy(Lexer *lexer) {
    if (lexer) {
        free(lexer->source);
        if (lexer->current_token.value) {
            free(lexer->current_token.value);
        }
        free(lexer);
    }
}

void lexer_error(Lexer *lexer, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(lexer->error_msg, MAX_ERROR_MSG, format, args);
    va_end(args);
    lexer->has_error = true;
}

static char lexer_peek(Lexer *lexer) {
    return *lexer->current;
}

static char lexer_peek_next(Lexer *lexer) {
    if (*lexer->current == '\0') return '\0';
    return *(lexer->current + 1);
}

static char lexer_advance(Lexer *lexer) {
    char c = *lexer->current;
    if (c != '\0') {
        lexer->current++;
        if (c == '\n') {
            lexer->line++;
            lexer->column = 1;
        } else {
            lexer->column++;
        }
    }
    return c;
}

static void lexer_skip_whitespace(Lexer *lexer) {
    while (isspace(lexer_peek(lexer)) && lexer_peek(lexer) != '\n') {
        lexer_advance(lexer);
    }
}

static void lexer_skip_line_comment(Lexer *lexer) {
    // 跳过 // 注释
    while (lexer_peek(lexer) != '\n' && lexer_peek(lexer) != '\0') {
        lexer_advance(lexer);
    }
}

static void lexer_skip_block_comment(Lexer *lexer) {
    // 跳过 /* */ 注释
    lexer_advance(lexer); // 跳过 *
    lexer_advance(lexer); // 跳过第二个 *
    
    while (lexer_peek(lexer) != '\0') {
        if (lexer_peek(lexer) == '*' && lexer_peek_next(lexer) == '/') {
            lexer_advance(lexer); // 跳过 *
            lexer_advance(lexer); // 跳过 /
            break;
        }
        lexer_advance(lexer);
    }
}

static TokenType lexer_lookup_keyword(char *identifier) {
    for (int i = 0; keywords[i].keyword != NULL; i++) {
        if (strcmp(identifier, keywords[i].keyword) == 0) {
            return keywords[i].token;
        }
    }
    return TOKEN_IDENTIFIER;
}

static void lexer_read_string(Lexer *lexer, Token *token) {
    char buffer[MAX_TOKEN_LENGTH];
    int pos = 0;
    
    lexer_advance(lexer); // 跳过开始的引号
    
    while (lexer_peek(lexer) != '"' && lexer_peek(lexer) != '\0' && pos < MAX_TOKEN_LENGTH - 1) {
        char c = lexer_peek(lexer);
        if (c == '\\') {
            lexer_advance(lexer);
            char escaped = lexer_peek(lexer);
            switch (escaped) {
                case 'n': buffer[pos++] = '\n'; break;
                case 't': buffer[pos++] = '\t'; break;
                case 'r': buffer[pos++] = '\r'; break;
                case '\\': buffer[pos++] = '\\'; break;
                case '"': buffer[pos++] = '"'; break;
                case '0': buffer[pos++] = '\0'; break;
                default: buffer[pos++] = escaped; break;
            }
            lexer_advance(lexer);
        } else {
            buffer[pos++] = lexer_advance(lexer);
        }
    }
    
    if (lexer_peek(lexer) == '"') {
        lexer_advance(lexer); // 跳过结束的引号
    } else {
        lexer_error(lexer, "Unterminated string literal");
        token->type = TOKEN_ERROR;
        return;
    }
    
    buffer[pos] = '\0';
    token->type = TOKEN_STRING;
    token->value = safe_strdup(buffer);
}

static void lexer_read_char(Lexer *lexer, Token *token) {
    lexer_advance(lexer); // 跳过开始的单引号
    
    if (lexer_peek(lexer) == '\0') {
        lexer_error(lexer, "Unterminated character literal");
        token->type = TOKEN_ERROR;
        return;
    }
    
    char c = lexer_peek(lexer);
    if (c == '\\') {
        lexer_advance(lexer);
        char escaped = lexer_peek(lexer);
        switch (escaped) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case '\\': c = '\\'; break;
            case '\'': c = '\''; break;
            case '0': c = '\0'; break;
            default: c = escaped; break;
        }
        lexer_advance(lexer);
    } else {
        lexer_advance(lexer);
    }
    
    if (lexer_peek(lexer) != '\'') {
        lexer_error(lexer, "Expected closing quote for character literal");
        token->type = TOKEN_ERROR;
        return;
    }
    
    lexer_advance(lexer); // 跳过结束的单引号
    
    token->type = TOKEN_CHAR;
    token->char_val = c;
    token->value = safe_malloc(2);
    token->value[0] = c;
    token->value[1] = '\0';
}

static void lexer_read_number(Lexer *lexer, Token *token) {
    char buffer[MAX_TOKEN_LENGTH];
    int pos = 0;
    bool is_float = false;
    
    // 读取数字字符
    while ((isdigit(lexer_peek(lexer)) || lexer_peek(lexer) == '.') && pos < MAX_TOKEN_LENGTH - 1) {
        char c = lexer_peek(lexer);
        if (c == '.') {
            if (is_float) break; // 已经有小数点了
            is_float = true;
        }
        buffer[pos++] = lexer_advance(lexer);
    }
    
    // 检查科学记数法 (e/E)
    if (lexer_peek(lexer) == 'e' || lexer_peek(lexer) == 'E') {
        is_float = true;
        buffer[pos++] = lexer_advance(lexer);
        if (lexer_peek(lexer) == '+' || lexer_peek(lexer) == '-') {
            buffer[pos++] = lexer_advance(lexer);
        }
        while (isdigit(lexer_peek(lexer)) && pos < MAX_TOKEN_LENGTH - 1) {
            buffer[pos++] = lexer_advance(lexer);
        }
    }
    
    // 检查浮点数后缀 (f/F)
    if (lexer_peek(lexer) == 'f' || lexer_peek(lexer) == 'F') {
        is_float = true;
        lexer_advance(lexer);
    }
    
    buffer[pos] = '\0';
    token->value = safe_strdup(buffer);
    
    if (is_float) {
        token->type = TOKEN_FLOAT;
        token->float_val = strtod(buffer, NULL);
    } else {
        token->type = TOKEN_NUMBER;
        token->int_val = strtol(buffer, NULL, 10);
    }
}

static void lexer_read_identifier(Lexer *lexer, Token *token) {
    char buffer[MAX_TOKEN_LENGTH];
    int pos = 0;
    
    while ((isalnum(lexer_peek(lexer)) || lexer_peek(lexer) == '_') && pos < MAX_TOKEN_LENGTH - 1) {
        buffer[pos++] = lexer_advance(lexer);
    }
    
    buffer[pos] = '\0';
    token->value = safe_strdup(buffer);
    token->type = lexer_lookup_keyword(buffer);
}

bool lexer_next_token(Lexer *lexer) {
    if (lexer->current_token.value) {
        free(lexer->current_token.value);
        lexer->current_token.value = NULL;
    }
    
    // 跳过空白字符和注释
    while (true) {
        lexer_skip_whitespace(lexer);
        
        if (lexer_peek(lexer) == '/' && lexer_peek_next(lexer) == '/') {
            lexer_skip_line_comment(lexer);
        } else if (lexer_peek(lexer) == '/' && lexer_peek_next(lexer) == '*') {
            lexer_skip_block_comment(lexer);
        } else {
            break;
        }
    }
    
    Token *token = &lexer->current_token;
    token->line = lexer->line;
    token->column = lexer->column;
    token->value = NULL;
    
    char c = lexer_peek(lexer);
    
    // 文件结束
    if (c == '\0') {
        token->type = TOKEN_EOF;
        return true;
    }
    
    // 换行
    if (c == '\n') {
        token->type = TOKEN_NEWLINE;
        lexer_advance(lexer);
        return true;
    }
    
    // 字符串字面量
    if (c == '"') {
        lexer_read_string(lexer, token);
        return !lexer->has_error;
    }
    
    // 字符字面量
    if (c == '\'') {
        lexer_read_char(lexer, token);
        return !lexer->has_error;
    }
    
    // 数字
    if (isdigit(c)) {
        lexer_read_number(lexer, token);
        return true;
    }
    
    // 标识符和关键字
    if (isalpha(c) || c == '_') {
        lexer_read_identifier(lexer, token);
        return true;
    }
    
    // 双字符运算符
    char next = lexer_peek_next(lexer);
    if (c == '+' && next == '+') {
        token->type = TOKEN_INCREMENT;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '-' && next == '-') {
        token->type = TOKEN_DECREMENT;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '+' && next == '=') {
        token->type = TOKEN_PLUS_ASSIGN;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '-' && next == '=') {
        token->type = TOKEN_MINUS_ASSIGN;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '*' && next == '=') {
        token->type = TOKEN_MUL_ASSIGN;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '/' && next == '=') {
        token->type = TOKEN_DIV_ASSIGN;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '=' && next == '=') {
        token->type = TOKEN_EQ;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '!' && next == '=') {
        token->type = TOKEN_NE;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '<' && next == '=') {
        token->type = TOKEN_LE;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '>' && next == '=') {
        token->type = TOKEN_GE;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '<' && next == '<') {
        token->type = TOKEN_LSHIFT;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '>' && next == '>') {
        token->type = TOKEN_RSHIFT;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '&' && next == '&') {
        token->type = TOKEN_AND;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '|' && next == '|') {
        token->type = TOKEN_OR;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    if (c == '-' && next == '>') {
        token->type = TOKEN_ARROW;
        lexer_advance(lexer);
        lexer_advance(lexer);
        return true;
    }
    
    // 单字符运算符和标点符号
    lexer_advance(lexer);
    switch (c) {
        case '+': token->type = TOKEN_PLUS; break;
        case '-': token->type = TOKEN_MINUS; break;
        case '*': token->type = TOKEN_MULTIPLY; break;
        case '/': token->type = TOKEN_DIVIDE; break;
        case '%': token->type = TOKEN_MODULO; break;
        case '=': token->type = TOKEN_ASSIGN; break;
        case '<': token->type = TOKEN_LT; break;
        case '>': token->type = TOKEN_GT; break;
        case '!': token->type = TOKEN_NOT; break;
        case '&': token->type = TOKEN_BIT_AND; break;
        case '|': token->type = TOKEN_BIT_OR; break;
        case '^': token->type = TOKEN_BIT_XOR; break;
        case '~': token->type = TOKEN_BIT_NOT; break;
        case ';': token->type = TOKEN_SEMICOLON; break;
        case ',': token->type = TOKEN_COMMA; break;
        case '.': token->type = TOKEN_DOT; break;
        case '(': token->type = TOKEN_LPAREN; break;
        case ')': token->type = TOKEN_RPAREN; break;
        case '{': token->type = TOKEN_LBRACE; break;
        case '}': token->type = TOKEN_RBRACE; break;
        case '[': token->type = TOKEN_LBRACKET; break;
        case ']': token->type = TOKEN_RBRACKET; break;
        case '?': token->type = TOKEN_QUESTION; break;
        case ':': token->type = TOKEN_COLON; break;
        default:
            lexer_error(lexer, "Unexpected character: '%c'", c);
            token->type = TOKEN_ERROR;
            return false;
    }
    
    return true;
}
