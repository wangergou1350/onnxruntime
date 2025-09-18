#include "riscv_cc.h"

// =============================================================================
// 词法分析器实现
// =============================================================================

// 关键字映射表
typedef struct {
    const char *keyword;
    TokenType token_type;
} KeywordMapping;

static KeywordMapping keywords[] = {
    // C语言标准关键字
    {"auto", TOKEN_AUTO},
    {"break", TOKEN_BREAK},
    {"case", TOKEN_CASE},
    {"char", TOKEN_CHAR},
    {"const", TOKEN_CONST},
    {"continue", TOKEN_CONTINUE},
    {"default", TOKEN_DEFAULT},
    {"do", TOKEN_DO},
    {"double", TOKEN_DOUBLE},
    {"else", TOKEN_ELSE},
    {"enum", TOKEN_ENUM},
    {"extern", TOKEN_EXTERN},
    {"float", TOKEN_FLOAT_KW},
    {"for", TOKEN_FOR},
    {"goto", TOKEN_GOTO},
    {"if", TOKEN_IF},
    {"int", TOKEN_INT},
    {"long", TOKEN_LONG},
    {"register", TOKEN_REGISTER},
    {"return", TOKEN_RETURN},
    {"short", TOKEN_SHORT},
    {"signed", TOKEN_SIGNED},
    {"sizeof", TOKEN_SIZEOF},
    {"static", TOKEN_STATIC},
    {"struct", TOKEN_STRUCT},
    {"switch", TOKEN_SWITCH},
    {"typedef", TOKEN_TYPEDEF},
    {"union", TOKEN_UNION},
    {"unsigned", TOKEN_UNSIGNED},
    {"void", TOKEN_VOID},
    {"volatile", TOKEN_VOLATILE},
    {"while", TOKEN_WHILE},
    
    // 并行计算扩展关键字
    {"parallel_for", TOKEN_PARALLEL_FOR},
    {"atomic", TOKEN_ATOMIC},
    {"thread_local", TOKEN_THREAD_LOCAL},
    {"barrier", TOKEN_BARRIER},
    {"critical", TOKEN_CRITICAL},
    
    {NULL, TOKEN_ERROR}  // 结束标记
};

Lexer *create_lexer(const char *source) {
    Lexer *lexer = malloc(sizeof(Lexer));
    if (!lexer) {
        error("内存分配失败：无法创建词法分析器");
        return NULL;
    }
    
    lexer->source = strdup(source);
    lexer->position = 0;
    lexer->line = 1;
    lexer->column = 1;
    lexer->length = strlen(source);
    
    return lexer;
}

void destroy_lexer(Lexer *lexer) {
    if (lexer) {
        free(lexer->source);
        free(lexer);
    }
}

static char current_char(Lexer *lexer) {
    if (lexer->position >= lexer->length) {
        return '\0';
    }
    return lexer->source[lexer->position];
}

static char peek_char(Lexer *lexer, int offset) {
    int pos = lexer->position + offset;
    if (pos >= lexer->length) {
        return '\0';
    }
    return lexer->source[pos];
}

static void advance_char(Lexer *lexer) {
    if (lexer->position < lexer->length) {
        if (lexer->source[lexer->position] == '\n') {
            lexer->line++;
            lexer->column = 1;
        } else {
            lexer->column++;
        }
        lexer->position++;
    }
}

static void skip_whitespace(Lexer *lexer) {
    char c = current_char(lexer);
    while (c == ' ' || c == '\t' || c == '\r') {
        advance_char(lexer);
        c = current_char(lexer);
    }
}

static void skip_line_comment(Lexer *lexer) {
    advance_char(lexer); // 跳过第一个 '/'
    advance_char(lexer); // 跳过第二个 '/'
    
    char c = current_char(lexer);
    while (c != '\0' && c != '\n') {
        advance_char(lexer);
        c = current_char(lexer);
    }
}

static void skip_block_comment(Lexer *lexer) {
    advance_char(lexer); // 跳过 '/'
    advance_char(lexer); // 跳过 '*'
    
    while (current_char(lexer) != '\0') {
        if (current_char(lexer) == '*' && peek_char(lexer, 1) == '/') {
            advance_char(lexer); // 跳过 '*'
            advance_char(lexer); // 跳过 '/'
            break;
        }
        advance_char(lexer);
    }
}

static TokenType lookup_keyword(const char *identifier) {
    for (int i = 0; keywords[i].keyword != NULL; i++) {
        if (strcmp(identifier, keywords[i].keyword) == 0) {
            return keywords[i].token_type;
        }
    }
    return TOKEN_IDENTIFIER;
}

static Token read_identifier(Lexer *lexer) {
    Token token;
    token.line = lexer->line;
    token.column = lexer->column;
    
    char buffer[MAX_IDENTIFIER_LENGTH];
    int index = 0;
    
    char c = current_char(lexer);
    while ((isalnum(c) || c == '_') && index < MAX_IDENTIFIER_LENGTH - 1) {
        buffer[index++] = c;
        advance_char(lexer);
        c = current_char(lexer);
    }
    
    buffer[index] = '\0';
    token.value = strdup(buffer);
    token.type = lookup_keyword(buffer);
    
    return token;
}

static Token read_number(Lexer *lexer) {
    Token token;
    token.line = lexer->line;
    token.column = lexer->column;
    
    char buffer[64];
    int index = 0;
    bool is_float = false;
    bool is_hex = false;
    bool is_octal = false;
    
    char c = current_char(lexer);
    
    // 检查十六进制前缀
    if (c == '0' && (peek_char(lexer, 1) == 'x' || peek_char(lexer, 1) == 'X')) {
        is_hex = true;
        buffer[index++] = c; advance_char(lexer);
        buffer[index++] = current_char(lexer); advance_char(lexer);
        c = current_char(lexer);
    }
    // 检查八进制前缀
    else if (c == '0' && isdigit(peek_char(lexer, 1))) {
        is_octal = true;
    }
    
    // 读取数字部分
    while (index < 63) {
        c = current_char(lexer);
        if (is_hex && isxdigit(c)) {
            buffer[index++] = c;
            advance_char(lexer);
        } else if (!is_hex && isdigit(c)) {
            buffer[index++] = c;
            advance_char(lexer);
        } else if (c == '.' && !is_float && !is_hex && !is_octal) {
            is_float = true;
            buffer[index++] = c;
            advance_char(lexer);
        } else if ((c == 'e' || c == 'E') && !is_hex && !is_octal) {
            is_float = true;
            buffer[index++] = c;
            advance_char(lexer);
            // 检查指数符号
            c = current_char(lexer);
            if (c == '+' || c == '-') {
                buffer[index++] = c;
                advance_char(lexer);
            }
        } else {
            break;
        }
    }
    
    buffer[index] = '\0';
    token.value = strdup(buffer);
    
    if (is_float) {
        token.type = TOKEN_FLOAT;
        token.float_value = strtof(buffer, NULL);
    } else {
        token.type = TOKEN_NUMBER;
        if (is_hex) {
            token.int_value = (int)strtol(buffer, NULL, 16);
        } else if (is_octal) {
            token.int_value = (int)strtol(buffer, NULL, 8);
        } else {
            token.int_value = atoi(buffer);
        }
    }
    
    return token;
}

static Token read_character(Lexer *lexer) {
    Token token;
    token.line = lexer->line;
    token.column = lexer->column;
    token.type = TOKEN_CHARACTER;
    
    advance_char(lexer); // 跳过开始的单引号
    
    char c = current_char(lexer);
    if (c == '\\') {
        advance_char(lexer);
        char escape = current_char(lexer);
        switch (escape) {
            case 'n': token.char_value = '\n'; break;
            case 't': token.char_value = '\t'; break;
            case 'r': token.char_value = '\r'; break;
            case 'b': token.char_value = '\b'; break;
            case 'f': token.char_value = '\f'; break;
            case 'a': token.char_value = '\a'; break;
            case 'v': token.char_value = '\v'; break;
            case '0': token.char_value = '\0'; break;
            case '\\': token.char_value = '\\'; break;
            case '\'': token.char_value = '\''; break;
            case '\"': token.char_value = '\"'; break;
            default:
                token.char_value = escape;
                warning("未知的转义字符: \\%c", escape);
                break;
        }
        advance_char(lexer);
    } else {
        token.char_value = c;
        advance_char(lexer);
    }
    
    // 跳过结束的单引号
    if (current_char(lexer) == '\'') {
        advance_char(lexer);
    } else {
        error("字符常量缺少结束的单引号");
    }
    
    char buffer[4];
    sprintf(buffer, "'%c'", token.char_value);
    token.value = strdup(buffer);
    
    return token;
}

static Token read_string(Lexer *lexer) {
    Token token;
    token.line = lexer->line;
    token.column = lexer->column;
    token.type = TOKEN_STRING;
    
    advance_char(lexer); // 跳过开始的双引号
    
    char buffer[MAX_STRING_LENGTH];
    int index = 0;
    
    char c = current_char(lexer);
    while (c != '\0' && c != '\"' && index < MAX_STRING_LENGTH - 1) {
        if (c == '\\') {
            advance_char(lexer);
            char escape = current_char(lexer);
            switch (escape) {
                case 'n': buffer[index++] = '\n'; break;
                case 't': buffer[index++] = '\t'; break;
                case 'r': buffer[index++] = '\r'; break;
                case 'b': buffer[index++] = '\b'; break;
                case 'f': buffer[index++] = '\f'; break;
                case 'a': buffer[index++] = '\a'; break;
                case 'v': buffer[index++] = '\v'; break;
                case '0': buffer[index++] = '\0'; break;
                case '\\': buffer[index++] = '\\'; break;
                case '\'': buffer[index++] = '\''; break;
                case '\"': buffer[index++] = '\"'; break;
                default:
                    buffer[index++] = escape;
                    warning("未知的转义字符: \\%c", escape);
                    break;
            }
        } else {
            buffer[index++] = c;
        }
        advance_char(lexer);
        c = current_char(lexer);
    }
    
    buffer[index] = '\0';
    
    // 跳过结束的双引号
    if (current_char(lexer) == '\"') {
        advance_char(lexer);
    } else {
        error("字符串常量缺少结束的双引号");
    }
    
    // 创建带引号的字符串表示
    char *quoted_string = malloc(strlen(buffer) + 3);
    sprintf(quoted_string, "\"%s\"", buffer);
    token.value = quoted_string;
    
    return token;
}

Token next_token(Lexer *lexer) {
    Token token;
    
    // 跳过空白字符
    skip_whitespace(lexer);
    
    char c = current_char(lexer);
    
    // 处理文件结束
    if (c == '\0') {
        token.type = TOKEN_EOF;
        token.value = strdup("EOF");
        token.line = lexer->line;
        token.column = lexer->column;
        return token;
    }
    
    // 处理换行
    if (c == '\n') {
        token.type = TOKEN_NEWLINE;
        token.value = strdup("\n");
        token.line = lexer->line;
        token.column = lexer->column;
        advance_char(lexer);
        return token;
    }
    
    // 处理注释
    if (c == '/' && peek_char(lexer, 1) == '/') {
        skip_line_comment(lexer);
        return next_token(lexer); // 递归获取下一个标记
    }
    
    if (c == '/' && peek_char(lexer, 1) == '*') {
        skip_block_comment(lexer);
        return next_token(lexer); // 递归获取下一个标记
    }
    
    token.line = lexer->line;
    token.column = lexer->column;
    
    // 处理标识符和关键字
    if (isalpha(c) || c == '_') {
        return read_identifier(lexer);
    }
    
    // 处理数字
    if (isdigit(c)) {
        return read_number(lexer);
    }
    
    // 处理字符常量
    if (c == '\'') {
        return read_character(lexer);
    }
    
    // 处理字符串常量
    if (c == '\"') {
        return read_string(lexer);
    }
    
    // 处理双字符操作符
    char next_c = peek_char(lexer, 1);
    
    if (c == '+' && next_c == '+') {
        token.type = TOKEN_INCREMENT;
        token.value = strdup("++");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '-' && next_c == '-') {
        token.type = TOKEN_DECREMENT;
        token.value = strdup("--");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '+' && next_c == '=') {
        token.type = TOKEN_PLUS_ASSIGN;
        token.value = strdup("+=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '-' && next_c == '=') {
        token.type = TOKEN_MINUS_ASSIGN;
        token.value = strdup("-=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '*' && next_c == '=') {
        token.type = TOKEN_MUL_ASSIGN;
        token.value = strdup("*=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '/' && next_c == '=') {
        token.type = TOKEN_DIV_ASSIGN;
        token.value = strdup("/=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '%' && next_c == '=') {
        token.type = TOKEN_MOD_ASSIGN;
        token.value = strdup("%=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '&' && next_c == '=') {
        token.type = TOKEN_AND_ASSIGN;
        token.value = strdup("&=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '|' && next_c == '=') {
        token.type = TOKEN_OR_ASSIGN;
        token.value = strdup("|=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '^' && next_c == '=') {
        token.type = TOKEN_XOR_ASSIGN;
        token.value = strdup("^=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '&' && next_c == '&') {
        token.type = TOKEN_LOGICAL_AND;
        token.value = strdup("&&");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '|' && next_c == '|') {
        token.type = TOKEN_LOGICAL_OR;
        token.value = strdup("||");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '=' && next_c == '=') {
        token.type = TOKEN_EQUAL;
        token.value = strdup("==");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '!' && next_c == '=') {
        token.type = TOKEN_NOT_EQUAL;
        token.value = strdup("!=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '<' && next_c == '=') {
        token.type = TOKEN_LESS_EQUAL;
        token.value = strdup("<=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '>' && next_c == '=') {
        token.type = TOKEN_GREATER_EQUAL;
        token.value = strdup(">=");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    if (c == '<' && next_c == '<') {
        token.type = TOKEN_LSHIFT;
        token.value = strdup("<<");
        advance_char(lexer); advance_char(lexer);
        
        // 检查 <<=
        if (current_char(lexer) == '=') {
            token.type = TOKEN_LSHIFT_ASSIGN;
            token.value = strdup("<<=");
            advance_char(lexer);
        }
        return token;
    }
    
    if (c == '>' && next_c == '>') {
        token.type = TOKEN_RSHIFT;
        token.value = strdup(">>");
        advance_char(lexer); advance_char(lexer);
        
        // 检查 >>=
        if (current_char(lexer) == '=') {
            token.type = TOKEN_RSHIFT_ASSIGN;
            token.value = strdup(">>=");
            advance_char(lexer);
        }
        return token;
    }
    
    if (c == '-' && next_c == '>') {
        token.type = TOKEN_ARROW;
        token.value = strdup("->");
        advance_char(lexer); advance_char(lexer);
        return token;
    }
    
    // 处理单字符标记
    char buffer[2] = {c, '\0'};
    token.value = strdup(buffer);
    advance_char(lexer);
    
    switch (c) {
        case '=': token.type = TOKEN_ASSIGN; break;
        case '+': token.type = TOKEN_PLUS; break;
        case '-': token.type = TOKEN_MINUS; break;
        case '*': token.type = TOKEN_MULTIPLY; break;
        case '/': token.type = TOKEN_DIVIDE; break;
        case '%': token.type = TOKEN_MODULO; break;
        case '&': token.type = TOKEN_BITWISE_AND; break;
        case '|': token.type = TOKEN_BITWISE_OR; break;
        case '^': token.type = TOKEN_BITWISE_XOR; break;
        case '~': token.type = TOKEN_BITWISE_NOT; break;
        case '!': token.type = TOKEN_LOGICAL_NOT; break;
        case '<': token.type = TOKEN_LESS; break;
        case '>': token.type = TOKEN_GREATER; break;
        case ';': token.type = TOKEN_SEMICOLON; break;
        case ',': token.type = TOKEN_COMMA; break;
        case '(': token.type = TOKEN_LPAREN; break;
        case ')': token.type = TOKEN_RPAREN; break;
        case '{': token.type = TOKEN_LBRACE; break;
        case '}': token.type = TOKEN_RBRACE; break;
        case '[': token.type = TOKEN_LBRACKET; break;
        case ']': token.type = TOKEN_RBRACKET; break;
        case '.': token.type = TOKEN_DOT; break;
        case '?': token.type = TOKEN_QUESTION; break;
        case ':': token.type = TOKEN_COLON; break;
        default:
            token.type = TOKEN_ERROR;
            error("未知字符: %c (ASCII %d)", c, (int)c);
            break;
    }
    
    return token;
}

Token *tokenize(Lexer *lexer, int *token_count) {
    Token *tokens = malloc(MAX_TOKENS * sizeof(Token));
    if (!tokens) {
        error("内存分配失败：无法分配标记数组");
        return NULL;
    }
    
    int count = 0;
    Token token;
    
    do {
        token = next_token(lexer);
        if (token.type != TOKEN_NEWLINE && token.type != TOKEN_WHITESPACE) {
            if (count >= MAX_TOKENS) {
                error("标记数量超过最大限制 %d", MAX_TOKENS);
                break;
            }
            tokens[count++] = token;
        }
    } while (token.type != TOKEN_EOF);
    
    *token_count = count;
    return tokens;
}

const char *token_type_to_string(TokenType type) {
    switch (type) {
        case TOKEN_NUMBER: return "NUMBER";
        case TOKEN_FLOAT: return "FLOAT";
        case TOKEN_CHARACTER: return "CHARACTER";
        case TOKEN_STRING: return "STRING";
        case TOKEN_IDENTIFIER: return "IDENTIFIER";
        
        // C语言关键字
        case TOKEN_AUTO: return "AUTO";
        case TOKEN_BREAK: return "BREAK";
        case TOKEN_CASE: return "CASE";
        case TOKEN_CHAR: return "CHAR";
        case TOKEN_CONST: return "CONST";
        case TOKEN_CONTINUE: return "CONTINUE";
        case TOKEN_DEFAULT: return "DEFAULT";
        case TOKEN_DO: return "DO";
        case TOKEN_DOUBLE: return "DOUBLE";
        case TOKEN_ELSE: return "ELSE";
        case TOKEN_ENUM: return "ENUM";
        case TOKEN_EXTERN: return "EXTERN";
        case TOKEN_FLOAT_KW: return "FLOAT";
        case TOKEN_FOR: return "FOR";
        case TOKEN_GOTO: return "GOTO";
        case TOKEN_IF: return "IF";
        case TOKEN_INT: return "INT";
        case TOKEN_LONG: return "LONG";
        case TOKEN_REGISTER: return "REGISTER";
        case TOKEN_RETURN: return "RETURN";
        case TOKEN_SHORT: return "SHORT";
        case TOKEN_SIGNED: return "SIGNED";
        case TOKEN_SIZEOF: return "SIZEOF";
        case TOKEN_STATIC: return "STATIC";
        case TOKEN_STRUCT: return "STRUCT";
        case TOKEN_SWITCH: return "SWITCH";
        case TOKEN_TYPEDEF: return "TYPEDEF";
        case TOKEN_UNION: return "UNION";
        case TOKEN_UNSIGNED: return "UNSIGNED";
        case TOKEN_VOID: return "VOID";
        case TOKEN_VOLATILE: return "VOLATILE";
        case TOKEN_WHILE: return "WHILE";
        
        // 并行计算扩展
        case TOKEN_PARALLEL_FOR: return "PARALLEL_FOR";
        case TOKEN_ATOMIC: return "ATOMIC";
        case TOKEN_THREAD_LOCAL: return "THREAD_LOCAL";
        case TOKEN_BARRIER: return "BARRIER";
        case TOKEN_CRITICAL: return "CRITICAL";
        
        // 操作符
        case TOKEN_ASSIGN: return "ASSIGN";
        case TOKEN_PLUS_ASSIGN: return "PLUS_ASSIGN";
        case TOKEN_MINUS_ASSIGN: return "MINUS_ASSIGN";
        case TOKEN_MUL_ASSIGN: return "MUL_ASSIGN";
        case TOKEN_DIV_ASSIGN: return "DIV_ASSIGN";
        case TOKEN_MOD_ASSIGN: return "MOD_ASSIGN";
        case TOKEN_AND_ASSIGN: return "AND_ASSIGN";
        case TOKEN_OR_ASSIGN: return "OR_ASSIGN";
        case TOKEN_XOR_ASSIGN: return "XOR_ASSIGN";
        case TOKEN_LSHIFT_ASSIGN: return "LSHIFT_ASSIGN";
        case TOKEN_RSHIFT_ASSIGN: return "RSHIFT_ASSIGN";
        case TOKEN_LOGICAL_AND: return "LOGICAL_AND";
        case TOKEN_LOGICAL_OR: return "LOGICAL_OR";
        case TOKEN_EQUAL: return "EQUAL";
        case TOKEN_NOT_EQUAL: return "NOT_EQUAL";
        case TOKEN_LESS_EQUAL: return "LESS_EQUAL";
        case TOKEN_GREATER_EQUAL: return "GREATER_EQUAL";
        case TOKEN_LSHIFT: return "LSHIFT";
        case TOKEN_RSHIFT: return "RSHIFT";
        case TOKEN_INCREMENT: return "INCREMENT";
        case TOKEN_DECREMENT: return "DECREMENT";
        case TOKEN_ARROW: return "ARROW";
        case TOKEN_PLUS: return "PLUS";
        case TOKEN_MINUS: return "MINUS";
        case TOKEN_MULTIPLY: return "MULTIPLY";
        case TOKEN_DIVIDE: return "DIVIDE";
        case TOKEN_MODULO: return "MODULO";
        case TOKEN_BITWISE_AND: return "BITWISE_AND";
        case TOKEN_BITWISE_OR: return "BITWISE_OR";
        case TOKEN_BITWISE_XOR: return "BITWISE_XOR";
        case TOKEN_BITWISE_NOT: return "BITWISE_NOT";
        case TOKEN_LOGICAL_NOT: return "LOGICAL_NOT";
        case TOKEN_LESS: return "LESS";
        case TOKEN_GREATER: return "GREATER";
        
        // 分隔符
        case TOKEN_SEMICOLON: return "SEMICOLON";
        case TOKEN_COMMA: return "COMMA";
        case TOKEN_LPAREN: return "LPAREN";
        case TOKEN_RPAREN: return "RPAREN";
        case TOKEN_LBRACE: return "LBRACE";
        case TOKEN_RBRACE: return "RBRACE";
        case TOKEN_LBRACKET: return "LBRACKET";
        case TOKEN_RBRACKET: return "RBRACKET";
        case TOKEN_DOT: return "DOT";
        case TOKEN_QUESTION: return "QUESTION";
        case TOKEN_COLON: return "COLON";
        
        // 特殊标记
        case TOKEN_EOF: return "EOF";
        case TOKEN_NEWLINE: return "NEWLINE";
        case TOKEN_WHITESPACE: return "WHITESPACE";
        case TOKEN_COMMENT: return "COMMENT";
        case TOKEN_PREPROCESSOR: return "PREPROCESSOR";
        case TOKEN_ERROR: return "ERROR";
        
        default: return "UNKNOWN";
    }
}

void print_token(Token token) {
    printf("Token(type=%s, value='%s', line=%d, col=%d", 
           token_type_to_string(token.type), 
           token.value, 
           token.line, 
           token.column);
    
    if (token.type == TOKEN_NUMBER) {
        printf(", int_value=%d", token.int_value);
    } else if (token.type == TOKEN_FLOAT) {
        printf(", float_value=%f", token.float_value);
    } else if (token.type == TOKEN_CHARACTER) {
        printf(", char_value='%c'", token.char_value);
    }
    
    printf(")\n");
}

void print_tokens(Token *tokens, int count) {
    printf("=== 词法分析结果 ===\n");
    printf("总计 %d 个标记:\n\n", count);
    
    for (int i = 0; i < count; i++) {
        printf("%3d: ", i + 1);
        print_token(tokens[i]);
    }
    
    printf("\n=== 词法分析完成 ===\n");
}
