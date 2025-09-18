/*
 * X86/X64 并行 C 编译器 - 工具函数
 * 提供内存管理、错误处理、文件操作等功能
 */

#include "x86_cc.h"

// =============================================================================
// 内存管理
// =============================================================================

void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Error: Memory allocation failed for %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t count, size_t size) {
    void *ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "Error: Memory allocation failed for %zu items of %zu bytes\n", count, size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

char *safe_strdup(const char *str) {
    if (!str) return NULL;
    
    size_t len = strlen(str) + 1;
    char *copy = (char *)safe_malloc(len);
    strcpy(copy, str);
    return copy;
}

// =============================================================================
// 错误处理和调试
// =============================================================================

void error_report(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "Error: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void warning_report(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "Warning: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void debug_print(const char *format, ...) {
    #ifdef DEBUG
    va_list args;
    va_start(args, format);
    fprintf(stderr, "Debug: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    #endif
}

// =============================================================================
// AST 节点创建函数
// =============================================================================

ASTNode *ast_create_number(long value) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_NUMBER;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->number.int_value = value;
    return node;
}

ASTNode *ast_create_float(double value) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_FLOAT;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->float_literal.float_value = value;
    return node;
}

ASTNode *ast_create_string(char *value) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_STRING;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->string.string_value = safe_strdup(value);
    return node;
}

ASTNode *ast_create_identifier(char *name) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_IDENTIFIER;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->identifier.name = safe_strdup(name);
    node->identifier.symbol = NULL;
    return node;
}

ASTNode *ast_create_binary_op(TokenType op, ASTNode *left, ASTNode *right) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_BINARY_OP;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->binary_op.operator = op;
    node->binary_op.left = left;
    node->binary_op.right = right;
    return node;
}

ASTNode *ast_create_unary_op(TokenType op, ASTNode *operand) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_UNARY_OP;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->unary_op.operator = op;
    node->unary_op.operand = operand;
    return node;
}

ASTNode *ast_create_assignment(ASTNode *left, ASTNode *right, TokenType op) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_ASSIGN;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->assignment.left = left;
    node->assignment.right = right;
    node->assignment.operator = op;
    return node;
}

ASTNode *ast_create_call(ASTNode *function, ASTNode **args, int arg_count) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_CALL;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->call.function = function;
    node->call.args = args;
    node->call.arg_count = arg_count;
    return node;
}

ASTNode *ast_create_block(ASTNode **statements, int stmt_count) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_BLOCK;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->block.statements = statements;
    node->block.stmt_count = stmt_count;
    return node;
}

ASTNode *ast_create_if(ASTNode *condition, ASTNode *then_stmt, ASTNode *else_stmt) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_IF;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->if_stmt.condition = condition;
    node->if_stmt.then_stmt = then_stmt;
    node->if_stmt.else_stmt = else_stmt;
    return node;
}

ASTNode *ast_create_for(ASTNode *init, ASTNode *condition, ASTNode *update, ASTNode *body) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_FOR;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->for_stmt.init = init;
    node->for_stmt.condition = condition;
    node->for_stmt.update = update;
    node->for_stmt.body = body;
    return node;
}

ASTNode *ast_create_while(ASTNode *condition, ASTNode *body) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_WHILE;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->while_stmt.condition = condition;
    node->while_stmt.body = body;
    return node;
}

ASTNode *ast_create_return(ASTNode *expression) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_RETURN;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->return_stmt.expression = expression;
    return node;
}

ASTNode *ast_create_var_decl(Type *type, char *name, ASTNode *initializer) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_VAR_DECL;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->var_decl.type = type;
    node->var_decl.name = safe_strdup(name);
    node->var_decl.initializer = initializer;
    return node;
}

ASTNode *ast_create_function_decl(Type *return_type, char *name, ASTNode **params, int param_count, ASTNode *body) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_FUNCTION_DECL;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->func_decl.return_type = return_type;
    node->func_decl.name = safe_strdup(name);
    node->func_decl.parameters = params;
    node->func_decl.param_count = param_count;
    node->func_decl.body = body;
    return node;
}

ASTNode *ast_create_parallel_for(ASTNode *init, ASTNode *condition, ASTNode *update, ASTNode *body, int num_threads) {
    ASTNode *node = (ASTNode *)safe_malloc(sizeof(ASTNode));
    node->type = AST_PARALLEL_FOR;
    node->line = 0;
    node->column = 0;
    node->data_type = NULL;
    node->parallel_for.init = init;
    node->parallel_for.condition = condition;
    node->parallel_for.update = update;
    node->parallel_for.body = body;
    node->parallel_for.num_threads = num_threads;
    return node;
}

void ast_destroy(ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_STRING:
            free(node->string.string_value);
            break;
            
        case AST_IDENTIFIER:
            free(node->identifier.name);
            break;
            
        case AST_BINARY_OP:
            ast_destroy(node->binary_op.left);
            ast_destroy(node->binary_op.right);
            break;
            
        case AST_UNARY_OP:
            ast_destroy(node->unary_op.operand);
            break;
            
        case AST_ASSIGN:
            ast_destroy(node->assignment.left);
            ast_destroy(node->assignment.right);
            break;
            
        case AST_CALL:
            ast_destroy(node->call.function);
            for (int i = 0; i < node->call.arg_count; i++) {
                ast_destroy(node->call.args[i]);
            }
            free(node->call.args);
            break;
            
        case AST_BLOCK:
            for (int i = 0; i < node->block.stmt_count; i++) {
                ast_destroy(node->block.statements[i]);
            }
            free(node->block.statements);
            break;
            
        case AST_IF:
            ast_destroy(node->if_stmt.condition);
            ast_destroy(node->if_stmt.then_stmt);
            ast_destroy(node->if_stmt.else_stmt);
            break;
            
        case AST_FOR:
            ast_destroy(node->for_stmt.init);
            ast_destroy(node->for_stmt.condition);
            ast_destroy(node->for_stmt.update);
            ast_destroy(node->for_stmt.body);
            break;
            
        case AST_PARALLEL_FOR:
            ast_destroy(node->parallel_for.init);
            ast_destroy(node->parallel_for.condition);
            ast_destroy(node->parallel_for.update);
            ast_destroy(node->parallel_for.body);
            break;
            
        case AST_WHILE:
            ast_destroy(node->while_stmt.condition);
            ast_destroy(node->while_stmt.body);
            break;
            
        case AST_RETURN:
            ast_destroy(node->return_stmt.expression);
            break;
            
        case AST_VAR_DECL:
            free(node->var_decl.name);
            ast_destroy(node->var_decl.initializer);
            break;
            
        case AST_FUNCTION_DECL:
            free(node->func_decl.name);
            for (int i = 0; i < node->func_decl.param_count; i++) {
                ast_destroy(node->func_decl.parameters[i]);
            }
            free(node->func_decl.parameters);
            ast_destroy(node->func_decl.body);
            break;
            
        case AST_ARRAY_ACCESS:
            ast_destroy(node->array_access.array);
            ast_destroy(node->array_access.index);
            break;
            
        case AST_MEMBER_ACCESS:
            ast_destroy(node->member_access.object);
            free(node->member_access.member);
            break;
            
        case AST_CAST:
            ast_destroy(node->cast.expression);
            break;
            
        case AST_SIZEOF:
            ast_destroy(node->sizeof_expr.expression);
            break;
            
        case AST_CONDITIONAL:
            ast_destroy(node->conditional.condition);
            ast_destroy(node->conditional.true_expr);
            ast_destroy(node->conditional.false_expr);
            break;
            
        case AST_EXPRESSION_STMT:
            ast_destroy(node->expr_stmt.expression);
            break;
            
        case AST_PARAM_DECL:
            free(node->param_decl.name);
            break;
            
        case AST_CRITICAL:
            ast_destroy(node->critical.body);
            break;
            
        default:
            // 对于没有子节点的节点类型，不需要特殊处理
            break;
    }
    
    free(node);
}

// =============================================================================
// 文件操作和命令行处理
// =============================================================================

char *read_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        error_report("Cannot open file: %s", filename);
        return NULL;
    }
    
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // 分配内存并读取文件
    char *content = (char *)safe_malloc(size + 1);
    size_t read_size = fread(content, 1, size, file);
    content[read_size] = '\0';
    
    fclose(file);
    return content;
}

bool write_file(const char *filename, const char *content) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        error_report("Cannot create file: %s", filename);
        return false;
    }
    
    size_t len = strlen(content);
    size_t written = fwrite(content, 1, len, file);
    fclose(file);
    
    return written == len;
}

// =============================================================================
// 编译器选项处理
// =============================================================================

typedef struct CompilerOptions {
    char *input_file;
    char *output_file;
    bool verbose;
    bool help;
    bool optimize;
    bool debug_info;
    bool parallel_enabled;
    bool vectorize_enabled;
    int optimization_level;
} CompilerOptions;

static void print_usage(const char *program_name) {
    printf("Usage: %s [options] <input_file>\n", program_name);
    printf("Options:\n");
    printf("  -o <file>     Specify output file\n");
    printf("  -v            Verbose output\n");
    printf("  -h            Show this help message\n");
    printf("  -O<level>     Optimization level (0-3)\n");
    printf("  -g            Generate debug information\n");
    printf("  -fopenmp      Enable parallel processing\n");
    printf("  -fvectorize   Enable vectorization\n");
    printf("\nExamples:\n");
    printf("  %s program.c -o program.s\n", program_name);
    printf("  %s -v -O2 program.c\n", program_name);
    printf("  %s -fopenmp parallel_program.c\n", program_name);
}

CompilerOptions parse_command_line(int argc, char **argv) {
    CompilerOptions opts = {0};
    opts.optimization_level = 0;
    opts.parallel_enabled = true; // 默认启用并行
    opts.vectorize_enabled = true; // 默认启用向量化
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            opts.help = true;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            opts.verbose = true;
        } else if (strcmp(argv[i], "-g") == 0) {
            opts.debug_info = true;
        } else if (strcmp(argv[i], "-fopenmp") == 0) {
            opts.parallel_enabled = true;
        } else if (strcmp(argv[i], "-fno-openmp") == 0) {
            opts.parallel_enabled = false;
        } else if (strcmp(argv[i], "-fvectorize") == 0) {
            opts.vectorize_enabled = true;
        } else if (strcmp(argv[i], "-fno-vectorize") == 0) {
            opts.vectorize_enabled = false;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                opts.output_file = argv[++i];
            } else {
                error_report("Option -o requires an argument");
                opts.help = true;
            }
        } else if (strncmp(argv[i], "-O", 2) == 0) {
            if (strlen(argv[i]) == 3 && isdigit(argv[i][2])) {
                opts.optimization_level = argv[i][2] - '0';
                opts.optimize = (opts.optimization_level > 0);
            } else {
                error_report("Invalid optimization level: %s", argv[i]);
                opts.help = true;
            }
        } else if (argv[i][0] == '-') {
            error_report("Unknown option: %s", argv[i]);
            opts.help = true;
        } else {
            if (!opts.input_file) {
                opts.input_file = argv[i];
            } else {
                error_report("Multiple input files specified");
                opts.help = true;
            }
        }
    }
    
    if (!opts.input_file && !opts.help) {
        error_report("No input file specified");
        opts.help = true;
    }
    
    // 设置默认输出文件
    if (!opts.output_file && opts.input_file) {
        size_t len = strlen(opts.input_file);
        opts.output_file = (char *)safe_malloc(len + 3);
        strcpy(opts.output_file, opts.input_file);
        
        // 替换扩展名为 .s
        char *dot = strrchr(opts.output_file, '.');
        if (dot) {
            strcpy(dot, ".s");
        } else {
            strcat(opts.output_file, ".s");
        }
    }
    
    return opts;
}

// =============================================================================
// 编译流水线
// =============================================================================

bool compile_file(const char *input_file, const char *output_file, CompilerOptions *opts) {
    debug_print("Reading input file: %s", input_file);
    
    // 读取源文件
    char *source_code = read_file(input_file);
    if (!source_code) {
        return false;
    }
    
    if (opts->verbose) {
        printf("Compiling %s -> %s\n", input_file, output_file);
    }
    
    // 词法分析
    debug_print("Starting lexical analysis");
    Lexer *lexer = lexer_create(source_code);
    if (lexer->has_error) {
        error_report("Lexical analysis failed: %s", lexer->error_msg);
        lexer_destroy(lexer);
        free(source_code);
        return false;
    }
    
    // 语法分析
    debug_print("Starting syntax analysis");
    Parser *parser = parser_create(lexer);
    ASTNode *ast = parser_parse(parser);
    if (parser->has_error || !ast) {
        error_report("Syntax analysis failed: %s", parser->error_msg);
        parser_destroy(parser);
        lexer_destroy(lexer);
        free(source_code);
        return false;
    }
    
    // 语义分析
    debug_print("Starting semantic analysis");
    if (!semantic_analyze(ast, parser->global_scope)) {
        error_report("Semantic analysis failed");
        ast_destroy(ast);
        parser_destroy(parser);
        lexer_destroy(lexer);
        free(source_code);
        return false;
    }
    
    // 代码生成
    debug_print("Starting code generation");
    FILE *output = fopen(output_file, "w");
    if (!output) {
        error_report("Cannot create output file: %s", output_file);
        ast_destroy(ast);
        parser_destroy(parser);
        lexer_destroy(lexer);
        free(source_code);
        return false;
    }
    
    CodeGenerator *generator = codegen_create(output);
    generator->optimize_level = opts->optimize;
    generator->vectorize_enabled = opts->vectorize_enabled;
    generator->parallel_enabled = opts->parallel_enabled;
    
    bool success = codegen_generate(generator, ast);
    if (!success) {
        error_report("Code generation failed: %s", generator->error_msg);
    }
    
    // 清理资源
    fclose(output);
    codegen_destroy(generator);
    ast_destroy(ast);
    parser_destroy(parser);
    lexer_destroy(lexer);
    free(source_code);
    
    if (opts->verbose) {
        if (success) {
            printf("Compilation successful: %s\n", output_file);
        } else {
            printf("Compilation failed\n");
        }
    }
    
    return success;
}

// =============================================================================
// 主编译函数
// =============================================================================

int compile_main(int argc, char **argv) {
    CompilerOptions opts = parse_command_line(argc, argv);
    
    if (opts.help) {
        print_usage(argv[0]);
        return opts.input_file ? 1 : 0;
    }
    
    if (opts.verbose) {
        printf("X86/X64 Parallel C Compiler\n");
        printf("Input:  %s\n", opts.input_file);
        printf("Output: %s\n", opts.output_file);
        printf("Optimization: %s (level %d)\n", 
               opts.optimize ? "enabled" : "disabled",
               opts.optimization_level);
        printf("Parallel processing: %s\n", 
               opts.parallel_enabled ? "enabled" : "disabled");
        printf("Vectorization: %s\n", 
               opts.vectorize_enabled ? "enabled" : "disabled");
        printf("\n");
    }
    
    bool success = compile_file(opts.input_file, opts.output_file, &opts);
    
    return success ? 0 : 1;
}
