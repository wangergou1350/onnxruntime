#include "riscv_cc.h"

// =============================================================================
// 工具函数实现
// =============================================================================

// 全局选项变量
static CompilerOptions g_options = {0};

// =============================================================================
// 错误和警告处理
// =============================================================================

void error(const char *format, ...) {
    va_list args;
    va_start(args, format);
    
    fprintf(stderr, "\033[31m错误: \033[0m");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    
    va_end(args);
    
    g_options.error_count++;
    
    if (g_options.error_count >= 10) {
        fprintf(stderr, "\033[31m致命错误: \033[0m错误数量过多，终止编译\n");
        exit(EXIT_FAILURE);
    }
}

void warning(const char *format, ...) {
    va_list args;
    va_start(args, format);
    
    fprintf(stderr, "\033[33m警告: \033[0m");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    
    va_end(args);
    
    g_options.warning_count++;
    
    if (g_options.warnings_as_errors) {
        g_options.error_count++;
    }
}

void debug_print(const char *format, ...) {
    if (!g_options.debug) return;
    
    va_list args;
    va_start(args, format);
    
    fprintf(stderr, "\033[36m调试: \033[0m");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    
    va_end(args);
}

// =============================================================================
// 文件操作
// =============================================================================

char *read_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        error("无法打开文件: %s", filename);
        return NULL;
    }
    
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (size < 0) {
        error("无法获取文件大小: %s", filename);
        fclose(file);
        return NULL;
    }
    
    // 分配缓冲区
    char *buffer = malloc(size + 1);
    if (!buffer) {
        error("内存分配失败：无法读取文件 %s", filename);
        fclose(file);
        return NULL;
    }
    
    // 读取文件内容
    size_t bytes_read = fread(buffer, 1, size, file);
    buffer[bytes_read] = '\0';
    
    fclose(file);
    
    if (g_options.verbose) {
        printf("读取文件 %s (%ld 字节)\n", filename, size);
    }
    
    return buffer;
}

bool write_file(const char *filename, const char *content) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        error("无法创建文件: %s", filename);
        return false;
    }
    
    size_t content_length = strlen(content);
    size_t bytes_written = fwrite(content, 1, content_length, file);
    
    fclose(file);
    
    if (bytes_written != content_length) {
        error("文件写入不完整: %s", filename);
        return false;
    }
    
    if (g_options.verbose) {
        printf("写入文件 %s (%zu 字节)\n", filename, content_length);
    }
    
    return true;
}

// =============================================================================
// 类型信息
// =============================================================================

int get_type_size(DataType type) {
    switch (type) {
        case TYPE_VOID:
            return 0;
        case TYPE_CHAR:
            return 1;
        case TYPE_SHORT:
            return 2;
        case TYPE_INT:
            return 4;
        case TYPE_LONG:
            return g_options.target_riscv64 ? 8 : 4;
        case TYPE_FLOAT:
            return 4;
        case TYPE_DOUBLE:
            return 8;
        case TYPE_POINTER:
            return g_options.target_riscv64 ? 8 : 4;
        case TYPE_ARRAY:
            return 0; // 数组大小需要动态计算
        case TYPE_FUNCTION:
            return 0; // 函数没有大小
        case TYPE_STRUCT:
        case TYPE_UNION:
        case TYPE_ENUM:
            return 0; // 需要查询符号表
        case TYPE_ATOMIC:
            return 4; // 简化为int大小
        default:
            return 0;
    }
}

bool is_signed_type(DataType type) {
    switch (type) {
        case TYPE_CHAR:
        case TYPE_SHORT:
        case TYPE_INT:
        case TYPE_LONG:
            return true;
        case TYPE_FLOAT:
        case TYPE_DOUBLE:
            return true; // 浮点数有符号
        default:
            return false;
    }
}

bool is_float_type(DataType type) {
    return type == TYPE_FLOAT || type == TYPE_DOUBLE;
}

const char *data_type_to_string(DataType type) {
    switch (type) {
        case TYPE_VOID: return "void";
        case TYPE_CHAR: return "char";
        case TYPE_SHORT: return "short";
        case TYPE_INT: return "int";
        case TYPE_LONG: return "long";
        case TYPE_FLOAT: return "float";
        case TYPE_DOUBLE: return "double";
        case TYPE_POINTER: return "pointer";
        case TYPE_ARRAY: return "array";
        case TYPE_FUNCTION: return "function";
        case TYPE_STRUCT: return "struct";
        case TYPE_UNION: return "union";
        case TYPE_ENUM: return "enum";
        case TYPE_ATOMIC: return "atomic";
        default: return "unknown";
    }
}

// =============================================================================
// 编译器创建和销毁
// =============================================================================

Compiler *create_compiler(void) {
    Compiler *compiler = malloc(sizeof(Compiler));
    if (!compiler) {
        error("内存分配失败：无法创建编译器");
        return NULL;
    }
    
    memset(compiler, 0, sizeof(Compiler));
    
    // 初始化默认选项
    compiler->options.verbose = false;
    compiler->options.debug = false;
    compiler->options.parallel_enabled = true;
    compiler->options.optimization_level = 0;
    compiler->options.target_riscv64 = true;
    compiler->options.only_lex = false;
    compiler->options.only_parse = false;
    compiler->options.only_semantic = false;
    compiler->options.print_ast = false;
    compiler->options.print_tokens = false;
    compiler->options.print_symbols = false;
    compiler->options.error_count = 0;
    compiler->options.warning_count = 0;
    compiler->options.warnings_as_errors = false;
    
    // 启用基本扩展
    compiler->options.enable_extensions[0] = true;  // M扩展（乘除法）
    compiler->options.enable_extensions[1] = true;  // A扩展（原子操作）
    compiler->options.enable_extensions[2] = true;  // F扩展（单精度浮点）
    compiler->options.enable_extensions[3] = true;  // D扩展（双精度浮点）
    
    // 复制全局选项
    g_options = compiler->options;
    
    return compiler;
}

void destroy_compiler(Compiler *compiler) {
    if (!compiler) return;
    
    // 销毁词法分析器
    destroy_lexer(&compiler->lexer);
    
    // 释放标记数组
    if (compiler->tokens) {
        for (int i = 0; i < compiler->token_count; i++) {
            free(compiler->tokens[i].value);
        }
        free(compiler->tokens);
    }
    
    // 销毁AST
    destroy_ast(compiler->ast);
    
    // 销毁语义分析器
    destroy_semantic_analyzer(&compiler->semantic);
    
    // 销毁代码生成器
    destroy_code_generator(&compiler->codegen);
    
    // 关闭输出文件
    if (compiler->output_file && compiler->output_file != stdout) {
        fclose(compiler->output_file);
    }
    
    free(compiler->options.input_filename);
    free(compiler->options.output_filename);
    
    free(compiler);
}

// =============================================================================
// 编译管道
// =============================================================================

bool compile_file(Compiler *compiler, const char *input_file, const char *output_file) {
    if (!compiler || !input_file) {
        error("编译器或输入文件为空");
        return false;
    }
    
    // 设置文件名
    compiler->options.input_filename = strdup(input_file);
    if (output_file) {
        compiler->options.output_filename = strdup(output_file);
    } else {
        // 生成默认输出文件名
        char *default_output = malloc(strlen(input_file) + 3);
        strcpy(default_output, input_file);
        char *dot = strrchr(default_output, '.');
        if (dot) {
            strcpy(dot, ".s");
        } else {
            strcat(default_output, ".s");
        }
        compiler->options.output_filename = default_output;
    }
    
    // 更新全局选项
    g_options = compiler->options;
    
    printf("RISC-V并行C编译器 v%d.%d.%d\n", 
           RISCV_CC_VERSION_MAJOR, RISCV_CC_VERSION_MINOR, RISCV_CC_VERSION_PATCH);
    printf("编译: %s -> %s\n", compiler->options.input_filename, compiler->options.output_filename);
    
    // 1. 读取源文件
    char *source_code = read_file(input_file);
    if (!source_code) {
        return false;
    }
    
    // 2. 词法分析
    printf("执行词法分析...\n");
    Lexer *lexer = create_lexer(source_code);
    if (!lexer) {
        free(source_code);
        return false;
    }
    
    compiler->tokens = tokenize(lexer, &compiler->token_count);
    if (!compiler->tokens) {
        destroy_lexer(lexer);
        free(source_code);
        return false;
    }
    
    if (compiler->options.print_tokens) {
        print_tokens(compiler->tokens, compiler->token_count);
    }
    
    if (compiler->options.only_lex) {
        printf("词法分析完成，输出 %d 个标记\n", compiler->token_count);
        destroy_lexer(lexer);
        free(source_code);
        return true;
    }
    
    // 3. 语法分析
    printf("执行语法分析...\n");
    compiler->ast = parse(compiler->tokens, compiler->token_count);
    if (!compiler->ast) {
        destroy_lexer(lexer);
        free(source_code);
        return false;
    }
    
    if (compiler->options.print_ast) {
        printf("\n=== 抽象语法树 ===\n");
        print_ast(compiler->ast, 0);
        printf("===================\n\n");
    }
    
    if (compiler->options.only_parse) {
        printf("语法分析完成\n");
        destroy_lexer(lexer);
        free(source_code);
        return true;
    }
    
    // 4. 语义分析
    printf("执行语义分析...\n");
    SemanticAnalyzer *semantic = create_semantic_analyzer();
    if (!semantic) {
        destroy_lexer(lexer);
        free(source_code);
        return false;
    }
    
    bool semantic_ok = analyze_semantics(semantic, compiler->ast);
    if (!semantic_ok) {
        destroy_semantic_analyzer(semantic);
        destroy_lexer(lexer);
        free(source_code);
        return false;
    }
    
    if (compiler->options.print_symbols) {
        printf("\n=== 符号表 ===\n");
        // 这里可以添加符号表打印函数
        printf("===============\n\n");
    }
    
    if (compiler->options.only_semantic) {
        printf("语义分析完成\n");
        destroy_semantic_analyzer(semantic);
        destroy_lexer(lexer);
        free(source_code);
        return true;
    }
    
    // 5. 代码生成
    printf("执行代码生成...\n");
    CodeGenerator *codegen = create_code_generator(compiler->options.target_riscv64);
    if (!codegen) {
        destroy_semantic_analyzer(semantic);
        destroy_lexer(lexer);
        free(source_code);
        return false;
    }
    
    // 打开输出文件
    compiler->output_file = fopen(compiler->options.output_filename, "w");
    if (!compiler->output_file) {
        error("无法创建输出文件: %s", compiler->options.output_filename);
        destroy_code_generator(codegen);
        destroy_semantic_analyzer(semantic);
        destroy_lexer(lexer);
        free(source_code);
        return false;
    }
    
    bool codegen_ok = generate_code(codegen, compiler->ast, compiler->output_file);
    if (!codegen_ok) {
        destroy_code_generator(codegen);
        destroy_semantic_analyzer(semantic);
        destroy_lexer(lexer);
        free(source_code);
        return false;
    }
    
    // 6. 优化（如果启用）
    if (compiler->options.optimization_level > 0) {
        printf("执行优化（级别 %d）...\n", compiler->options.optimization_level);
        if (compiler->options.optimization_level >= 1) {
            optimize_instructions(codegen);
        }
        if (compiler->options.optimization_level >= 2) {
            optimize_registers(codegen);
        }
        if (compiler->options.optimization_level >= 3 && compiler->options.parallel_enabled) {
            optimize_parallel(codegen);
        }
    }
    
    // 清理资源
    destroy_code_generator(codegen);
    destroy_semantic_analyzer(semantic);
    destroy_lexer(lexer);
    free(source_code);
    
    printf("编译成功完成！\n");
    printf("错误: %d, 警告: %d\n", compiler->options.error_count, compiler->options.warning_count);
    
    return compiler->options.error_count == 0;
}

// =============================================================================
// 命令行参数解析
// =============================================================================

void print_usage(const char *program_name) {
    printf("RISC-V并行C编译器 v%d.%d.%d\n\n", 
           RISCV_CC_VERSION_MAJOR, RISCV_CC_VERSION_MINOR, RISCV_CC_VERSION_PATCH);
    
    printf("用法: %s [选项] <输入文件>\n\n", program_name);
    
    printf("选项:\n");
    printf("  -o <文件>         指定输出文件名\n");
    printf("  -v, --verbose     详细输出模式\n");
    printf("  -g, --debug       调试模式\n");
    printf("  -O<级别>          优化级别 (0-3，默认0)\n");
    printf("  -m32              生成32位RISC-V代码\n");
    printf("  -m64              生成64位RISC-V代码 (默认)\n");
    printf("  --parallel        启用并行计算支持 (默认)\n");
    printf("  --no-parallel     禁用并行计算支持\n");
    printf("  -Werror           将警告视为错误\n");
    printf("  \n");
    printf("  调试选项:\n");
    printf("  --lex-only        仅执行词法分析\n");
    printf("  --parse-only      仅执行到语法分析\n");
    printf("  --semantic-only   仅执行到语义分析\n");
    printf("  --print-tokens    打印词法分析结果\n");
    printf("  --print-ast       打印抽象语法树\n");
    printf("  --print-symbols   打印符号表\n");
    printf("  \n");
    printf("  信息选项:\n");
    printf("  -h, --help        显示此帮助信息\n");
    printf("  --version         显示版本信息\n");
    printf("\n");
    
    printf("支持的C语言特性:\n");
    printf("  - 基本数据类型: int, char, float, double, void\n");
    printf("  - 指针和数组\n");
    printf("  - 结构体和联合体\n");
    printf("  - 函数定义和调用\n");
    printf("  - 控制流语句: if, while, for, switch\n");
    printf("  - 表达式和运算符\n");
    printf("\n");
    
    printf("并行计算扩展:\n");
    printf("  - parallel_for: 并行for循环\n");
    printf("  - atomic: 原子操作\n");
    printf("  - barrier: 内存屏障\n");
    printf("  - critical: 临界区\n");
    printf("  - thread_local: 线程局部存储\n");
    printf("\n");
    
    printf("目标架构: RISC-V (RV32I/RV64I + M/A/F/D扩展)\n");
    printf("\n");
    
    printf("示例:\n");
    printf("  %s hello.c                    # 编译为hello.s\n", program_name);
    printf("  %s -o hello.s hello.c         # 指定输出文件\n", program_name);
    printf("  %s -O2 -v hello.c             # 优化编译，详细输出\n", program_name);
    printf("  %s --print-ast hello.c        # 打印AST\n", program_name);
    printf("  %s -m32 --no-parallel hello.c # 32位，禁用并行\n", program_name);
}

int parse_arguments(int argc, char *argv[], CompilerOptions *options) {
    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }
    
    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return -1;
        }
        else if (strcmp(arg, "--version") == 0) {
            printf("RISC-V并行C编译器 v%d.%d.%d\n", 
                   RISCV_CC_VERSION_MAJOR, RISCV_CC_VERSION_MINOR, RISCV_CC_VERSION_PATCH);
            return -1;
        }
        else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            options->verbose = true;
        }
        else if (strcmp(arg, "-g") == 0 || strcmp(arg, "--debug") == 0) {
            options->debug = true;
        }
        else if (strncmp(arg, "-O", 2) == 0) {
            if (strlen(arg) == 3 && arg[2] >= '0' && arg[2] <= '3') {
                options->optimization_level = arg[2] - '0';
            } else {
                error("无效的优化级别: %s", arg);
                return -1;
            }
        }
        else if (strcmp(arg, "-m32") == 0) {
            options->target_riscv64 = false;
        }
        else if (strcmp(arg, "-m64") == 0) {
            options->target_riscv64 = true;
        }
        else if (strcmp(arg, "--parallel") == 0) {
            options->parallel_enabled = true;
        }
        else if (strcmp(arg, "--no-parallel") == 0) {
            options->parallel_enabled = false;
        }
        else if (strcmp(arg, "-Werror") == 0) {
            options->warnings_as_errors = true;
        }
        else if (strcmp(arg, "--lex-only") == 0) {
            options->only_lex = true;
        }
        else if (strcmp(arg, "--parse-only") == 0) {
            options->only_parse = true;
        }
        else if (strcmp(arg, "--semantic-only") == 0) {
            options->only_semantic = true;
        }
        else if (strcmp(arg, "--print-tokens") == 0) {
            options->print_tokens = true;
        }
        else if (strcmp(arg, "--print-ast") == 0) {
            options->print_ast = true;
        }
        else if (strcmp(arg, "--print-symbols") == 0) {
            options->print_symbols = true;
        }
        else if (strcmp(arg, "-o") == 0) {
            if (i + 1 >= argc) {
                error("-o 选项需要文件名参数");
                return -1;
            }
            options->output_filename = strdup(argv[++i]);
        }
        else if (arg[0] == '-') {
            error("未知选项: %s", arg);
            return -1;
        }
        else {
            // 输入文件
            if (options->input_filename) {
                error("只能指定一个输入文件");
                return -1;
            }
            options->input_filename = strdup(arg);
        }
    }
    
    if (!options->input_filename) {
        error("未指定输入文件");
        return -1;
    }
    
    return 0;
}

// 用于在头文件中声明但在semantic.c中需要实现的函数
void insert_builtin_symbols(SemanticAnalyzer *analyzer) {
    // 此函数在semantic.c中实现，这里提供声明
}
