/*
 * ParallelC Compiler - Main Entry Point
 */

#include "pcc.h"
#include <getopt.h>

static void print_usage(const char *program_name) {
    printf("Usage: %s [options] input_file\n", program_name);
    printf("Options:\n");
    printf("  -o <file>     Output file name\n");
    printf("  -O            Enable optimizations\n");
    printf("  -g            Generate debug information\n");
    printf("  -t <num>      Number of threads for parallel execution (default: 4)\n");
    printf("  -h            Show this help message\n");
    printf("  -v            Verbose output\n");
    printf("\nExamples:\n");
    printf("  %s program.c -o program\n", program_name);
    printf("  %s -O -t 8 parallel_program.c -o fast_program\n", program_name);
}

int main(int argc, char *argv[]) {
    // Default options
    CompilerOptions options = {
        .thread_count = 4,
        .optimization_enabled = false,
        .debug_info = false
    };
    
    char *input_file = NULL;
    char *output_file = "a.out";
    bool verbose = false;
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "o:Ogt:hv")) != -1) {
        switch (opt) {
            case 'o':
                output_file = optarg;
                break;
            case 'O':
                options.optimization_enabled = true;
                break;
            case 'g':
                options.debug_info = true;
                break;
            case 't':
                options.thread_count = atoi(optarg);
                if (options.thread_count <= 0) {
                    error("Invalid thread count: %s", optarg);
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                verbose = true;
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Check for input file
    if (optind >= argc) {
        error("No input file specified");
        print_usage(argv[0]);
        return 1;
    }
    
    input_file = argv[optind];
    
    if (verbose) {
        printf("ParallelC Compiler v1.0\n");
        printf("Input file: %s\n", input_file);
        printf("Output file: %s\n", output_file);
        printf("Optimizations: %s\n", options.optimization_enabled ? "enabled" : "disabled");
        printf("Debug info: %s\n", options.debug_info ? "enabled" : "disabled");
        printf("Thread count: %d\n", options.thread_count);
        printf("\n");
    }
    
    // Read source file
    if (verbose) printf("Reading source file...\n");
    char *source = read_file(input_file);
    if (!source) {
        error("Failed to read input file: %s", input_file);
        return 1;
    }
    
    // Lexical analysis
    if (verbose) printf("Performing lexical analysis...\n");
    int token_count;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) {
        error("Lexical analysis failed");
        free(source);
        return 1;
    }
    
    if (verbose) printf("Found %d tokens\n", token_count);
    
    // Syntax analysis
    if (verbose) printf("Performing syntax analysis...\n");
    ASTNode *ast = parse(tokens, token_count);
    if (!ast) {
        error("Syntax analysis failed");
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }
    
    if (verbose) printf("Abstract syntax tree constructed successfully\n");
    
    // Semantic analysis
    if (verbose) printf("Performing semantic analysis...\n");
    SymbolTable *symbol_table = create_symbol_table(NULL);
    if (!semantic_analysis(ast, symbol_table)) {
        error("Semantic analysis failed");
        free_ast(ast);
        free_symbol_table(symbol_table);
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }
    
    if (verbose) printf("Semantic analysis completed successfully\n");
    
    // Optimization
    if (options.optimization_enabled) {
        if (verbose) printf("Applying optimizations...\n");
        ast = optimize_ast(ast);
        if (verbose) printf("Optimizations applied\n");
    }
    
    // Code generation
    if (verbose) printf("Generating target code...\n");
    char *generated_code = generate_code(ast, &options);
    if (!generated_code) {
        error("Code generation failed");
        free_ast(ast);
        free_symbol_table(symbol_table);
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }
    
    if (verbose) printf("Code generation completed\n");
    
    // Write output file
    if (verbose) printf("Writing output file...\n");
    
    // Create a temporary C file with the generated code
    char temp_filename[256];
    snprintf(temp_filename, sizeof(temp_filename), "%s.generated.c", output_file);
    
    write_file(temp_filename, generated_code);
    
    // Compile the generated C code with gcc
    char compile_command[512];
    snprintf(compile_command, sizeof(compile_command), 
             "gcc %s -pthread -o %s %s", 
             options.optimization_enabled ? "-O2" : "-O0",
             output_file, 
             temp_filename);
    
    if (verbose) printf("Executing: %s\n", compile_command);
    
    int compile_result = system(compile_command);
    if (compile_result != 0) {
        error("Failed to compile generated code");
        free(generated_code);
        free_ast(ast);
        free_symbol_table(symbol_table);
        free_tokens(tokens, token_count);
        free(source);
        return 1;
    }
    
    // Clean up temporary file unless debug mode
    if (!options.debug_info) {
        remove(temp_filename);
    } else {
        if (verbose) printf("Generated C code saved to: %s\n", temp_filename);
    }
    
    if (verbose) printf("Compilation completed successfully!\n");
    printf("Output: %s\n", output_file);
    
    // Cleanup
    free(generated_code);
    free_ast(ast);
    free_symbol_table(symbol_table);
    free_tokens(tokens, token_count);
    free(source);
    
    return 0;
}
