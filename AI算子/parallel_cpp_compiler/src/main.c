#include "pcpp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(const char *program_name) {
    printf("Parallel C++ Compiler - C++ to C Transpiler\n");
    printf("Usage: %s [options] <input_file>\n", program_name);
    printf("\nOptions:\n");
    printf("  -o <file>     Output file (default: output.c)\n");
    printf("  -v            Verbose mode\n");
    printf("  -h, --help    Show this help message\n");
    printf("  --ast         Print AST and exit\n");
    printf("  --tokens      Print tokens and exit\n");
    printf("  --semantic    Run semantic analysis only\n");
    printf("\nExamples:\n");
    printf("  %s program.cpp\n", program_name);
    printf("  %s -o output.c -v program.cpp\n", program_name);
    printf("  %s --ast program.cpp\n", program_name);
}

char *read_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate buffer
    char *content = malloc(size + 1);
    if (!content) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    // Read file
    size_t read_size = fread(content, 1, size, file);
    content[read_size] = '\0';
    
    fclose(file);
    return content;
}

int main(int argc, char *argv[]) {
    char *input_file = NULL;
    char *output_file = "output.c";
    bool verbose = false;
    bool print_ast = false;
    bool print_tokens = false;
    bool semantic_only = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                fprintf(stderr, "Error: -o requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--ast") == 0) {
            print_ast = true;
        } else if (strcmp(argv[i], "--tokens") == 0) {
            print_tokens = true;
        } else if (strcmp(argv[i], "--semantic") == 0) {
            semantic_only = true;
        } else if (argv[i][0] != '-') {
            if (input_file == NULL) {
                input_file = argv[i];
            } else {
                fprintf(stderr, "Error: Multiple input files not supported\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!input_file) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (verbose) {
        printf("Parallel C++ Compiler v1.0\n");
        printf("Input file: %s\n", input_file);
        printf("Output file: %s\n", output_file);
    }
    
    // Read input file
    char *source_code = read_file(input_file);
    if (!source_code) {
        return 1;
    }
    
    if (verbose) {
        printf("Source code loaded (%zu bytes)\n", strlen(source_code));
    }
    
    // Lexical analysis
    if (verbose) {
        printf("Running lexical analysis...\n");
    }
    
    int token_count;
    Token *tokens = tokenize_cpp(source_code, &token_count);
    if (!tokens) {
        fprintf(stderr, "Error: Lexical analysis failed\n");
        free(source_code);
        return 1;
    }
    
    if (verbose) {
        printf("Lexical analysis complete (%d tokens)\n", token_count);
    }
    
    if (print_tokens) {
        printf("\n=== TOKENS ===\n");
        for (int i = 0; i < token_count; i++) {
            printf("Token %d: Type=%d, Value='%s', Line=%d\n", 
                   i, tokens[i].type, tokens[i].value ? tokens[i].value : "(null)", tokens[i].line);
        }
        goto cleanup;
    }
    
    // Syntax analysis
    if (verbose) {
        printf("Running syntax analysis...\n");
    }
    
    ASTNode *ast = parse_cpp(tokens, token_count);
    if (!ast) {
        fprintf(stderr, "Error: Syntax analysis failed\n");
        goto cleanup;
    }
    
    if (verbose) {
        printf("Syntax analysis complete\n");
    }
    
    if (print_ast) {
        printf("\n=== ABSTRACT SYNTAX TREE ===\n");
        print_ast(ast, 0);
        goto cleanup;
    }
    
    // Semantic analysis
    if (verbose) {
        printf("Running semantic analysis...\n");
    }
    
    char *semantic_error = NULL;
    bool semantic_success = analyze_semantics(ast, &semantic_error);
    
    if (!semantic_success) {
        fprintf(stderr, "Error: Semantic analysis failed\n");
        if (semantic_error) {
            fprintf(stderr, "%s\n", semantic_error);
            free(semantic_error);
        }
        goto cleanup;
    }
    
    if (verbose) {
        printf("Semantic analysis complete\n");
    }
    
    if (semantic_only) {
        printf("Semantic analysis passed\n");
        goto cleanup;
    }
    
    // Code generation
    if (verbose) {
        printf("Running code generation...\n");
    }
    
    bool codegen_success = generate_code(ast, output_file);
    if (!codegen_success) {
        fprintf(stderr, "Error: Code generation failed\n");
        goto cleanup;
    }
    
    if (verbose) {
        printf("Code generation complete\n");
        printf("Output written to: %s\n", output_file);
    } else {
        printf("Compilation successful: %s -> %s\n", input_file, output_file);
    }
    
cleanup:
    // Free memory
    if (tokens) {
        for (int i = 0; i < token_count; i++) {
            if (tokens[i].value) {
                free(tokens[i].value);
            }
        }
        free(tokens);
    }
    
    if (ast) {
        free_ast(ast);
    }
    
    if (source_code) {
        free(source_code);
    }
    
    return 0;
}
