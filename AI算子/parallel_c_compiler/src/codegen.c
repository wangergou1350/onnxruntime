/*
 * ParallelC Compiler - Code Generator
 * 
 * This module generates standard C code with pthread support from the AST.
 * It translates parallel constructs into pthread-based implementations.
 */

#include "pcc.h"

typedef struct CodeGenerator {
    FILE *output;
    int indent_level;
    int temp_var_counter;
    int label_counter;
    bool in_parallel_region;
} CodeGenerator;

static void generate_indent(CodeGenerator *gen) {
    for (int i = 0; i < gen->indent_level; i++) {
        fprintf(gen->output, "    ");
    }
}

static void generate_line(CodeGenerator *gen, const char *format, ...) {
    generate_indent(gen);
    
    va_list args;
    va_start(args, format);
    vfprintf(gen->output, format, args);
    va_end(args);
    
    fprintf(gen->output, "\n");
}

static void generate_raw(CodeGenerator *gen, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(gen->output, format, args);
    va_end(args);
}

static char *get_temp_var(CodeGenerator *gen) {
    static char buffer[32];
    snprintf(buffer, sizeof(buffer), "_temp_%d", gen->temp_var_counter++);
    return strdup(buffer);
}

static char *get_label(CodeGenerator *gen) {
    static char buffer[32];
    snprintf(buffer, sizeof(buffer), "_label_%d", gen->label_counter++);
    return strdup(buffer);
}

static const char *data_type_to_string(DataType type) {
    switch (type) {
        case TYPE_INT: return "int";
        case TYPE_FLOAT: return "float";
        case TYPE_CHAR: return "char";
        case TYPE_VOID: return "void";
        case TYPE_POINTER: return "void*";
        default: return "int";
    }
}

static const char *binary_op_to_string(TokenType op) {
    switch (op) {
        case TOKEN_PLUS: return "+";
        case TOKEN_MINUS: return "-";
        case TOKEN_MULTIPLY: return "*";
        case TOKEN_DIVIDE: return "/";
        case TOKEN_LESS: return "<";
        case TOKEN_GREATER: return ">";
        case TOKEN_LESS_EQUAL: return "<=";
        case TOKEN_GREATER_EQUAL: return ">=";
        case TOKEN_EQUAL: return "==";
        case TOKEN_NOT_EQUAL: return "!=";
        case TOKEN_ASSIGN: return "=";
        default: return "?";
    }
}

static const char *unary_op_to_string(TokenType op) {
    switch (op) {
        case TOKEN_MINUS: return "-";
        case TOKEN_PLUS: return "+";
        default: return "?";
    }
}

// Forward declarations
static void generate_expression(CodeGenerator *gen, ASTNode *node);
static void generate_statement(CodeGenerator *gen, ASTNode *node);

static void generate_number(CodeGenerator *gen, ASTNode *node) {
    if (node->data_type == TYPE_FLOAT) {
        generate_raw(gen, "%f", node->number.float_value);
    } else {
        generate_raw(gen, "%d", node->number.int_value);
    }
}

static void generate_string(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "\"%s\"", node->string.value);
}

static void generate_identifier(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "%s", node->identifier.name);
}

static void generate_binary_op(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "(");
    generate_expression(gen, node->binary_op.left);
    generate_raw(gen, " %s ", binary_op_to_string(node->binary_op.operator));
    generate_expression(gen, node->binary_op.right);
    generate_raw(gen, ")");
}

static void generate_unary_op(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "%s", unary_op_to_string(node->unary_op.operator));
    generate_expression(gen, node->unary_op.operand);
}

static void generate_function_call(CodeGenerator *gen, ASTNode *node) {
    // Handle parallel runtime functions specially
    if (strcmp(node->function_call.name, "thread_id") == 0) {
        generate_raw(gen, "((int)pthread_self())");
        return;
    } else if (strcmp(node->function_call.name, "num_threads") == 0) {
        generate_raw(gen, "_pcc_num_threads");
        return;
    } else if (strcmp(node->function_call.name, "atomic_add") == 0) {
        if (node->function_call.arg_count >= 2) {
            generate_raw(gen, "__sync_fetch_and_add(");
            generate_expression(gen, node->function_call.args[0]);
            generate_raw(gen, ", ");
            generate_expression(gen, node->function_call.args[1]);
            generate_raw(gen, ")");
        }
        return;
    } else if (strcmp(node->function_call.name, "atomic_sub") == 0) {
        if (node->function_call.arg_count >= 2) {
            generate_raw(gen, "__sync_fetch_and_sub(");
            generate_expression(gen, node->function_call.args[0]);
            generate_raw(gen, ", ");
            generate_expression(gen, node->function_call.args[1]);
            generate_raw(gen, ")");
        }
        return;
    } else if (strcmp(node->function_call.name, "barrier") == 0) {
        generate_raw(gen, "pthread_barrier_wait(&_pcc_barrier)");
        return;
    }
    
    // Regular function call
    generate_raw(gen, "%s(", node->function_call.name);
    
    for (int i = 0; i < node->function_call.arg_count; i++) {
        if (i > 0) {
            generate_raw(gen, ", ");
        }
        generate_expression(gen, node->function_call.args[i]);
    }
    
    generate_raw(gen, ")");
}

static void generate_array_access(CodeGenerator *gen, ASTNode *node) {
    generate_expression(gen, node->array_access.array);
    generate_raw(gen, "[");
    generate_expression(gen, node->array_access.index);
    generate_raw(gen, "]");
}

static void generate_assignment(CodeGenerator *gen, ASTNode *node) {
    generate_expression(gen, node->assignment.target);
    generate_raw(gen, " = ");
    generate_expression(gen, node->assignment.value);
}

static void generate_expression(CodeGenerator *gen, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_NUMBER:
            generate_number(gen, node);
            break;
            
        case AST_STRING:
            generate_string(gen, node);
            break;
            
        case AST_IDENTIFIER:
            generate_identifier(gen, node);
            break;
            
        case AST_BINARY_OP:
            generate_binary_op(gen, node);
            break;
            
        case AST_UNARY_OP:
            generate_unary_op(gen, node);
            break;
            
        case AST_FUNCTION_CALL:
            generate_function_call(gen, node);
            break;
            
        case AST_ARRAY_ACCESS:
            generate_array_access(gen, node);
            break;
            
        case AST_ASSIGNMENT:
            generate_assignment(gen, node);
            break;
            
        default:
            generate_raw(gen, "/* unknown expression */");
            break;
    }
}

static void generate_variable_declaration(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "%s %s", 
                data_type_to_string(node->var_decl.var_type),
                node->var_decl.name);
    
    if (node->var_decl.array_size > 0) {
        generate_raw(gen, "[%d]", node->var_decl.array_size);
    }
    
    if (node->var_decl.init_value) {
        generate_raw(gen, " = ");
        generate_expression(gen, node->var_decl.init_value);
    }
    
    generate_raw(gen, ";");
}

static void generate_if_statement(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "if (");
    generate_expression(gen, node->if_stmt.condition);
    generate_raw(gen, ") ");
    
    if (node->if_stmt.then_stmt->type == AST_BLOCK) {
        generate_raw(gen, "{\n");
        gen->indent_level++;
        
        ASTNode *block = node->if_stmt.then_stmt;
        for (int i = 0; i < block->block.statement_count; i++) {
            generate_statement(gen, block->block.statements[i]);
        }
        
        gen->indent_level--;
        generate_line(gen, "}");
    } else {
        generate_raw(gen, "\n");
        gen->indent_level++;
        generate_statement(gen, node->if_stmt.then_stmt);
        gen->indent_level--;
    }
    
    if (node->if_stmt.else_stmt) {
        generate_line(gen, "else ");
        
        if (node->if_stmt.else_stmt->type == AST_BLOCK) {
            generate_raw(gen, "{\n");
            gen->indent_level++;
            
            ASTNode *block = node->if_stmt.else_stmt;
            for (int i = 0; i < block->block.statement_count; i++) {
                generate_statement(gen, block->block.statements[i]);
            }
            
            gen->indent_level--;
            generate_line(gen, "}");
        } else {
            generate_raw(gen, "\n");
            gen->indent_level++;
            generate_statement(gen, node->if_stmt.else_stmt);
            gen->indent_level--;
        }
    }
}

static void generate_while_statement(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "while (");
    generate_expression(gen, node->while_stmt.condition);
    generate_raw(gen, ") ");
    
    if (node->while_stmt.body->type == AST_BLOCK) {
        generate_raw(gen, "{\n");
        gen->indent_level++;
        
        ASTNode *block = node->while_stmt.body;
        for (int i = 0; i < block->block.statement_count; i++) {
            generate_statement(gen, block->block.statements[i]);
        }
        
        gen->indent_level--;
        generate_line(gen, "}");
    } else {
        generate_raw(gen, "\n");
        gen->indent_level++;
        generate_statement(gen, node->while_stmt.body);
        gen->indent_level--;
    }
}

static void generate_for_statement(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "for (");
    
    if (node->for_stmt.init) {
        // Remove semicolon from init statement if it's a variable declaration
        if (node->for_stmt.init->type == AST_VARIABLE_DECL) {
            ASTNode *var_decl = node->for_stmt.init;
            generate_raw(gen, "%s %s", 
                        data_type_to_string(var_decl->var_decl.var_type),
                        var_decl->var_decl.name);
            
            if (var_decl->var_decl.init_value) {
                generate_raw(gen, " = ");
                generate_expression(gen, var_decl->var_decl.init_value);
            }
        } else {
            generate_expression(gen, node->for_stmt.init);
        }
    }
    
    generate_raw(gen, "; ");
    
    if (node->for_stmt.condition) {
        generate_expression(gen, node->for_stmt.condition);
    }
    
    generate_raw(gen, "; ");
    
    if (node->for_stmt.update) {
        generate_expression(gen, node->for_stmt.update);
    }
    
    generate_raw(gen, ") ");
    
    if (node->for_stmt.body->type == AST_BLOCK) {
        generate_raw(gen, "{\n");
        gen->indent_level++;
        
        ASTNode *block = node->for_stmt.body;
        for (int i = 0; i < block->block.statement_count; i++) {
            generate_statement(gen, block->block.statements[i]);
        }
        
        gen->indent_level--;
        generate_line(gen, "}");
    } else {
        generate_raw(gen, "\n");
        gen->indent_level++;
        generate_statement(gen, node->for_stmt.body);
        gen->indent_level--;
    }
}

static void generate_parallel_for(CodeGenerator *gen, ASTNode *node) {
    char *temp_start = get_temp_var(gen);
    char *temp_end = get_temp_var(gen);
    char *temp_data = get_temp_var(gen);
    char *temp_thread_func = get_temp_var(gen);
    
    // Generate thread function
    generate_line(gen, "typedef struct {");
    gen->indent_level++;
    generate_line(gen, "int start;");
    generate_line(gen, "int end;");
    generate_line(gen, "// Add other shared data here");
    gen->indent_level--;
    generate_line(gen, "} %s_data_t;", temp_data);
    generate_line(gen, "");
    
    generate_line(gen, "void* %s(void* arg) {", temp_thread_func);
    gen->indent_level++;
    generate_line(gen, "%s_data_t* data = (%s_data_t*)arg;", temp_data, temp_data);
    generate_line(gen, "for (int i = data->start; i < data->end; i++) {");
    gen->indent_level++;
    
    // Generate the parallel body (replace loop variable references with 'i')
    generate_statement(gen, node->parallel_for.body);
    
    gen->indent_level--;
    generate_line(gen, "}");
    generate_line(gen, "return NULL;");
    gen->indent_level--;
    generate_line(gen, "}");
    generate_line(gen, "");
    
    // Generate parallel execution code
    generate_line(gen, "{");
    gen->indent_level++;
    
    generate_line(gen, "int %s = ", temp_start);
    generate_expression(gen, node->parallel_for.start);
    generate_raw(gen, ";\n");
    
    generate_line(gen, "int %s = ", temp_end);
    generate_expression(gen, node->parallel_for.end);
    generate_raw(gen, ";\n");
    
    generate_line(gen, "int range = %s - %s;", temp_end, temp_start);
    generate_line(gen, "int chunk_size = (range + _pcc_num_threads - 1) / _pcc_num_threads;");
    generate_line(gen, "pthread_t threads[_pcc_num_threads];");
    generate_line(gen, "%s_data_t thread_data[_pcc_num_threads];", temp_data);
    generate_line(gen, "");
    
    generate_line(gen, "for (int t = 0; t < _pcc_num_threads; t++) {");
    gen->indent_level++;
    generate_line(gen, "thread_data[t].start = %s + t * chunk_size;", temp_start);
    generate_line(gen, "thread_data[t].end = thread_data[t].start + chunk_size;");
    generate_line(gen, "if (thread_data[t].end > %s) thread_data[t].end = %s;", temp_end, temp_end);
    generate_line(gen, "");
    generate_line(gen, "if (thread_data[t].start < thread_data[t].end) {");
    gen->indent_level++;
    generate_line(gen, "pthread_create(&threads[t], NULL, %s, &thread_data[t]);", temp_thread_func);
    gen->indent_level--;
    generate_line(gen, "}");
    gen->indent_level--;
    generate_line(gen, "}");
    generate_line(gen, "");
    
    generate_line(gen, "for (int t = 0; t < _pcc_num_threads; t++) {");
    gen->indent_level++;
    generate_line(gen, "if (thread_data[t].start < thread_data[t].end) {");
    gen->indent_level++;
    generate_line(gen, "pthread_join(threads[t], NULL);");
    gen->indent_level--;
    generate_line(gen, "}");
    gen->indent_level--;
    generate_line(gen, "}");
    
    gen->indent_level--;
    generate_line(gen, "}");
    
    free(temp_start);
    free(temp_end);
    free(temp_data);
    free(temp_thread_func);
}

static void generate_return_statement(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "return");
    
    if (node->return_stmt.value) {
        generate_raw(gen, " ");
        generate_expression(gen, node->return_stmt.value);
    }
    
    generate_raw(gen, ";");
}

static void generate_block(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "{\n");
    gen->indent_level++;
    
    for (int i = 0; i < node->block.statement_count; i++) {
        generate_statement(gen, node->block.statements[i]);
    }
    
    gen->indent_level--;
    generate_line(gen, "}");
}

static void generate_statement(CodeGenerator *gen, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_VARIABLE_DECL:
            generate_indent(gen);
            generate_variable_declaration(gen, node);
            generate_raw(gen, "\n");
            break;
            
        case AST_IF_STMT:
            generate_indent(gen);
            generate_if_statement(gen, node);
            break;
            
        case AST_WHILE_STMT:
            generate_indent(gen);
            generate_while_statement(gen, node);
            break;
            
        case AST_FOR_STMT:
            generate_indent(gen);
            generate_for_statement(gen, node);
            break;
            
        case AST_PARALLEL_FOR:
            generate_parallel_for(gen, node);
            break;
            
        case AST_RETURN_STMT:
            generate_indent(gen);
            generate_return_statement(gen, node);
            generate_raw(gen, "\n");
            break;
            
        case AST_BLOCK:
            generate_indent(gen);
            generate_block(gen, node);
            break;
            
        case AST_ASSIGNMENT:
        case AST_FUNCTION_CALL:
        case AST_BINARY_OP:
        case AST_UNARY_OP:
            // Expression statements
            generate_indent(gen);
            generate_expression(gen, node);
            generate_raw(gen, ";\n");
            break;
            
        default:
            generate_line(gen, "/* unknown statement */");
            break;
    }
}

static void generate_function(CodeGenerator *gen, ASTNode *node) {
    generate_raw(gen, "%s %s(", 
                data_type_to_string(node->function_def.return_type),
                node->function_def.name);
    
    // Parameters
    if (node->function_def.param_count == 0) {
        generate_raw(gen, "void");
    } else {
        for (int i = 0; i < node->function_def.param_count; i++) {
            if (i > 0) {
                generate_raw(gen, ", ");
            }
            
            ASTNode *param = node->function_def.params[i];
            generate_raw(gen, "%s %s", 
                        data_type_to_string(param->var_decl.var_type),
                        param->var_decl.name);
            
            if (param->var_decl.array_size > 0) {
                generate_raw(gen, "[%d]", param->var_decl.array_size);
            }
        }
    }
    
    generate_raw(gen, ") ");
    
    // Function body
    if (node->function_def.body) {
        generate_statement(gen, node->function_def.body);
    } else {
        generate_raw(gen, "{\n}\n");
    }
    
    generate_raw(gen, "\n");
}

static void generate_headers(CodeGenerator *gen) {
    generate_line(gen, "#include <stdio.h>");
    generate_line(gen, "#include <stdlib.h>");
    generate_line(gen, "#include <pthread.h>");
    generate_line(gen, "#include <unistd.h>");
    generate_line(gen, "");
    
    // Runtime variables
    generate_line(gen, "static int _pcc_num_threads = 4; // Default thread count");
    generate_line(gen, "static pthread_barrier_t _pcc_barrier;");
    generate_line(gen, "");
    
    // Runtime initialization function
    generate_line(gen, "void _pcc_init_runtime() {");
    gen->indent_level++;
    generate_line(gen, "_pcc_num_threads = sysconf(_SC_NPROCESSORS_ONLN);");
    generate_line(gen, "if (_pcc_num_threads <= 0) _pcc_num_threads = 4;");
    generate_line(gen, "pthread_barrier_init(&_pcc_barrier, NULL, _pcc_num_threads);");
    gen->indent_level--;
    generate_line(gen, "}");
    generate_line(gen, "");
    
    generate_line(gen, "void _pcc_cleanup_runtime() {");
    gen->indent_level++;
    generate_line(gen, "pthread_barrier_destroy(&_pcc_barrier);");
    gen->indent_level--;
    generate_line(gen, "}");
    generate_line(gen, "");
}

bool generate_code(ASTNode *ast, const char *output_filename) {
    if (!ast || ast->type != AST_PROGRAM) {
        error("Invalid AST for code generation");
        return false;
    }
    
    FILE *output = fopen(output_filename, "w");
    if (!output) {
        error("Cannot open output file: %s", output_filename);
        return false;
    }
    
    CodeGenerator gen = {
        .output = output,
        .indent_level = 0,
        .temp_var_counter = 0,
        .label_counter = 0,
        .in_parallel_region = false
    };
    
    // Generate headers and runtime
    generate_headers(&gen);
    
    // Generate all functions
    for (int i = 0; i < ast->program.function_count; i++) {
        generate_function(&gen, ast->program.functions[i]);
    }
    
    // Add runtime initialization to main if it exists
    generate_line(&gen, "// Runtime initialization wrapper");
    generate_line(&gen, "int _pcc_main_wrapper(int argc, char* argv[]) {");
    gen.indent_level++;
    generate_line(&gen, "_pcc_init_runtime();");
    generate_line(&gen, "int result = main(argc, argv);");
    generate_line(&gen, "_pcc_cleanup_runtime();");
    generate_line(&gen, "return result;");
    gen.indent_level--;
    generate_line(&gen, "}");
    generate_line(&gen, "");
    
    generate_line(&gen, "#define main _pcc_main_wrapper");
    
    fclose(output);
    return true;
}
