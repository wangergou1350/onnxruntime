#include "pcpp.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct CodeGenerator {
    FILE *output;
    int indent_level;
    bool in_class;
    const char *current_class_name;
    int temp_var_counter;
    int label_counter;
} CodeGenerator;

static void write_indent(CodeGenerator *gen) {
    for (int i = 0; i < gen->indent_level; i++) {
        fprintf(gen->output, "    ");
    }
}

static void write_line(CodeGenerator *gen, const char *format, ...) {
    write_indent(gen);
    va_list args;
    va_start(args, format);
    vfprintf(gen->output, format, args);
    va_end(args);
    fprintf(gen->output, "\n");
}

static void write_raw(CodeGenerator *gen, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(gen->output, format, args);
    va_end(args);
}

static const char *data_type_to_string(DataType type) {
    switch (type) {
        case TYPE_INT: return "int";
        case TYPE_FLOAT: return "float";
        case TYPE_DOUBLE: return "double";
        case TYPE_CHAR: return "char";
        case TYPE_BOOL: return "bool";
        case TYPE_VOID: return "void";
        case TYPE_STRING: return "char*";
        default: return "int";
    }
}

static const char *access_modifier_to_string(AccessModifier modifier) {
    switch (modifier) {
        case ACCESS_PUBLIC: return "public";
        case ACCESS_PRIVATE: return "private";
        case ACCESS_PROTECTED: return "protected";
        default: return "private";
    }
}

// Forward declarations
static void generate_node(CodeGenerator *gen, ASTNode *node);
static void generate_expression(CodeGenerator *gen, ASTNode *node);

static void generate_includes(CodeGenerator *gen) {
    write_line(gen, "// Generated C++ to C translation");
    write_line(gen, "#include <stdio.h>");
    write_line(gen, "#include <stdlib.h>");
    write_line(gen, "#include <string.h>");
    write_line(gen, "#include <stdbool.h>");
    write_line(gen, "#include <pthread.h>");
    write_line(gen, "#include <stdatomic.h>");
    write_line(gen, "");
    
    // Define C++ style memory management
    write_line(gen, "// C++ style memory management");
    write_line(gen, "#define new(type) ((type*)malloc(sizeof(type)))");
    write_line(gen, "#define new_array(type, size) ((type*)malloc(sizeof(type) * (size)))");
    write_line(gen, "#define delete(ptr) free(ptr)");
    write_line(gen, "#define delete_array(ptr) free(ptr)");
    write_line(gen, "");
    
    // Parallel computing support
    write_line(gen, "// Parallel computing support");
    write_line(gen, "typedef struct {");
    gen->indent_level++;
    write_line(gen, "void (*func)(void*);");
    write_line(gen, "void *data;");
    write_line(gen, "int start;");
    write_line(gen, "int end;");
    gen->indent_level--;
    write_line(gen, "} parallel_task_t;");
    write_line(gen, "");
    
    write_line(gen, "void* parallel_worker(void* arg) {");
    gen->indent_level++;
    write_line(gen, "parallel_task_t* task = (parallel_task_t*)arg;");
    write_line(gen, "for (int i = task->start; i < task->end; i++) {");
    gen->indent_level++;
    write_line(gen, "task->func(task->data);");
    gen->indent_level--;
    write_line(gen, "}");
    write_line(gen, "return NULL;");
    gen->indent_level--;
    write_line(gen, "}");
    write_line(gen, "");
    
    write_line(gen, "#define parallel_for(start, end, body) do { \\");
    gen->indent_level++;
    write_line(gen, "int num_threads = 4; \\");
    write_line(gen, "pthread_t threads[num_threads]; \\");
    write_line(gen, "parallel_task_t tasks[num_threads]; \\");
    write_line(gen, "int range = ((end) - (start)) / num_threads; \\");
    write_line(gen, "for (int t = 0; t < num_threads; t++) { \\");
    gen->indent_level++;
    write_line(gen, "tasks[t].func = body; \\");
    write_line(gen, "tasks[t].data = NULL; \\");
    write_line(gen, "tasks[t].start = (start) + t * range; \\");
    write_line(gen, "tasks[t].end = (t == num_threads - 1) ? (end) : (start) + (t + 1) * range; \\");
    write_line(gen, "pthread_create(&threads[t], NULL, parallel_worker, &tasks[t]); \\");
    gen->indent_level--;
    write_line(gen, "} \\");
    write_line(gen, "for (int t = 0; t < num_threads; t++) { \\");
    gen->indent_level++;
    write_line(gen, "pthread_join(threads[t], NULL); \\");
    gen->indent_level--;
    write_line(gen, "} \\");
    gen->indent_level--;
    write_line(gen, "} while(0)");
    write_line(gen, "");
}

static void generate_class_struct(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_CLASS_DEF) return;
    
    write_line(gen, "// Class %s structure", node->class_def.name);
    write_line(gen, "typedef struct %s_s {", node->class_def.name);
    gen->indent_level++;
    
    // Virtual function table pointer (if class has virtual methods)
    bool has_virtual_methods = false;
    for (int i = 0; i < node->class_def.member_count; i++) {
        ASTNode *member = node->class_def.members[i];
        if (member->type == AST_METHOD && member->function_def.is_virtual) {
            has_virtual_methods = true;
            break;
        }
    }
    
    if (has_virtual_methods) {
        write_line(gen, "struct %s_vtable_s *vtable;", node->class_def.name);
    }
    
    // Base class data (if inheritance)
    if (node->class_def.has_base_class) {
        write_line(gen, "struct %s_s base;", node->class_def.base_class);
    }
    
    // Member variables
    for (int i = 0; i < node->class_def.member_count; i++) {
        ASTNode *member = node->class_def.members[i];
        if (member->type == AST_VARIABLE_DECL) {
            write_raw(gen, "    %s %s", 
                     data_type_to_string(member->var_decl.var_type),
                     member->var_decl.name);
            
            if (member->var_decl.array_size > 0) {
                write_raw(gen, "[%d]", member->var_decl.array_size);
            }
            write_raw(gen, ";\n");
        }
    }
    
    // Thread safety for parallel classes
    if (node->class_def.is_parallel_class) {
        write_line(gen, "pthread_mutex_t mutex;");
        write_line(gen, "atomic_int ref_count;");
    }
    
    gen->indent_level--;
    write_line(gen, "} %s_t;", node->class_def.name);
    write_line(gen, "");
    
    // Generate virtual function table if needed
    if (has_virtual_methods) {
        write_line(gen, "// Virtual function table for %s", node->class_def.name);
        write_line(gen, "typedef struct %s_vtable_s {", node->class_def.name);
        gen->indent_level++;
        
        for (int i = 0; i < node->class_def.member_count; i++) {
            ASTNode *member = node->class_def.members[i];
            if (member->type == AST_METHOD && member->function_def.is_virtual) {
                write_raw(gen, "    %s (*%s)(%s_t*", 
                         data_type_to_string(member->function_def.return_type),
                         member->function_def.name,
                         node->class_def.name);
                
                for (int j = 0; j < member->function_def.param_count; j++) {
                    write_raw(gen, ", %s", 
                             data_type_to_string(member->function_def.params[j]->var_decl.var_type));
                }
                write_raw(gen, ");\n");
            }
        }
        
        gen->indent_level--;
        write_line(gen, "} %s_vtable_t;", node->class_def.name);
        write_line(gen, "");
    }
}

static void generate_class_constructor(CodeGenerator *gen, ASTNode *class_node, ASTNode *constructor) {
    if (!class_node || !constructor) return;
    
    write_line(gen, "// Constructor for %s", class_node->class_def.name);
    write_raw(gen, "%s_t* %s_new(", class_node->class_def.name, class_node->class_def.name);
    
    // Parameters
    for (int i = 0; i < constructor->constructor.param_count; i++) {
        if (i > 0) write_raw(gen, ", ");
        write_raw(gen, "%s %s", 
                 data_type_to_string(constructor->constructor.params[i]->var_decl.var_type),
                 constructor->constructor.params[i]->var_decl.name);
    }
    write_raw(gen, ") {\n");
    
    gen->indent_level++;
    write_line(gen, "%s_t* self = new(%s_t);", class_node->class_def.name, class_node->class_def.name);
    write_line(gen, "if (!self) return NULL;");
    
    // Initialize virtual table
    bool has_virtual_methods = false;
    for (int i = 0; i < class_node->class_def.member_count; i++) {
        ASTNode *member = class_node->class_def.members[i];
        if (member->type == AST_METHOD && member->function_def.is_virtual) {
            has_virtual_methods = true;
            break;
        }
    }
    
    if (has_virtual_methods) {
        write_line(gen, "static %s_vtable_t vtable;", class_node->class_def.name);
        write_line(gen, "self->vtable = &vtable;");
    }
    
    // Initialize parallel class features
    if (class_node->class_def.is_parallel_class) {
        write_line(gen, "pthread_mutex_init(&self->mutex, NULL);");
        write_line(gen, "atomic_store(&self->ref_count, 1);");
    }
    
    // Generate constructor body
    if (constructor->constructor.body) {
        generate_node(gen, constructor->constructor.body);
    }
    
    write_line(gen, "return self;");
    gen->indent_level--;
    write_line(gen, "}");
    write_line(gen, "");
}

static void generate_class_destructor(CodeGenerator *gen, ASTNode *class_node, ASTNode *destructor) {
    if (!class_node || !destructor) return;
    
    write_line(gen, "// Destructor for %s", class_node->class_def.name);
    write_line(gen, "void %s_delete(%s_t* self) {", class_node->class_def.name, class_node->class_def.name);
    gen->indent_level++;
    
    write_line(gen, "if (!self) return;");
    
    // Parallel class cleanup
    if (class_node->class_def.is_parallel_class) {
        write_line(gen, "if (atomic_fetch_sub(&self->ref_count, 1) > 1) return;");
        write_line(gen, "pthread_mutex_destroy(&self->mutex);");
    }
    
    // Generate destructor body
    if (destructor->destructor.body) {
        generate_node(gen, destructor->destructor.body);
    }
    
    write_line(gen, "free(self);");
    gen->indent_level--;
    write_line(gen, "}");
    write_line(gen, "");
}

static void generate_class_method(CodeGenerator *gen, ASTNode *class_node, ASTNode *method) {
    if (!class_node || !method || method->type != AST_METHOD) return;
    
    write_line(gen, "// Method %s::%s", class_node->class_def.name, method->function_def.name);
    
    // Generate method signature
    write_raw(gen, "%s %s_%s(%s_t* self", 
             data_type_to_string(method->function_def.return_type),
             class_node->class_def.name,
             method->function_def.name,
             class_node->class_def.name);
    
    for (int i = 0; i < method->function_def.param_count; i++) {
        write_raw(gen, ", %s %s", 
                 data_type_to_string(method->function_def.params[i]->var_decl.var_type),
                 method->function_def.params[i]->var_decl.name);
    }
    write_raw(gen, ") {\n");
    
    gen->indent_level++;
    
    // Thread safety for parallel classes
    if (class_node->class_def.is_parallel_class && method->function_def.is_thread_safe) {
        write_line(gen, "pthread_mutex_lock(&self->mutex);");
    }
    
    // Generate method body
    if (method->function_def.body) {
        const char *old_class_name = gen->current_class_name;
        gen->current_class_name = class_node->class_def.name;
        gen->in_class = true;
        
        generate_node(gen, method->function_def.body);
        
        gen->current_class_name = old_class_name;
        gen->in_class = false;
    }
    
    // Thread safety cleanup
    if (class_node->class_def.is_parallel_class && method->function_def.is_thread_safe) {
        write_line(gen, "pthread_mutex_unlock(&self->mutex);");
    }
    
    gen->indent_level--;
    write_line(gen, "}");
    write_line(gen, "");
}

static void generate_class_definition(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_CLASS_DEF) return;
    
    // Generate class structure
    generate_class_struct(gen, node);
    
    // Find and generate constructor
    ASTNode *constructor = NULL;
    ASTNode *destructor = NULL;
    
    for (int i = 0; i < node->class_def.member_count; i++) {
        ASTNode *member = node->class_def.members[i];
        if (member->type == AST_CONSTRUCTOR) {
            constructor = member;
        } else if (member->type == AST_DESTRUCTOR) {
            destructor = member;
        }
    }
    
    // Generate default constructor if none provided
    if (!constructor) {
        write_line(gen, "// Default constructor for %s", node->class_def.name);
        write_line(gen, "%s_t* %s_new() {", node->class_def.name, node->class_def.name);
        gen->indent_level++;
        write_line(gen, "%s_t* self = new(%s_t);", node->class_def.name, node->class_def.name);
        write_line(gen, "if (!self) return NULL;");
        
        if (node->class_def.is_parallel_class) {
            write_line(gen, "pthread_mutex_init(&self->mutex, NULL);");
            write_line(gen, "atomic_store(&self->ref_count, 1);");
        }
        
        write_line(gen, "return self;");
        gen->indent_level--;
        write_line(gen, "}");
        write_line(gen, "");
    } else {
        generate_class_constructor(gen, node, constructor);
    }
    
    // Generate destructor
    if (destructor) {
        generate_class_destructor(gen, node, destructor);
    } else {
        // Generate default destructor
        write_line(gen, "// Default destructor for %s", node->class_def.name);
        write_line(gen, "void %s_delete(%s_t* self) {", node->class_def.name, node->class_def.name);
        gen->indent_level++;
        write_line(gen, "if (!self) return;");
        
        if (node->class_def.is_parallel_class) {
            write_line(gen, "if (atomic_fetch_sub(&self->ref_count, 1) > 1) return;");
            write_line(gen, "pthread_mutex_destroy(&self->mutex);");
        }
        
        write_line(gen, "free(self);");
        gen->indent_level--;
        write_line(gen, "}");
        write_line(gen, "");
    }
    
    // Generate methods
    for (int i = 0; i < node->class_def.member_count; i++) {
        ASTNode *member = node->class_def.members[i];
        if (member->type == AST_METHOD) {
            generate_class_method(gen, node, member);
        }
    }
}

static void generate_function_definition(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION_DEF) return;
    
    // Generate function signature
    write_raw(gen, "%s %s(", 
             data_type_to_string(node->function_def.return_type),
             node->function_def.name);
    
    for (int i = 0; i < node->function_def.param_count; i++) {
        if (i > 0) write_raw(gen, ", ");
        write_raw(gen, "%s %s", 
                 data_type_to_string(node->function_def.params[i]->var_decl.var_type),
                 node->function_def.params[i]->var_decl.name);
    }
    
    if (node->function_def.param_count == 0) {
        write_raw(gen, "void");
    }
    
    write_raw(gen, ") {\n");
    
    gen->indent_level++;
    
    // Generate function body
    if (node->function_def.body) {
        generate_node(gen, node->function_def.body);
    }
    
    gen->indent_level--;
    write_line(gen, "}");
    write_line(gen, "");
}

static void generate_variable_declaration(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_VARIABLE_DECL) return;
    
    write_indent(gen);
    
    if (node->var_decl.is_static) {
        write_raw(gen, "static ");
    }
    
    if (node->var_decl.is_const) {
        write_raw(gen, "const ");
    }
    
    write_raw(gen, "%s %s", 
             data_type_to_string(node->var_decl.var_type),
             node->var_decl.name);
    
    if (node->var_decl.array_size > 0) {
        write_raw(gen, "[%d]", node->var_decl.array_size);
    }
    
    if (node->var_decl.init_value) {
        write_raw(gen, " = ");
        generate_expression(gen, node->var_decl.init_value);
    }
    
    write_raw(gen, ";\n");
}

static void generate_assignment(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_ASSIGNMENT) return;
    
    write_indent(gen);
    generate_expression(gen, node->assignment.left);
    write_raw(gen, " = ");
    generate_expression(gen, node->assignment.right);
    write_raw(gen, ";\n");
}

static void generate_if_statement(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_IF_STMT) return;
    
    write_indent(gen);
    write_raw(gen, "if (");
    generate_expression(gen, node->if_stmt.condition);
    write_raw(gen, ") {\n");
    
    gen->indent_level++;
    generate_node(gen, node->if_stmt.then_stmt);
    gen->indent_level--;
    
    if (node->if_stmt.else_stmt) {
        write_line(gen, "} else {");
        gen->indent_level++;
        generate_node(gen, node->if_stmt.else_stmt);
        gen->indent_level--;
    }
    
    write_line(gen, "}");
}

static void generate_while_statement(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_WHILE_STMT) return;
    
    write_indent(gen);
    write_raw(gen, "while (");
    generate_expression(gen, node->while_stmt.condition);
    write_raw(gen, ") {\n");
    
    gen->indent_level++;
    generate_node(gen, node->while_stmt.body);
    gen->indent_level--;
    
    write_line(gen, "}");
}

static void generate_for_statement(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_FOR_STMT) return;
    
    write_indent(gen);
    write_raw(gen, "for (");
    
    if (node->for_stmt.init) {
        // Generate init without newline
        if (node->for_stmt.init->type == AST_VARIABLE_DECL) {
            ASTNode *init = node->for_stmt.init;
            write_raw(gen, "%s %s", 
                     data_type_to_string(init->var_decl.var_type),
                     init->var_decl.name);
            if (init->var_decl.init_value) {
                write_raw(gen, " = ");
                generate_expression(gen, init->var_decl.init_value);
            }
        } else {
            generate_expression(gen, node->for_stmt.init);
        }
    }
    write_raw(gen, "; ");
    
    if (node->for_stmt.condition) {
        generate_expression(gen, node->for_stmt.condition);
    }
    write_raw(gen, "; ");
    
    if (node->for_stmt.update) {
        generate_expression(gen, node->for_stmt.update);
    }
    write_raw(gen, ") {\n");
    
    gen->indent_level++;
    generate_node(gen, node->for_stmt.body);
    gen->indent_level--;
    
    write_line(gen, "}");
}

static void generate_parallel_for(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_PARALLEL_FOR) return;
    
    write_indent(gen);
    write_raw(gen, "parallel_for(");
    generate_expression(gen, node->parallel_for.start);
    write_raw(gen, ", ");
    generate_expression(gen, node->parallel_for.end);
    write_raw(gen, ", ");
    
    // Generate lambda-like function for the body
    int temp_func = gen->temp_var_counter++;
    write_raw(gen, "parallel_body_%d", temp_func);
    write_raw(gen, ");\n");
    
    // TODO: Generate the parallel body function
}

static void generate_return_statement(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_RETURN_STMT) return;
    
    write_indent(gen);
    write_raw(gen, "return");
    
    if (node->return_stmt.value) {
        write_raw(gen, " ");
        generate_expression(gen, node->return_stmt.value);
    }
    
    write_raw(gen, ";\n");
}

static void generate_block(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_BLOCK) return;
    
    for (int i = 0; i < node->block.statement_count; i++) {
        generate_node(gen, node->block.statements[i]);
    }
}

static void generate_binary_operation(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_BINARY_OP) return;
    
    write_raw(gen, "(");
    generate_expression(gen, node->binary_op.left);
    
    switch (node->binary_op.operator) {
        case TOKEN_PLUS: write_raw(gen, " + "); break;
        case TOKEN_MINUS: write_raw(gen, " - "); break;
        case TOKEN_MULTIPLY: write_raw(gen, " * "); break;
        case TOKEN_DIVIDE: write_raw(gen, " / "); break;
        case TOKEN_MODULO: write_raw(gen, " %% "); break;
        case TOKEN_EQUAL: write_raw(gen, " == "); break;
        case TOKEN_NOT_EQUAL: write_raw(gen, " != "); break;
        case TOKEN_LESS: write_raw(gen, " < "); break;
        case TOKEN_GREATER: write_raw(gen, " > "); break;
        case TOKEN_LESS_EQUAL: write_raw(gen, " <= "); break;
        case TOKEN_GREATER_EQUAL: write_raw(gen, " >= "); break;
        case TOKEN_LOGICAL_AND: write_raw(gen, " && "); break;
        case TOKEN_LOGICAL_OR: write_raw(gen, " || "); break;
        case TOKEN_BITWISE_AND: write_raw(gen, " & "); break;
        case TOKEN_BITWISE_OR: write_raw(gen, " | "); break;
        case TOKEN_BITWISE_XOR: write_raw(gen, " ^ "); break;
        case TOKEN_BITSHIFT_LEFT: write_raw(gen, " << "); break;
        case TOKEN_BITSHIFT_RIGHT: write_raw(gen, " >> "); break;
        default: write_raw(gen, " ? "); break;
    }
    
    generate_expression(gen, node->binary_op.right);
    write_raw(gen, ")");
}

static void generate_unary_operation(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_UNARY_OP) return;
    
    switch (node->unary_op.operator) {
        case TOKEN_MINUS: write_raw(gen, "-"); break;
        case TOKEN_PLUS: write_raw(gen, "+"); break;
        case TOKEN_LOGICAL_NOT: write_raw(gen, "!"); break;
        case TOKEN_BITWISE_NOT: write_raw(gen, "~"); break;
        case TOKEN_MULTIPLY: write_raw(gen, "*"); break; // Dereference
        case TOKEN_BITWISE_AND: write_raw(gen, "&"); break; // Address-of
        default: break;
    }
    
    write_raw(gen, "(");
    generate_expression(gen, node->unary_op.operand);
    write_raw(gen, ")");
}

static void generate_member_access(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_MEMBER_ACCESS) return;
    
    generate_expression(gen, node->member_access.object);
    
    if (node->member_access.is_pointer) {
        write_raw(gen, "->");
    } else {
        write_raw(gen, ".");
    }
    
    write_raw(gen, "%s", node->member_access.member);
}

static void generate_function_call(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION_CALL) return;
    
    // Handle method calls differently
    if (gen->in_class && node->function_call.object) {
        // Method call: object->method(args)
        generate_expression(gen, node->function_call.object);
        write_raw(gen, "_%s(", node->function_call.name);
        
        // Add object as first parameter for method calls
        generate_expression(gen, node->function_call.object);
        
        if (node->function_call.arg_count > 0) {
            write_raw(gen, ", ");
        }
    } else {
        write_raw(gen, "%s(", node->function_call.name);
    }
    
    for (int i = 0; i < node->function_call.arg_count; i++) {
        if (i > 0) write_raw(gen, ", ");
        generate_expression(gen, node->function_call.args[i]);
    }
    
    write_raw(gen, ")");
}

static void generate_new_expression(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_NEW_EXPR) return;
    
    if (node->new_expr.is_array) {
        write_raw(gen, "new_array(%s, ", node->new_expr.type_name);
        generate_expression(gen, node->new_expr.size);
        write_raw(gen, ")");
    } else {
        write_raw(gen, "%s_new(", node->new_expr.type_name);
        
        for (int i = 0; i < node->new_expr.arg_count; i++) {
            if (i > 0) write_raw(gen, ", ");
            generate_expression(gen, node->new_expr.args[i]);
        }
        
        write_raw(gen, ")");
    }
}

static void generate_delete_expression(CodeGenerator *gen, ASTNode *node) {
    if (!node || node->type != AST_DELETE_EXPR) return;
    
    if (node->delete_expr.is_array) {
        write_raw(gen, "delete_array(");
    } else {
        // Check if it's a class object
        write_raw(gen, "delete("); // Simplified - should check for destructor calls
    }
    
    generate_expression(gen, node->delete_expr.object);
    write_raw(gen, ")");
}

static void generate_expression(CodeGenerator *gen, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_NUMBER:
            if (node->number.is_float) {
                write_raw(gen, "%f", node->number.float_value);
            } else {
                write_raw(gen, "%d", node->number.int_value);
            }
            break;
            
        case AST_IDENTIFIER:
            write_raw(gen, "%s", node->identifier.name);
            break;
            
        case AST_STRING:
            write_raw(gen, "\"%s\"", node->string.value);
            break;
            
        case AST_BINARY_OP:
            generate_binary_operation(gen, node);
            break;
            
        case AST_UNARY_OP:
            generate_unary_operation(gen, node);
            break;
            
        case AST_MEMBER_ACCESS:
            generate_member_access(gen, node);
            break;
            
        case AST_FUNCTION_CALL:
            generate_function_call(gen, node);
            break;
            
        case AST_NEW_EXPR:
            generate_new_expression(gen, node);
            break;
            
        case AST_DELETE_EXPR:
            generate_delete_expression(gen, node);
            break;
            
        case AST_ASSIGNMENT:
            generate_expression(gen, node->assignment.left);
            write_raw(gen, " = ");
            generate_expression(gen, node->assignment.right);
            break;
            
        default:
            write_raw(gen, "/* unknown expression */");
            break;
    }
}

static void generate_node(CodeGenerator *gen, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_PROGRAM:
            for (int i = 0; i < node->program.declaration_count; i++) {
                generate_node(gen, node->program.declarations[i]);
            }
            break;
            
        case AST_CLASS_DEF:
            generate_class_definition(gen, node);
            break;
            
        case AST_FUNCTION_DEF:
            generate_function_definition(gen, node);
            break;
            
        case AST_VARIABLE_DECL:
            generate_variable_declaration(gen, node);
            break;
            
        case AST_ASSIGNMENT:
            generate_assignment(gen, node);
            break;
            
        case AST_IF_STMT:
            generate_if_statement(gen, node);
            break;
            
        case AST_WHILE_STMT:
            generate_while_statement(gen, node);
            break;
            
        case AST_FOR_STMT:
            generate_for_statement(gen, node);
            break;
            
        case AST_PARALLEL_FOR:
            generate_parallel_for(gen, node);
            break;
            
        case AST_RETURN_STMT:
            generate_return_statement(gen, node);
            break;
            
        case AST_BLOCK:
            generate_block(gen, node);
            break;
            
        case AST_BREAK_STMT:
            write_line(gen, "break;");
            break;
            
        case AST_CONTINUE_STMT:
            write_line(gen, "continue;");
            break;
            
        default:
            // Expression statements
            write_indent(gen);
            generate_expression(gen, node);
            write_raw(gen, ";\n");
            break;
    }
}

bool generate_code(ASTNode *ast, const char *output_filename) {
    if (!ast) return false;
    
    FILE *output = fopen(output_filename, "w");
    if (!output) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
        return false;
    }
    
    CodeGenerator gen = {
        .output = output,
        .indent_level = 0,
        .in_class = false,
        .current_class_name = NULL,
        .temp_var_counter = 0,
        .label_counter = 0
    };
    
    generate_includes(&gen);
    generate_node(&gen, ast);
    
    fclose(output);
    return true;
}
