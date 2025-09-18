/*
 * X86/X64 并行 C 编译器 - 代码生成器
 * 负责将 AST 转换为 x86/x64 汇编代码
 */

#include "x86_cc.h"

// =============================================================================
// 寄存器分配器实现
// =============================================================================

RegisterAllocator *reg_alloc_create(void) {
    RegisterAllocator *alloc = (RegisterAllocator *)safe_malloc(sizeof(RegisterAllocator));
    
    // 初始化寄存器使用状态
    for (int i = 0; i < 16; i++) {
        alloc->general_regs[i] = false;
        alloc->xmm_regs[i] = false;
        alloc->ymm_regs[i] = false;
        alloc->reg_to_symbol[i] = NULL;
    }
    
    // 设置调用者保存寄存器 (caller-saved)
    alloc->caller_saved[0] = REG_RAX;
    alloc->caller_saved[1] = REG_RCX;
    alloc->caller_saved[2] = REG_RDX;
    alloc->caller_saved[3] = REG_RSI;
    alloc->caller_saved[4] = REG_RDI;
    alloc->caller_saved[5] = REG_R8;
    alloc->caller_saved[6] = REG_R9;
    alloc->caller_saved[7] = REG_R10;
    alloc->caller_saved[8] = REG_R11;
    alloc->caller_saved_count = 9;
    
    // 设置被调用者保存寄存器 (callee-saved)
    alloc->callee_saved[0] = REG_RBX;
    alloc->callee_saved[1] = REG_RBP;
    alloc->callee_saved[2] = REG_R12;
    alloc->callee_saved[3] = REG_R13;
    alloc->callee_saved[4] = REG_R14;
    alloc->callee_saved[5] = REG_R15;
    alloc->callee_saved_count = 6;
    
    // 设置参数寄存器 (System V ABI)
    alloc->param_regs[0] = REG_RDI;
    alloc->param_regs[1] = REG_RSI;
    alloc->param_regs[2] = REG_RDX;
    alloc->param_regs[3] = REG_RCX;
    alloc->param_regs[4] = REG_R8;
    alloc->param_regs[5] = REG_R9;
    alloc->param_reg_count = 6;
    
    // 设置返回值寄存器
    alloc->return_reg = REG_RAX;
    alloc->return_reg_float = REG_XMM0;
    
    // 设置栈和帧指针
    alloc->stack_pointer = REG_RSP;
    alloc->frame_pointer = REG_RBP;
    
    // 保留特殊寄存器
    alloc->general_regs[REG_RSP - REG_RAX] = true; // 栈指针
    alloc->general_regs[REG_RBP - REG_RAX] = true; // 帧指针
    
    return alloc;
}

void reg_alloc_destroy(RegisterAllocator *alloc) {
    if (alloc) {
        free(alloc);
    }
}

X86Register reg_alloc_get_free_reg(RegisterAllocator *alloc) {
    // 优先使用调用者保存寄存器
    for (int i = 0; i < alloc->caller_saved_count; i++) {
        X86Register reg = alloc->caller_saved[i];
        int index = reg - REG_RAX;
        if (index >= 0 && index < 16 && !alloc->general_regs[index]) {
            alloc->general_regs[index] = true;
            return reg;
        }
    }
    
    // 如果没有可用的调用者保存寄存器，使用被调用者保存寄存器
    for (int i = 0; i < alloc->callee_saved_count; i++) {
        X86Register reg = alloc->callee_saved[i];
        int index = reg - REG_RAX;
        if (index >= 0 && index < 16 && !alloc->general_regs[index]) {
            alloc->general_regs[index] = true;
            return reg;
        }
    }
    
    return REG_NONE;
}

X86Register reg_alloc_get_float_reg(RegisterAllocator *alloc) {
    for (int i = 0; i < 16; i++) {
        if (!alloc->xmm_regs[i]) {
            alloc->xmm_regs[i] = true;
            return REG_XMM0 + i;
        }
    }
    return REG_NONE;
}

void reg_alloc_free_reg(RegisterAllocator *alloc, X86Register reg) {
    if (reg >= REG_RAX && reg <= REG_R15) {
        int index = reg - REG_RAX;
        alloc->general_regs[index] = false;
        alloc->reg_to_symbol[index] = NULL;
    } else if (reg >= REG_XMM0 && reg <= REG_XMM15) {
        int index = reg - REG_XMM0;
        alloc->xmm_regs[index] = false;
    }
}

void reg_alloc_save_caller_saved(RegisterAllocator *alloc, CodeGenerator *gen) {
    for (int i = 0; i < alloc->caller_saved_count; i++) {
        X86Register reg = alloc->caller_saved[i];
        int index = reg - REG_RAX;
        if (index >= 0 && index < 16 && alloc->general_regs[index]) {
            Operand src = operand_register(reg, 8);
            fprintf(gen->output, "    push    %s\n", x86_register_name(reg, 8));
        }
    }
}

void reg_alloc_restore_caller_saved(RegisterAllocator *alloc, CodeGenerator *gen) {
    // 逆序恢复
    for (int i = alloc->caller_saved_count - 1; i >= 0; i--) {
        X86Register reg = alloc->caller_saved[i];
        int index = reg - REG_RAX;
        if (index >= 0 && index < 16 && alloc->general_regs[index]) {
            fprintf(gen->output, "    pop     %s\n", x86_register_name(reg, 8));
        }
    }
}

// =============================================================================
// 操作数创建函数
// =============================================================================

Operand operand_register(X86Register reg, int size) {
    Operand op;
    op.type = OPERAND_REGISTER;
    op.reg = reg;
    op.size = size;
    return op;
}

Operand operand_immediate(long value, int size) {
    Operand op;
    op.type = OPERAND_IMMEDIATE;
    op.immediate = value;
    op.size = size;
    return op;
}

Operand operand_memory(X86Register base, int displacement, int size) {
    Operand op;
    op.type = OPERAND_MEMORY;
    op.memory.base = base;
    op.memory.index = REG_NONE;
    op.memory.scale = 1;
    op.memory.displacement = displacement;
    op.size = size;
    return op;
}

Operand operand_memory_indexed(X86Register base, X86Register index, int scale, int displacement, int size) {
    Operand op;
    op.type = OPERAND_MEMORY;
    op.memory.base = base;
    op.memory.index = index;
    op.memory.scale = scale;
    op.memory.displacement = displacement;
    op.size = size;
    return op;
}

Operand operand_label(char *label) {
    Operand op;
    op.type = OPERAND_LABEL;
    op.label = label;
    op.size = 8;
    return op;
}

// =============================================================================
// x86 寄存器工具函数
// =============================================================================

char *x86_register_name(X86Register reg, int size) {
    static char reg_names[256][8];
    static int name_index = 0;
    char *name = reg_names[name_index % 256];
    name_index++;
    
    switch (reg) {
        case REG_RAX:
            strcpy(name, size == 8 ? "rax" : size == 4 ? "eax" : size == 2 ? "ax" : "al");
            break;
        case REG_RBX:
            strcpy(name, size == 8 ? "rbx" : size == 4 ? "ebx" : size == 2 ? "bx" : "bl");
            break;
        case REG_RCX:
            strcpy(name, size == 8 ? "rcx" : size == 4 ? "ecx" : size == 2 ? "cx" : "cl");
            break;
        case REG_RDX:
            strcpy(name, size == 8 ? "rdx" : size == 4 ? "edx" : size == 2 ? "dx" : "dl");
            break;
        case REG_RSI:
            strcpy(name, size == 8 ? "rsi" : size == 4 ? "esi" : size == 2 ? "si" : "sil");
            break;
        case REG_RDI:
            strcpy(name, size == 8 ? "rdi" : size == 4 ? "edi" : size == 2 ? "di" : "dil");
            break;
        case REG_RSP:
            strcpy(name, size == 8 ? "rsp" : size == 4 ? "esp" : "sp");
            break;
        case REG_RBP:
            strcpy(name, size == 8 ? "rbp" : size == 4 ? "ebp" : "bp");
            break;
        case REG_R8:
            strcpy(name, size == 8 ? "r8" : size == 4 ? "r8d" : size == 2 ? "r8w" : "r8b");
            break;
        case REG_R9:
            strcpy(name, size == 8 ? "r9" : size == 4 ? "r9d" : size == 2 ? "r9w" : "r9b");
            break;
        case REG_R10:
            strcpy(name, size == 8 ? "r10" : size == 4 ? "r10d" : size == 2 ? "r10w" : "r10b");
            break;
        case REG_R11:
            strcpy(name, size == 8 ? "r11" : size == 4 ? "r11d" : size == 2 ? "r11w" : "r11b");
            break;
        case REG_R12:
            strcpy(name, size == 8 ? "r12" : size == 4 ? "r12d" : size == 2 ? "r12w" : "r12b");
            break;
        case REG_R13:
            strcpy(name, size == 8 ? "r13" : size == 4 ? "r13d" : size == 2 ? "r13w" : "r13b");
            break;
        case REG_R14:
            strcpy(name, size == 8 ? "r14" : size == 4 ? "r14d" : size == 2 ? "r14w" : "r14b");
            break;
        case REG_R15:
            strcpy(name, size == 8 ? "r15" : size == 4 ? "r15d" : size == 2 ? "r15w" : "r15b");
            break;
        case REG_XMM0: strcpy(name, "xmm0"); break;
        case REG_XMM1: strcpy(name, "xmm1"); break;
        case REG_XMM2: strcpy(name, "xmm2"); break;
        case REG_XMM3: strcpy(name, "xmm3"); break;
        case REG_XMM4: strcpy(name, "xmm4"); break;
        case REG_XMM5: strcpy(name, "xmm5"); break;
        case REG_XMM6: strcpy(name, "xmm6"); break;
        case REG_XMM7: strcpy(name, "xmm7"); break;
        case REG_XMM8: strcpy(name, "xmm8"); break;
        case REG_XMM9: strcpy(name, "xmm9"); break;
        case REG_XMM10: strcpy(name, "xmm10"); break;
        case REG_XMM11: strcpy(name, "xmm11"); break;
        case REG_XMM12: strcpy(name, "xmm12"); break;
        case REG_XMM13: strcpy(name, "xmm13"); break;
        case REG_XMM14: strcpy(name, "xmm14"); break;
        case REG_XMM15: strcpy(name, "xmm15"); break;
        default:
            strcpy(name, "unknown");
            break;
    }
    
    return name;
}

bool x86_is_caller_saved(X86Register reg) {
    return (reg == REG_RAX || reg == REG_RCX || reg == REG_RDX ||
            reg == REG_RSI || reg == REG_RDI || reg == REG_R8 ||
            reg == REG_R9 || reg == REG_R10 || reg == REG_R11);
}

bool x86_is_callee_saved(X86Register reg) {
    return (reg == REG_RBX || reg == REG_RBP || reg == REG_R12 ||
            reg == REG_R13 || reg == REG_R14 || reg == REG_R15);
}

// =============================================================================
// 代码生成器实现
// =============================================================================

CodeGenerator *codegen_create(FILE *output) {
    CodeGenerator *gen = (CodeGenerator *)safe_malloc(sizeof(CodeGenerator));
    gen->output = output;
    gen->reg_alloc = reg_alloc_create();
    gen->current_scope = NULL;
    gen->label_counter = 0;
    gen->temp_var_counter = 0;
    gen->current_stack_offset = 0;
    gen->current_function = NULL;
    gen->frame_size = 0;
    gen->in_parallel_region = false;
    gen->thread_count = 1;
    gen->optimize_level = false;
    gen->vectorize_enabled = true;
    gen->parallel_enabled = true;
    gen->has_error = false;
    
    return gen;
}

void codegen_destroy(CodeGenerator *gen) {
    if (gen) {
        reg_alloc_destroy(gen->reg_alloc);
        free(gen);
    }
}

void codegen_error(CodeGenerator *gen, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(gen->error_msg, MAX_ERROR_MSG, format, args);
    va_end(args);
    gen->has_error = true;
}

static char *codegen_new_label(CodeGenerator *gen, const char *prefix) {
    char *label = (char *)safe_malloc(64);
    snprintf(label, 64, "%s%d", prefix, gen->label_counter++);
    return label;
}

void codegen_emit_comment(CodeGenerator *gen, char *comment) {
    fprintf(gen->output, "    ; %s\n", comment);
}

void codegen_emit_label(CodeGenerator *gen, char *label) {
    fprintf(gen->output, "%s:\n", label);
}

static void codegen_emit_operand(CodeGenerator *gen, Operand *op) {
    switch (op->type) {
        case OPERAND_REGISTER:
            fprintf(gen->output, "%s", x86_register_name(op->reg, op->size));
            break;
            
        case OPERAND_IMMEDIATE:
            if (op->size == 8) {
                fprintf(gen->output, "%ld", op->immediate);
            } else {
                fprintf(gen->output, "%d", (int)op->immediate);
            }
            break;
            
        case OPERAND_MEMORY:
            if (op->memory.index == REG_NONE) {
                if (op->memory.displacement == 0) {
                    fprintf(gen->output, "[%s]", x86_register_name(op->memory.base, 8));
                } else {
                    fprintf(gen->output, "[%s%+d]", x86_register_name(op->memory.base, 8), op->memory.displacement);
                }
            } else {
                if (op->memory.displacement == 0) {
                    fprintf(gen->output, "[%s+%s*%d]",
                           x86_register_name(op->memory.base, 8),
                           x86_register_name(op->memory.index, 8),
                           op->memory.scale);
                } else {
                    fprintf(gen->output, "[%s+%s*%d%+d]",
                           x86_register_name(op->memory.base, 8),
                           x86_register_name(op->memory.index, 8),
                           op->memory.scale,
                           op->memory.displacement);
                }
            }
            break;
            
        case OPERAND_LABEL:
            fprintf(gen->output, "%s", op->label);
            break;
    }
}

void codegen_emit_instruction(CodeGenerator *gen, X86Instruction inst, Operand *dst, Operand *src) {
    const char *inst_name = "";
    
    switch (inst) {
        case INST_MOV: inst_name = "mov"; break;
        case INST_MOVSX: inst_name = "movsx"; break;
        case INST_MOVZX: inst_name = "movzx"; break;
        case INST_LEA: inst_name = "lea"; break;
        case INST_ADD: inst_name = "add"; break;
        case INST_SUB: inst_name = "sub"; break;
        case INST_IMUL: inst_name = "imul"; break;
        case INST_IDIV: inst_name = "idiv"; break;
        case INST_NEG: inst_name = "neg"; break;
        case INST_INC: inst_name = "inc"; break;
        case INST_DEC: inst_name = "dec"; break;
        case INST_AND: inst_name = "and"; break;
        case INST_OR: inst_name = "or"; break;
        case INST_XOR: inst_name = "xor"; break;
        case INST_NOT: inst_name = "not"; break;
        case INST_SHL: inst_name = "shl"; break;
        case INST_SHR: inst_name = "shr"; break;
        case INST_SAR: inst_name = "sar"; break;
        case INST_CMP: inst_name = "cmp"; break;
        case INST_TEST: inst_name = "test"; break;
        case INST_JMP: inst_name = "jmp"; break;
        case INST_JE: inst_name = "je"; break;
        case INST_JNE: inst_name = "jne"; break;
        case INST_JL: inst_name = "jl"; break;
        case INST_JLE: inst_name = "jle"; break;
        case INST_JG: inst_name = "jg"; break;
        case INST_JGE: inst_name = "jge"; break;
        case INST_JZ: inst_name = "jz"; break;
        case INST_JNZ: inst_name = "jnz"; break;
        case INST_PUSH: inst_name = "push"; break;
        case INST_POP: inst_name = "pop"; break;
        case INST_CALL: inst_name = "call"; break;
        case INST_RET: inst_name = "ret"; break;
        case INST_MOVSS: inst_name = "movss"; break;
        case INST_MOVSD: inst_name = "movsd"; break;
        case INST_ADDSS: inst_name = "addss"; break;
        case INST_ADDSD: inst_name = "addsd"; break;
        case INST_SUBSS: inst_name = "subss"; break;
        case INST_SUBSD: inst_name = "subsd"; break;
        case INST_MULSS: inst_name = "mulss"; break;
        case INST_MULSD: inst_name = "mulsd"; break;
        case INST_DIVSS: inst_name = "divss"; break;
        case INST_DIVSD: inst_name = "divsd"; break;
        case INST_LOCK: inst_name = "lock"; break;
        case INST_XCHG: inst_name = "xchg"; break;
        case INST_CMPXCHG: inst_name = "cmpxchg"; break;
        case INST_XADD: inst_name = "xadd"; break;
        case INST_MFENCE: inst_name = "mfence"; break;
        case INST_LFENCE: inst_name = "lfence"; break;
        case INST_SFENCE: inst_name = "sfence"; break;
        case INST_NOP: inst_name = "nop"; break;
        case INST_CDQ: inst_name = "cdq"; break;
        case INST_CQO: inst_name = "cqo"; break;
        default: inst_name = "unknown"; break;
    }
    
    fprintf(gen->output, "    %-8s", inst_name);
    
    if (dst) {
        codegen_emit_operand(gen, dst);
        if (src) {
            fprintf(gen->output, ", ");
            codegen_emit_operand(gen, src);
        }
    } else if (src) {
        codegen_emit_operand(gen, src);
    }
    
    fprintf(gen->output, "\n");
}

// =============================================================================
// x86 指令生成辅助函数
// =============================================================================

void x86_emit_mov(CodeGenerator *gen, Operand *dst, Operand *src) {
    codegen_emit_instruction(gen, INST_MOV, dst, src);
}

void x86_emit_add(CodeGenerator *gen, Operand *dst, Operand *src) {
    codegen_emit_instruction(gen, INST_ADD, dst, src);
}

void x86_emit_sub(CodeGenerator *gen, Operand *dst, Operand *src) {
    codegen_emit_instruction(gen, INST_SUB, dst, src);
}

void x86_emit_mul(CodeGenerator *gen, Operand *dst, Operand *src) {
    codegen_emit_instruction(gen, INST_IMUL, dst, src);
}

void x86_emit_div(CodeGenerator *gen, Operand *dividend, Operand *divisor) {
    // x86 除法需要特殊处理
    if (dividend->size == 8) {
        codegen_emit_instruction(gen, INST_CQO, NULL, NULL); // 符号扩展
    } else {
        codegen_emit_instruction(gen, INST_CDQ, NULL, NULL); // 符号扩展
    }
    codegen_emit_instruction(gen, INST_IDIV, divisor, NULL);
}

void x86_emit_cmp(CodeGenerator *gen, Operand *left, Operand *right) {
    codegen_emit_instruction(gen, INST_CMP, left, right);
}

void x86_emit_jump(CodeGenerator *gen, X86Instruction jmp_type, char *label) {
    Operand label_op = operand_label(label);
    codegen_emit_instruction(gen, jmp_type, &label_op, NULL);
}

void x86_emit_call(CodeGenerator *gen, char *function_name) {
    Operand func_op = operand_label(function_name);
    codegen_emit_instruction(gen, INST_CALL, &func_op, NULL);
}

void x86_emit_ret(CodeGenerator *gen) {
    codegen_emit_instruction(gen, INST_RET, NULL, NULL);
}

// =============================================================================
// 表达式代码生成
// =============================================================================

static X86Register codegen_expression(CodeGenerator *gen, ASTNode *expr, SymbolTable *scope);

static X86Register codegen_load_variable(CodeGenerator *gen, Symbol *symbol) {
    X86Register reg = reg_alloc_get_free_reg(gen->reg_alloc);
    if (reg == REG_NONE) {
        codegen_error(gen, "No free register available");
        return REG_NONE;
    }
    
    if (symbol->is_global) {
        // 全局变量
        Operand reg_op = operand_register(reg, type_size(symbol->type));
        Operand mem_op = operand_label(symbol->name);
        x86_emit_mov(gen, &reg_op, &mem_op);
    } else {
        // 局部变量 (基于栈)
        Operand reg_op = operand_register(reg, type_size(symbol->type));
        Operand mem_op = operand_memory(REG_RBP, symbol->offset, type_size(symbol->type));
        x86_emit_mov(gen, &reg_op, &mem_op);
    }
    
    return reg;
}

static void codegen_store_variable(CodeGenerator *gen, Symbol *symbol, X86Register src_reg) {
    if (symbol->is_global) {
        // 全局变量
        Operand mem_op = operand_label(symbol->name);
        Operand reg_op = operand_register(src_reg, type_size(symbol->type));
        x86_emit_mov(gen, &mem_op, &reg_op);
    } else {
        // 局部变量 (基于栈)
        Operand mem_op = operand_memory(REG_RBP, symbol->offset, type_size(symbol->type));
        Operand reg_op = operand_register(src_reg, type_size(symbol->type));
        x86_emit_mov(gen, &mem_op, &reg_op);
    }
}

static X86Register codegen_binary_operation(CodeGenerator *gen, ASTNode *expr, SymbolTable *scope) {
    X86Register left_reg = codegen_expression(gen, expr->binary_op.left, scope);
    X86Register right_reg = codegen_expression(gen, expr->binary_op.right, scope);
    
    if (left_reg == REG_NONE || right_reg == REG_NONE) {
        return REG_NONE;
    }
    
    Operand left_op = operand_register(left_reg, type_size(expr->data_type));
    Operand right_op = operand_register(right_reg, type_size(expr->data_type));
    
    switch (expr->binary_op.operator) {
        case TOKEN_PLUS:
            x86_emit_add(gen, &left_op, &right_op);
            break;
        case TOKEN_MINUS:
            x86_emit_sub(gen, &left_op, &right_op);
            break;
        case TOKEN_MULTIPLY:
            x86_emit_mul(gen, &left_op, &right_op);
            break;
        case TOKEN_DIVIDE:
            // 除法需要特殊处理 - 移动到 RAX
            if (left_reg != REG_RAX) {
                Operand rax_op = operand_register(REG_RAX, type_size(expr->data_type));
                x86_emit_mov(gen, &rax_op, &left_op);
                reg_alloc_free_reg(gen->reg_alloc, left_reg);
                left_reg = REG_RAX;
            }
            x86_emit_div(gen, &left_op, &right_op);
            break;
        case TOKEN_MODULO:
            // 模运算 - 结果在 RDX
            if (left_reg != REG_RAX) {
                Operand rax_op = operand_register(REG_RAX, type_size(expr->data_type));
                x86_emit_mov(gen, &rax_op, &left_op);
                reg_alloc_free_reg(gen->reg_alloc, left_reg);
                left_reg = REG_RAX;
            }
            x86_emit_div(gen, &left_op, &right_op);
            reg_alloc_free_reg(gen->reg_alloc, left_reg);
            left_reg = REG_RDX;
            break;
        case TOKEN_BIT_AND:
            codegen_emit_instruction(gen, INST_AND, &left_op, &right_op);
            break;
        case TOKEN_BIT_OR:
            codegen_emit_instruction(gen, INST_OR, &left_op, &right_op);
            break;
        case TOKEN_BIT_XOR:
            codegen_emit_instruction(gen, INST_XOR, &left_op, &right_op);
            break;
        case TOKEN_LSHIFT:
            codegen_emit_instruction(gen, INST_SHL, &left_op, &right_op);
            break;
        case TOKEN_RSHIFT:
            codegen_emit_instruction(gen, INST_SHR, &left_op, &right_op);
            break;
        default:
            codegen_error(gen, "Unsupported binary operator");
            break;
    }
    
    reg_alloc_free_reg(gen->reg_alloc, right_reg);
    return left_reg;
}

static X86Register codegen_comparison(CodeGenerator *gen, ASTNode *expr, SymbolTable *scope) {
    X86Register left_reg = codegen_expression(gen, expr->binary_op.left, scope);
    X86Register right_reg = codegen_expression(gen, expr->binary_op.right, scope);
    
    if (left_reg == REG_NONE || right_reg == REG_NONE) {
        return REG_NONE;
    }
    
    Operand left_op = operand_register(left_reg, type_size(expr->binary_op.left->data_type));
    Operand right_op = operand_register(right_reg, type_size(expr->binary_op.right->data_type));
    
    x86_emit_cmp(gen, &left_op, &right_op);
    
    // 设置标志位到寄存器
    X86Register result_reg = left_reg;
    Operand result_op = operand_register(result_reg, 4);
    Operand zero_op = operand_immediate(0, 4);
    Operand one_op = operand_immediate(1, 4);
    
    x86_emit_mov(gen, &result_op, &zero_op);
    
    char *true_label = codegen_new_label(gen, ".true");
    char *end_label = codegen_new_label(gen, ".end");
    
    switch (expr->binary_op.operator) {
        case TOKEN_EQ:
            x86_emit_jump(gen, INST_JE, true_label);
            break;
        case TOKEN_NE:
            x86_emit_jump(gen, INST_JNE, true_label);
            break;
        case TOKEN_LT:
            x86_emit_jump(gen, INST_JL, true_label);
            break;
        case TOKEN_LE:
            x86_emit_jump(gen, INST_JLE, true_label);
            break;
        case TOKEN_GT:
            x86_emit_jump(gen, INST_JG, true_label);
            break;
        case TOKEN_GE:
            x86_emit_jump(gen, INST_JGE, true_label);
            break;
    }
    
    x86_emit_jump(gen, INST_JMP, end_label);
    codegen_emit_label(gen, true_label);
    x86_emit_mov(gen, &result_op, &one_op);
    codegen_emit_label(gen, end_label);
    
    reg_alloc_free_reg(gen->reg_alloc, right_reg);
    free(true_label);
    free(end_label);
    
    return result_reg;
}

static X86Register codegen_expression(CodeGenerator *gen, ASTNode *expr, SymbolTable *scope) {
    if (!expr) return REG_NONE;
    
    switch (expr->type) {
        case AST_NUMBER: {
            X86Register reg = reg_alloc_get_free_reg(gen->reg_alloc);
            if (reg == REG_NONE) {
                codegen_error(gen, "No free register available");
                return REG_NONE;
            }
            
            Operand reg_op = operand_register(reg, type_size(expr->data_type));
            Operand imm_op = operand_immediate(expr->number.int_value, type_size(expr->data_type));
            x86_emit_mov(gen, &reg_op, &imm_op);
            return reg;
        }
        
        case AST_FLOAT: {
            X86Register reg = reg_alloc_get_float_reg(gen->reg_alloc);
            if (reg == REG_NONE) {
                codegen_error(gen, "No free float register available");
                return REG_NONE;
            }
            
            // 浮点常量需要先存储在内存中
            char *const_label = codegen_new_label(gen, ".float_const");
            fprintf(gen->output, "section .data\n");
            if (expr->data_type->kind == TYPE_FLOAT) {
                fprintf(gen->output, "%s: dd %f\n", const_label, (float)expr->float_literal.float_value);
            } else {
                fprintf(gen->output, "%s: dq %f\n", const_label, expr->float_literal.float_value);
            }
            fprintf(gen->output, "section .text\n");
            
            Operand reg_op = operand_register(reg, type_size(expr->data_type));
            Operand mem_op = operand_label(const_label);
            
            if (expr->data_type->kind == TYPE_FLOAT) {
                codegen_emit_instruction(gen, INST_MOVSS, &reg_op, &mem_op);
            } else {
                codegen_emit_instruction(gen, INST_MOVSD, &reg_op, &mem_op);
            }
            
            free(const_label);
            return reg;
        }
        
        case AST_IDENTIFIER: {
            if (!expr->identifier.symbol) {
                codegen_error(gen, "Undefined identifier");
                return REG_NONE;
            }
            return codegen_load_variable(gen, expr->identifier.symbol);
        }
        
        case AST_BINARY_OP: {
            if (expr->binary_op.operator == TOKEN_EQ || expr->binary_op.operator == TOKEN_NE ||
                expr->binary_op.operator == TOKEN_LT || expr->binary_op.operator == TOKEN_LE ||
                expr->binary_op.operator == TOKEN_GT || expr->binary_op.operator == TOKEN_GE) {
                return codegen_comparison(gen, expr, scope);
            } else {
                return codegen_binary_operation(gen, expr, scope);
            }
        }
        
        case AST_UNARY_OP: {
            X86Register operand_reg = codegen_expression(gen, expr->unary_op.operand, scope);
            if (operand_reg == REG_NONE) return REG_NONE;
            
            Operand operand_op = operand_register(operand_reg, type_size(expr->data_type));
            
            switch (expr->unary_op.operator) {
                case TOKEN_MINUS:
                    codegen_emit_instruction(gen, INST_NEG, &operand_op, NULL);
                    break;
                case TOKEN_NOT:
                    codegen_emit_instruction(gen, INST_NOT, &operand_op, NULL);
                    break;
                case TOKEN_BIT_NOT:
                    codegen_emit_instruction(gen, INST_NOT, &operand_op, NULL);
                    break;
                case TOKEN_INCREMENT:
                    codegen_emit_instruction(gen, INST_INC, &operand_op, NULL);
                    break;
                case TOKEN_DECREMENT:
                    codegen_emit_instruction(gen, INST_DEC, &operand_op, NULL);
                    break;
                default:
                    codegen_error(gen, "Unsupported unary operator");
                    break;
            }
            
            return operand_reg;
        }
        
        case AST_ASSIGN: {
            X86Register value_reg = codegen_expression(gen, expr->assignment.right, scope);
            if (value_reg == REG_NONE) return REG_NONE;
            
            if (expr->assignment.left->type == AST_IDENTIFIER) {
                Symbol *symbol = expr->assignment.left->identifier.symbol;
                codegen_store_variable(gen, symbol, value_reg);
            } else {
                codegen_error(gen, "Complex assignment not implemented");
            }
            
            return value_reg;
        }
        
        case AST_CALL: {
            // 函数调用代码生成
            // 保存调用者保存寄存器
            reg_alloc_save_caller_saved(gen->reg_alloc, gen);
            
            // 计算参数并放入正确的寄存器/栈位置
            for (int i = 0; i < expr->call.arg_count; i++) {
                X86Register arg_reg = codegen_expression(gen, expr->call.args[i], scope);
                if (arg_reg == REG_NONE) return REG_NONE;
                
                if (i < gen->reg_alloc->param_reg_count) {
                    // 使用寄存器传参
                    X86Register param_reg = gen->reg_alloc->param_regs[i];
                    if (arg_reg != param_reg) {
                        Operand dst_op = operand_register(param_reg, 8);
                        Operand src_op = operand_register(arg_reg, 8);
                        x86_emit_mov(gen, &dst_op, &src_op);
                        reg_alloc_free_reg(gen->reg_alloc, arg_reg);
                    }
                } else {
                    // 使用栈传参
                    Operand src_op = operand_register(arg_reg, 8);
                    codegen_emit_instruction(gen, INST_PUSH, &src_op, NULL);
                    reg_alloc_free_reg(gen->reg_alloc, arg_reg);
                }
            }
            
            // 调用函数
            if (expr->call.function->type == AST_IDENTIFIER) {
                x86_emit_call(gen, expr->call.function->identifier.name);
            } else {
                codegen_error(gen, "Indirect function calls not implemented");
                return REG_NONE;
            }
            
            // 清理栈参数
            if (expr->call.arg_count > gen->reg_alloc->param_reg_count) {
                int stack_args = expr->call.arg_count - gen->reg_alloc->param_reg_count;
                Operand rsp_op = operand_register(REG_RSP, 8);
                Operand cleanup_op = operand_immediate(stack_args * 8, 8);
                x86_emit_add(gen, &rsp_op, &cleanup_op);
            }
            
            // 恢复调用者保存寄存器
            reg_alloc_restore_caller_saved(gen->reg_alloc, gen);
            
            // 返回值在 RAX 中
            return REG_RAX;
        }
        
        default:
            codegen_error(gen, "Expression code generation not implemented");
            return REG_NONE;
    }
}

// =============================================================================
// 语句代码生成
// =============================================================================

static bool codegen_statement(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope);

static bool codegen_function_prologue(CodeGenerator *gen, Symbol *function) {
    codegen_emit_comment(gen, "Function prologue");
    
    // 保存旧的帧指针
    Operand rbp_op = operand_register(REG_RBP, 8);
    codegen_emit_instruction(gen, INST_PUSH, &rbp_op, NULL);
    
    // 设置新的帧指针
    Operand rsp_op = operand_register(REG_RSP, 8);
    x86_emit_mov(gen, &rbp_op, &rsp_op);
    
    // 为局部变量分配栈空间
    if (gen->frame_size > 0) {
        Operand frame_op = operand_immediate(gen->frame_size, 8);
        x86_emit_sub(gen, &rsp_op, &frame_op);
    }
    
    // 保存被调用者保存寄存器
    for (int i = 0; i < gen->reg_alloc->callee_saved_count; i++) {
        X86Register reg = gen->reg_alloc->callee_saved[i];
        if (reg != REG_RBP) { // RBP 已经保存了
            Operand reg_op = operand_register(reg, 8);
            codegen_emit_instruction(gen, INST_PUSH, &reg_op, NULL);
        }
    }
    
    return true;
}

static bool codegen_function_epilogue(CodeGenerator *gen, Symbol *function) {
    codegen_emit_comment(gen, "Function epilogue");
    
    // 恢复被调用者保存寄存器
    for (int i = gen->reg_alloc->callee_saved_count - 1; i >= 0; i--) {
        X86Register reg = gen->reg_alloc->callee_saved[i];
        if (reg != REG_RBP) { // RBP 将在最后恢复
            Operand reg_op = operand_register(reg, 8);
            codegen_emit_instruction(gen, INST_POP, &reg_op, NULL);
        }
    }
    
    // 恢复栈指针
    Operand rsp_op = operand_register(REG_RSP, 8);
    Operand rbp_op = operand_register(REG_RBP, 8);
    x86_emit_mov(gen, &rsp_op, &rbp_op);
    
    // 恢复旧的帧指针
    codegen_emit_instruction(gen, INST_POP, &rbp_op, NULL);
    
    // 返回
    x86_emit_ret(gen);
    
    return true;
}

static bool codegen_if_statement(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    char *else_label = codegen_new_label(gen, ".else");
    char *end_label = codegen_new_label(gen, ".endif");
    
    // 计算条件表达式
    X86Register cond_reg = codegen_expression(gen, stmt->if_stmt.condition, scope);
    if (cond_reg == REG_NONE) {
        free(else_label);
        free(end_label);
        return false;
    }
    
    // 比较条件并跳转
    Operand cond_op = operand_register(cond_reg, 4);
    Operand zero_op = operand_immediate(0, 4);
    x86_emit_cmp(gen, &cond_op, &zero_op);
    
    if (stmt->if_stmt.else_stmt) {
        x86_emit_jump(gen, INST_JE, else_label);
    } else {
        x86_emit_jump(gen, INST_JE, end_label);
    }
    
    reg_alloc_free_reg(gen->reg_alloc, cond_reg);
    
    // 生成then分支
    if (!codegen_statement(gen, stmt->if_stmt.then_stmt, scope)) {
        free(else_label);
        free(end_label);
        return false;
    }
    
    if (stmt->if_stmt.else_stmt) {
        x86_emit_jump(gen, INST_JMP, end_label);
        codegen_emit_label(gen, else_label);
        
        if (!codegen_statement(gen, stmt->if_stmt.else_stmt, scope)) {
            free(else_label);
            free(end_label);
            return false;
        }
    }
    
    codegen_emit_label(gen, end_label);
    free(else_label);
    free(end_label);
    
    return true;
}

static bool codegen_for_statement(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    char *loop_label = codegen_new_label(gen, ".loop");
    char *end_label = codegen_new_label(gen, ".endloop");
    
    // 生成初始化代码
    if (stmt->for_stmt.init) {
        if (stmt->for_stmt.init->type == AST_VAR_DECL) {
            if (!codegen_statement(gen, stmt->for_stmt.init, scope)) {
                free(loop_label);
                free(end_label);
                return false;
            }
        } else {
            X86Register init_reg = codegen_expression(gen, stmt->for_stmt.init, scope);
            if (init_reg == REG_NONE) {
                free(loop_label);
                free(end_label);
                return false;
            }
            reg_alloc_free_reg(gen->reg_alloc, init_reg);
        }
    }
    
    // 循环开始标签
    codegen_emit_label(gen, loop_label);
    
    // 检查循环条件
    if (stmt->for_stmt.condition) {
        X86Register cond_reg = codegen_expression(gen, stmt->for_stmt.condition, scope);
        if (cond_reg == REG_NONE) {
            free(loop_label);
            free(end_label);
            return false;
        }
        
        Operand cond_op = operand_register(cond_reg, 4);
        Operand zero_op = operand_immediate(0, 4);
        x86_emit_cmp(gen, &cond_op, &zero_op);
        x86_emit_jump(gen, INST_JE, end_label);
        
        reg_alloc_free_reg(gen->reg_alloc, cond_reg);
    }
    
    // 生成循环体
    if (!codegen_statement(gen, stmt->for_stmt.body, scope)) {
        free(loop_label);
        free(end_label);
        return false;
    }
    
    // 生成更新代码
    if (stmt->for_stmt.update) {
        X86Register update_reg = codegen_expression(gen, stmt->for_stmt.update, scope);
        if (update_reg == REG_NONE) {
            free(loop_label);
            free(end_label);
            return false;
        }
        reg_alloc_free_reg(gen->reg_alloc, update_reg);
    }
    
    // 跳回循环开始
    x86_emit_jump(gen, INST_JMP, loop_label);
    
    // 循环结束标签
    codegen_emit_label(gen, end_label);
    
    free(loop_label);
    free(end_label);
    return true;
}

static bool codegen_while_statement(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    char *loop_label = codegen_new_label(gen, ".while");
    char *end_label = codegen_new_label(gen, ".endwhile");
    
    // 循环开始标签
    codegen_emit_label(gen, loop_label);
    
    // 检查循环条件
    X86Register cond_reg = codegen_expression(gen, stmt->while_stmt.condition, scope);
    if (cond_reg == REG_NONE) {
        free(loop_label);
        free(end_label);
        return false;
    }
    
    Operand cond_op = operand_register(cond_reg, 4);
    Operand zero_op = operand_immediate(0, 4);
    x86_emit_cmp(gen, &cond_op, &zero_op);
    x86_emit_jump(gen, INST_JE, end_label);
    
    reg_alloc_free_reg(gen->reg_alloc, cond_reg);
    
    // 生成循环体
    if (!codegen_statement(gen, stmt->while_stmt.body, scope)) {
        free(loop_label);
        free(end_label);
        return false;
    }
    
    // 跳回循环开始
    x86_emit_jump(gen, INST_JMP, loop_label);
    
    // 循环结束标签
    codegen_emit_label(gen, end_label);
    
    free(loop_label);
    free(end_label);
    return true;
}

static bool codegen_return_statement(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    if (stmt->return_stmt.expression) {
        X86Register value_reg = codegen_expression(gen, stmt->return_stmt.expression, scope);
        if (value_reg == REG_NONE) return false;
        
        // 将返回值移动到返回值寄存器
        if (value_reg != gen->reg_alloc->return_reg) {
            Operand dst_op = operand_register(gen->reg_alloc->return_reg, 8);
            Operand src_op = operand_register(value_reg, 8);
            x86_emit_mov(gen, &dst_op, &src_op);
            reg_alloc_free_reg(gen->reg_alloc, value_reg);
        }
    }
    
    // 生成函数尾声
    codegen_function_epilogue(gen, gen->current_function);
    
    return true;
}

static bool codegen_variable_declaration(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    Symbol *symbol = symbol_table_lookup_current_scope(scope, stmt->var_decl.name);
    if (!symbol) {
        // 创建符号并分配栈空间
        symbol = symbol_create(SYMBOL_VAR, stmt->var_decl.name, stmt->var_decl.type);
        symbol->is_global = (scope->scope_level == 0);
        
        if (symbol->is_global) {
            // 全局变量
            fprintf(gen->output, "section .bss\n");
            fprintf(gen->output, "%s: resb %d\n", symbol->name, type_size(symbol->type));
            fprintf(gen->output, "section .text\n");
        } else {
            // 局部变量 - 分配栈空间
            gen->current_stack_offset -= type_size(symbol->type);
            symbol->offset = gen->current_stack_offset;
            gen->frame_size = -gen->current_stack_offset;
        }
        
        symbol_table_insert(scope, symbol);
    }
    
    // 处理初始化
    if (stmt->var_decl.initializer) {
        X86Register init_reg = codegen_expression(gen, stmt->var_decl.initializer, scope);
        if (init_reg == REG_NONE) return false;
        
        codegen_store_variable(gen, symbol, init_reg);
        reg_alloc_free_reg(gen->reg_alloc, init_reg);
    }
    
    return true;
}

static bool codegen_function_declaration(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    // 生成函数标签
    fprintf(gen->output, "global %s\n", stmt->func_decl.name);
    codegen_emit_label(gen, stmt->func_decl.name);
    
    // 设置当前函数
    Symbol *func_symbol = symbol_table_lookup(scope, stmt->func_decl.name);
    gen->current_function = func_symbol;
    gen->current_stack_offset = 0;
    gen->frame_size = 0;
    
    // 如果有函数体，生成代码
    if (stmt->func_decl.body) {
        // 创建函数作用域
        SymbolTable *func_scope = symbol_table_create(scope);
        gen->current_scope = func_scope;
        
        // 添加参数到作用域并分配位置
        int param_offset = 16; // 跳过返回地址和保存的RBP
        for (int i = 0; i < stmt->func_decl.param_count; i++) {
            ASTNode *param = stmt->func_decl.parameters[i];
            Symbol *param_symbol = symbol_create(SYMBOL_PARAM, param->param_decl.name, param->param_decl.type);
            
            if (i < gen->reg_alloc->param_reg_count) {
                // 参数在寄存器中，需要保存到栈上
                param_symbol->offset = -(param_offset + type_size(param_symbol->type));
                param_offset += type_size(param_symbol->type);
            } else {
                // 参数在栈上
                param_symbol->offset = param_offset;
                param_offset += type_size(param_symbol->type);
            }
            
            symbol_table_insert(func_scope, param_symbol);
        }
        
        // 生成函数序言
        codegen_function_prologue(gen, func_symbol);
        
        // 保存寄存器参数到栈上
        for (int i = 0; i < stmt->func_decl.param_count && i < gen->reg_alloc->param_reg_count; i++) {
            ASTNode *param = stmt->func_decl.parameters[i];
            Symbol *param_symbol = symbol_table_lookup(func_scope, param->param_decl.name);
            X86Register param_reg = gen->reg_alloc->param_regs[i];
            
            Operand mem_op = operand_memory(REG_RBP, param_symbol->offset, type_size(param_symbol->type));
            Operand reg_op = operand_register(param_reg, type_size(param_symbol->type));
            x86_emit_mov(gen, &mem_op, &reg_op);
        }
        
        // 生成函数体
        bool result = codegen_statement(gen, stmt->func_decl.body, func_scope);
        
        // 如果函数没有显式返回，生成默认返回
        if (result && stmt->func_decl.return_type->kind == TYPE_VOID) {
            codegen_function_epilogue(gen, func_symbol);
        }
        
        symbol_table_destroy(func_scope);
        gen->current_scope = scope;
        return result;
    }
    
    return true;
}

static bool codegen_block_statement(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    // 创建新的作用域
    SymbolTable *block_scope = symbol_table_create(scope);
    
    for (int i = 0; i < stmt->block.stmt_count; i++) {
        if (!codegen_statement(gen, stmt->block.statements[i], block_scope)) {
            symbol_table_destroy(block_scope);
            return false;
        }
    }
    
    symbol_table_destroy(block_scope);
    return true;
}

static bool codegen_statement(CodeGenerator *gen, ASTNode *stmt, SymbolTable *scope) {
    if (!stmt) return true;
    
    switch (stmt->type) {
        case AST_BLOCK:
            return codegen_block_statement(gen, stmt, scope);
            
        case AST_IF:
            return codegen_if_statement(gen, stmt, scope);
            
        case AST_FOR:
            return codegen_for_statement(gen, stmt, scope);
            
        case AST_WHILE:
            return codegen_while_statement(gen, stmt, scope);
            
        case AST_RETURN:
            return codegen_return_statement(gen, stmt, scope);
            
        case AST_VAR_DECL:
            return codegen_variable_declaration(gen, stmt, scope);
            
        case AST_FUNCTION_DECL:
            return codegen_function_declaration(gen, stmt, scope);
            
        case AST_EXPRESSION_STMT: {
            X86Register expr_reg = codegen_expression(gen, stmt->expr_stmt.expression, scope);
            if (expr_reg != REG_NONE) {
                reg_alloc_free_reg(gen->reg_alloc, expr_reg);
            }
            return expr_reg != REG_NONE;
        }
        
        case AST_PARALLEL_FOR:
            return x86_emit_parallel_for(gen, stmt);
            
        case AST_CRITICAL:
            return x86_emit_critical_section(gen, stmt);
            
        case AST_BARRIER:
            x86_emit_barrier(gen);
            return true;
            
        default:
            codegen_error(gen, "Statement code generation not implemented");
            return false;
    }
}

// =============================================================================
// 并行代码生成
// =============================================================================

void x86_emit_parallel_for(CodeGenerator *gen, ASTNode *parallel_for) {
    codegen_emit_comment(gen, "Parallel for loop begin");
    
    gen->in_parallel_region = true;
    
    // 简化实现：生成 OpenMP 风格的并行循环
    // 在实际实现中，这里会生成线程创建和任务分发代码
    
    char *parallel_label = codegen_new_label(gen, ".parallel_region");
    char *end_parallel_label = codegen_new_label(gen, ".end_parallel");
    
    // 保存线程数
    int old_thread_count = gen->thread_count;
    if (parallel_for->parallel_for.num_threads > 0) {
        gen->thread_count = parallel_for->parallel_for.num_threads;
    } else {
        gen->thread_count = 4; // 默认线程数
    }
    
    // 生成线程同步点
    x86_emit_barrier(gen);
    
    // 生成并行区域标签
    codegen_emit_label(gen, parallel_label);
    
    // 创建并行作用域
    SymbolTable *parallel_scope = symbol_table_create(gen->current_scope);
    
    // 生成初始化代码 (每个线程执行一次)
    if (parallel_for->parallel_for.init) {
        if (parallel_for->parallel_for.init->type == AST_VAR_DECL) {
            codegen_statement(gen, parallel_for->parallel_for.init, parallel_scope);
        } else {
            X86Register init_reg = codegen_expression(gen, parallel_for->parallel_for.init, parallel_scope);
            if (init_reg != REG_NONE) {
                reg_alloc_free_reg(gen->reg_alloc, init_reg);
            }
        }
    }
    
    // 生成线程ID获取代码 (简化版本)
    codegen_emit_comment(gen, "Get thread ID");
    fprintf(gen->output, "    call    get_thread_id\n");
    fprintf(gen->output, "    mov     r12, rax        ; r12 = thread_id\n");
    
    // 生成迭代范围计算
    codegen_emit_comment(gen, "Calculate iteration range");
    fprintf(gen->output, "    mov     r13, %d         ; r13 = num_threads\n", gen->thread_count);
    
    // 简化的并行循环：每个线程处理部分迭代
    char *loop_label = codegen_new_label(gen, ".parallel_loop");
    char *end_loop_label = codegen_new_label(gen, ".end_parallel_loop");
    
    codegen_emit_label(gen, loop_label);
    
    // 检查循环条件 (需要根据线程ID调整)
    if (parallel_for->parallel_for.condition) {
        X86Register cond_reg = codegen_expression(gen, parallel_for->parallel_for.condition, parallel_scope);
        if (cond_reg != REG_NONE) {
            Operand cond_op = operand_register(cond_reg, 4);
            Operand zero_op = operand_immediate(0, 4);
            x86_emit_cmp(gen, &cond_op, &zero_op);
            x86_emit_jump(gen, INST_JE, end_loop_label);
            reg_alloc_free_reg(gen->reg_alloc, cond_reg);
        }
    }
    
    // 生成循环体
    codegen_statement(gen, parallel_for->parallel_for.body, parallel_scope);
    
    // 生成更新代码
    if (parallel_for->parallel_for.update) {
        X86Register update_reg = codegen_expression(gen, parallel_for->parallel_for.update, parallel_scope);
        if (update_reg != REG_NONE) {
            reg_alloc_free_reg(gen->reg_alloc, update_reg);
        }
    }
    
    x86_emit_jump(gen, INST_JMP, loop_label);
    codegen_emit_label(gen, end_loop_label);
    
    // 线程同步
    x86_emit_barrier(gen);
    
    codegen_emit_label(gen, end_parallel_label);
    
    symbol_table_destroy(parallel_scope);
    gen->thread_count = old_thread_count;
    gen->in_parallel_region = false;
    
    free(parallel_label);
    free(end_parallel_label);
    free(loop_label);
    free(end_loop_label);
    
    codegen_emit_comment(gen, "Parallel for loop end");
}

void x86_emit_atomic_operation(CodeGenerator *gen, ASTNode *atomic_op) {
    codegen_emit_comment(gen, "Atomic operation");
    
    switch (atomic_op->atomic_op.operation) {
        case TOKEN_PLUS: {
            // atomic_add
            X86Register target_reg = codegen_expression(gen, atomic_op->atomic_op.target, gen->current_scope);
            X86Register value_reg = codegen_expression(gen, atomic_op->atomic_op.value, gen->current_scope);
            
            if (target_reg != REG_NONE && value_reg != REG_NONE) {
                Operand target_mem = operand_memory(target_reg, 0, 4);
                Operand value_op = operand_register(value_reg, 4);
                
                // 使用 lock 前缀
                fprintf(gen->output, "    lock    ");
                codegen_emit_instruction(gen, INST_XADD, &target_mem, &value_op);
                
                reg_alloc_free_reg(gen->reg_alloc, target_reg);
                reg_alloc_free_reg(gen->reg_alloc, value_reg);
            }
            break;
        }
        
        default:
            codegen_emit_comment(gen, "Unsupported atomic operation");
            break;
    }
}

void x86_emit_barrier(CodeGenerator *gen) {
    codegen_emit_comment(gen, "Memory barrier");
    codegen_emit_instruction(gen, INST_MFENCE, NULL, NULL);
}

void x86_emit_critical_section(CodeGenerator *gen, ASTNode *critical) {
    codegen_emit_comment(gen, "Critical section begin");
    
    // 简化实现：使用自旋锁
    char *lock_label = codegen_new_label(gen, ".lock_acquire");
    char *unlock_label = codegen_new_label(gen, ".lock_release");
    
    // 获取锁
    codegen_emit_label(gen, lock_label);
    fprintf(gen->output, "    mov     eax, 1\n");
    fprintf(gen->output, "    xchg    eax, [lock_variable]\n");
    fprintf(gen->output, "    test    eax, eax\n");
    x86_emit_jump(gen, INST_JNZ, lock_label);
    
    // 生成临界区代码
    codegen_statement(gen, critical->critical.body, gen->current_scope);
    
    // 释放锁
    codegen_emit_label(gen, unlock_label);
    fprintf(gen->output, "    mov     dword [lock_variable], 0\n");
    
    free(lock_label);
    free(unlock_label);
    
    codegen_emit_comment(gen, "Critical section end");
}

// =============================================================================
// 主代码生成函数
// =============================================================================

bool codegen_generate(CodeGenerator *gen, ASTNode *root) {
    if (!root || !gen) return false;
    
    // 生成文件头
    fprintf(gen->output, "; X86/X64 Parallel C Compiler Generated Assembly\n");
    fprintf(gen->output, "; Target: x86_64\n\n");
    
    fprintf(gen->output, "section .text\n");
    fprintf(gen->output, "default rel\n\n");
    
    // 生成全局锁变量 (用于critical sections)
    fprintf(gen->output, "section .bss\n");
    fprintf(gen->output, "lock_variable: resd 1\n\n");
    fprintf(gen->output, "section .text\n\n");
    
    // 生成外部函数声明
    fprintf(gen->output, "extern printf\n");
    fprintf(gen->output, "extern malloc\n");
    fprintf(gen->output, "extern free\n");
    fprintf(gen->output, "extern get_thread_id\n\n");
    
    // 生成主代码
    gen->current_scope = symbol_table_create(NULL);
    bool result = codegen_statement(gen, root, gen->current_scope);
    symbol_table_destroy(gen->current_scope);
    
    return result && !gen->has_error;
}
