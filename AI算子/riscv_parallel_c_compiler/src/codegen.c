#include "riscv_cc.h"

// =============================================================================
// RISC-V代码生成器实现
// =============================================================================

// RISC-V寄存器名称映射
static const char *register_names[] = {
    // 整数寄存器
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
    
    // 浮点寄存器
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
    "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
    "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"
};

// RISC-V指令名称映射
static const char *instruction_names[] = {
    // RV32I基础指令集
    "lui", "auipc", "jal", "jalr", "beq", "bne", "blt", "bge",
    "bltu", "bgeu", "lb", "lh", "lw", "lbu", "lhu", "sb", "sh", "sw",
    "addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai",
    "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and",
    
    // RV64I
    "lwu", "ld", "sd", "addiw", "slliw", "srliw", "sraiw",
    "addw", "subw", "sllw", "srlw", "sraw",
    
    // RV32M乘除法扩展
    "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
    
    // RV64M
    "mulw", "divw", "divuw", "remw", "remuw",
    
    // RV32A原子指令扩展
    "lr.w", "sc.w", "amoswap.w", "amoadd.w", "amoxor.w", "amoand.w",
    "amoor.w", "amomin.w", "amomax.w", "amominu.w", "amomaxu.w",
    
    // RV64A
    "lr.d", "sc.d", "amoswap.d", "amoadd.d", "amoxor.d", "amoand.d",
    "amoor.d", "amomin.d", "amomax.d", "amominu.d", "amomaxu.d",
    
    // RV32F单精度浮点
    "flw", "fsw", "fmadd.s", "fmsub.s", "fnmsub.s", "fnmadd.s",
    "fadd.s", "fsub.s", "fmul.s", "fdiv.s", "fsqrt.s",
    
    // RV32D双精度浮点
    "fld", "fsd", "fmadd.d", "fmsub.d", "fnmsub.d", "fnmadd.d",
    "fadd.d", "fsub.d", "fmul.d", "fdiv.d", "fsqrt.d",
    
    // 伪指令
    "nop", "mv", "not", "neg", "ret", "j", "call", "tail", "li", "la",
    
    // 并行和同步指令
    "fence", "fence.i"
};

// =============================================================================
// 代码生成器创建和销毁
// =============================================================================

CodeGenerator *create_code_generator(bool riscv64) {
    CodeGenerator *codegen = malloc(sizeof(CodeGenerator));
    if (!codegen) {
        error("内存分配失败：无法创建代码生成器");
        return NULL;
    }
    
    memset(codegen, 0, sizeof(CodeGenerator));
    
    // 初始化寄存器分配状态
    for (int i = 0; i < REG_COUNT; i++) {
        codegen->register_used[i] = false;
    }
    
    // 保留系统寄存器
    codegen->register_used[REG_ZERO] = true;  // 硬编码为0
    codegen->register_used[REG_RA] = true;    // 返回地址
    codegen->register_used[REG_SP] = true;    // 栈指针
    codegen->register_used[REG_GP] = true;    // 全局指针
    codegen->register_used[REG_TP] = true;    // 线程指针
    
    codegen->next_temp_reg = REG_T0;          // 临时寄存器从t0开始
    codegen->next_temp_freg = REG_FT0;        // 临时浮点寄存器从ft0开始
    
    codegen->stack_offset = 0;
    codegen->max_stack_size = 0;
    codegen->next_label_id = 1;
    
    // 优化设置
    codegen->optimize_registers = true;
    codegen->optimize_instructions = true;
    codegen->optimize_parallel = true;
    
    codegen->instructions = NULL;
    codegen->last_instruction = NULL;
    codegen->instruction_count = 0;
    
    return codegen;
}

void destroy_code_generator(CodeGenerator *codegen) {
    if (!codegen) return;
    
    // 释放指令链表
    Instruction *inst = codegen->instructions;
    while (inst) {
        Instruction *next = inst->next;
        free(inst->comment);
        free(inst);
        inst = next;
    }
    
    free(codegen);
}

// =============================================================================
// 寄存器管理
// =============================================================================

RiscVRegister allocate_register(CodeGenerator *codegen, bool is_float) {
    int start_reg = is_float ? REG_FT0 : REG_T0;
    int end_reg = is_float ? REG_FT11 : REG_T6;
    
    // 首先尝试临时寄存器
    for (int i = start_reg; i <= end_reg; i++) {
        if (!codegen->register_used[i]) {
            codegen->register_used[i] = true;
            return i;
        }
    }
    
    // 如果临时寄存器用完，尝试保存寄存器
    if (!is_float) {
        for (int i = REG_S2; i <= REG_S11; i++) {
            if (!codegen->register_used[i]) {
                codegen->register_used[i] = true;
                return i;
            }
        }
    } else {
        for (int i = REG_FS2; i <= REG_FS11; i++) {
            if (!codegen->register_used[i]) {
                codegen->register_used[i] = true;
                return i;
            }
        }
    }
    
    error("寄存器分配失败：所有寄存器都已被使用");
    return REG_NONE;
}

void free_register(CodeGenerator *codegen, RiscVRegister reg) {
    if (reg != REG_NONE && reg < REG_COUNT) {
        codegen->register_used[reg] = false;
    }
}

// =============================================================================
// 指令生成
// =============================================================================

static Instruction *create_instruction(RiscVInstruction opcode) {
    Instruction *inst = malloc(sizeof(Instruction));
    if (!inst) {
        error("内存分配失败：无法创建指令");
        return NULL;
    }
    
    memset(inst, 0, sizeof(Instruction));
    inst->opcode = opcode;
    inst->operand_count = 0;
    inst->comment = NULL;
    inst->next = NULL;
    
    return inst;
}

static void add_operand(Instruction *inst, OperandType type, ...) {
    if (inst->operand_count >= 3) {
        error("指令操作数过多");
        return;
    }
    
    va_list args;
    va_start(args, type);
    
    Operand *op = &inst->operands[inst->operand_count++];
    op->type = type;
    
    switch (type) {
        case OPERAND_REGISTER:
            op->reg = va_arg(args, RiscVRegister);
            break;
        case OPERAND_IMMEDIATE:
            op->immediate = va_arg(args, int);
            break;
        case OPERAND_LABEL:
            op->label = strdup(va_arg(args, char*));
            break;
        case OPERAND_MEMORY:
            op->memory.base = va_arg(args, RiscVRegister);
            op->memory.offset = va_arg(args, int);
            break;
        case OPERAND_OFFSET:
            op->immediate = va_arg(args, int);
            break;
    }
    
    va_end(args);
}

void emit_instruction(CodeGenerator *codegen, RiscVInstruction opcode, ...) {
    Instruction *inst = create_instruction(opcode);
    if (!inst) return;
    
    va_list args;
    va_start(args, opcode);
    
    // 根据指令类型解析操作数
    switch (opcode) {
        // R-type指令 (rd, rs1, rs2)
        case INST_ADD:
        case INST_SUB:
        case INST_SLL:
        case INST_SLT:
        case INST_SLTU:
        case INST_XOR:
        case INST_SRL:
        case INST_SRA:
        case INST_OR:
        case INST_AND:
        case INST_MUL:
        case INST_DIV:
        case INST_REM:
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rd
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rs1
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rs2
            break;
            
        // I-type指令 (rd, rs1, imm)
        case INST_ADDI:
        case INST_SLTI:
        case INST_SLTIU:
        case INST_XORI:
        case INST_ORI:
        case INST_ANDI:
        case INST_SLLI:
        case INST_SRLI:
        case INST_SRAI:
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rd
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rs1
            add_operand(inst, OPERAND_IMMEDIATE, va_arg(args, int));          // imm
            break;
            
        // Load指令 (rd, offset(rs1))
        case INST_LB:
        case INST_LH:
        case INST_LW:
        case INST_LD:
        case INST_LBU:
        case INST_LHU:
        case INST_LWU:
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rd
            add_operand(inst, OPERAND_MEMORY, va_arg(args, RiscVRegister), va_arg(args, int)); // offset(rs1)
            break;
            
        // Store指令 (rs2, offset(rs1))
        case INST_SB:
        case INST_SH:
        case INST_SW:
        case INST_SD:
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rs2
            add_operand(inst, OPERAND_MEMORY, va_arg(args, RiscVRegister), va_arg(args, int)); // offset(rs1)
            break;
            
        // Branch指令 (rs1, rs2, label)
        case INST_BEQ:
        case INST_BNE:
        case INST_BLT:
        case INST_BGE:
        case INST_BLTU:
        case INST_BGEU:
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rs1
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rs2
            add_operand(inst, OPERAND_LABEL, va_arg(args, char*));           // label
            break;
            
        // U-type指令 (rd, imm)
        case INST_LUI:
        case INST_AUIPC:
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rd
            add_operand(inst, OPERAND_IMMEDIATE, va_arg(args, int));          // imm
            break;
            
        // J-type指令 (rd, label)
        case INST_JAL:
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rd
            add_operand(inst, OPERAND_LABEL, va_arg(args, char*));           // label
            break;
            
        // 伪指令
        case INST_MV: // mv rd, rs
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rd
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rs
            break;
            
        case INST_LI: // li rd, imm
            add_operand(inst, OPERAND_REGISTER, va_arg(args, RiscVRegister)); // rd
            add_operand(inst, OPERAND_IMMEDIATE, va_arg(args, int));          // imm
            break;
            
        case INST_J: // j label
            add_operand(inst, OPERAND_LABEL, va_arg(args, char*));           // label
            break;
            
        case INST_RET: // ret (无操作数)
        case INST_NOP: // nop (无操作数)
            break;
            
        default:
            warning("未处理的指令类型: %d", opcode);
            break;
    }
    
    va_end(args);
    
    // 添加到指令链表
    if (codegen->last_instruction) {
        codegen->last_instruction->next = inst;
    } else {
        codegen->instructions = inst;
    }
    codegen->last_instruction = inst;
    codegen->instruction_count++;
}

char *generate_label(CodeGenerator *codegen) {
    char *label = malloc(32);
    sprintf(label, ".L%d", codegen->next_label_id++);
    return label;
}

// =============================================================================
// 表达式代码生成
// =============================================================================

static RiscVRegister generate_expression_code(CodeGenerator *codegen, ASTNode *node) {
    if (!node) return REG_NONE;
    
    switch (node->type) {
        case AST_NUMBER_LITERAL: {
            RiscVRegister reg = allocate_register(codegen, false);
            emit_instruction(codegen, INST_LI, reg, node->literal.int_value);
            return reg;
        }
        
        case AST_FLOAT_LITERAL: {
            RiscVRegister reg = allocate_register(codegen, true);
            // 浮点常量加载需要更复杂的处理
            // 这里简化为加载到内存然后载入
            warning("浮点常量加载未完全实现");
            return reg;
        }
        
        case AST_CHAR_LITERAL: {
            RiscVRegister reg = allocate_register(codegen, false);
            emit_instruction(codegen, INST_LI, reg, (int)node->literal.char_value);
            return reg;
        }
        
        case AST_STRING_LITERAL: {
            RiscVRegister reg = allocate_register(codegen, false);
            // 字符串常量需要在数据段中分配
            // 这里简化为返回地址
            warning("字符串常量加载未完全实现");
            return reg;
        }
        
        case AST_IDENTIFIER: {
            // 加载变量值
            RiscVRegister reg = allocate_register(codegen, false);
            // 简化：假设变量在栈上
            emit_instruction(codegen, INST_LW, reg, REG_SP, -8); // 示例偏移
            return reg;
        }
        
        case AST_BINARY_EXPR: {
            RiscVRegister left_reg = generate_expression_code(codegen, node->binary_expr.left);
            RiscVRegister right_reg = generate_expression_code(codegen, node->binary_expr.right);
            RiscVRegister result_reg = allocate_register(codegen, false);
            
            switch (node->binary_expr.operator) {
                case TOKEN_PLUS:
                    emit_instruction(codegen, INST_ADD, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_MINUS:
                    emit_instruction(codegen, INST_SUB, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_MULTIPLY:
                    emit_instruction(codegen, INST_MUL, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_DIVIDE:
                    emit_instruction(codegen, INST_DIV, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_MODULO:
                    emit_instruction(codegen, INST_REM, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_BITWISE_AND:
                    emit_instruction(codegen, INST_AND, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_BITWISE_OR:
                    emit_instruction(codegen, INST_OR, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_BITWISE_XOR:
                    emit_instruction(codegen, INST_XOR, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_LSHIFT:
                    emit_instruction(codegen, INST_SLL, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_RSHIFT:
                    emit_instruction(codegen, INST_SRL, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_LESS:
                    emit_instruction(codegen, INST_SLT, result_reg, left_reg, right_reg);
                    break;
                case TOKEN_EQUAL:
                    // eq可以用xor然后sltiu实现
                    emit_instruction(codegen, INST_XOR, result_reg, left_reg, right_reg);
                    emit_instruction(codegen, INST_SLTIU, result_reg, result_reg, 1);
                    break;
                default:
                    warning("未支持的二元运算符: %d", node->binary_expr.operator);
                    break;
            }
            
            free_register(codegen, left_reg);
            free_register(codegen, right_reg);
            return result_reg;
        }
        
        case AST_UNARY_EXPR: {
            RiscVRegister operand_reg = generate_expression_code(codegen, node->unary_expr.operand);
            RiscVRegister result_reg = allocate_register(codegen, false);
            
            switch (node->unary_expr.operator) {
                case TOKEN_MINUS:
                    emit_instruction(codegen, INST_SUB, result_reg, REG_ZERO, operand_reg);
                    break;
                case TOKEN_BITWISE_NOT:
                    emit_instruction(codegen, INST_XORI, result_reg, operand_reg, -1);
                    break;
                case TOKEN_LOGICAL_NOT:
                    emit_instruction(codegen, INST_SLTIU, result_reg, operand_reg, 1);
                    break;
                default:
                    warning("未支持的一元运算符: %d", node->unary_expr.operator);
                    break;
            }
            
            free_register(codegen, operand_reg);
            return result_reg;
        }
        
        case AST_ASSIGN_EXPR: {
            RiscVRegister value_reg = generate_expression_code(codegen, node->assign_expr.right);
            
            // 简化：假设左值是变量
            if (node->assign_expr.left->type == AST_IDENTIFIER) {
                emit_instruction(codegen, INST_SW, value_reg, REG_SP, -8); // 示例偏移
            }
            
            return value_reg;
        }
        
        case AST_CALL_EXPR: {
            // 函数调用
            // 1. 保存caller-saved寄存器
            // 2. 准备参数
            // 3. 调用函数
            // 4. 恢复寄存器
            
            // 准备参数（最多8个寄存器参数）
            for (int i = 0; i < node->call_expr.argument_count && i < 8; i++) {
                RiscVRegister arg_reg = generate_expression_code(codegen, node->call_expr.arguments[i]);
                RiscVRegister param_reg = REG_A0 + i;
                
                if (arg_reg != param_reg) {
                    emit_instruction(codegen, INST_MV, param_reg, arg_reg);
                }
                free_register(codegen, arg_reg);
            }
            
            // 调用函数
            if (node->call_expr.function->type == AST_IDENTIFIER) {
                char *func_name = node->call_expr.function->identifier.name;
                emit_instruction(codegen, INST_CALL, func_name);
            }
            
            // 返回值在a0中
            RiscVRegister result_reg = allocate_register(codegen, false);
            emit_instruction(codegen, INST_MV, result_reg, REG_A0);
            return result_reg;
        }
        
        default:
            warning("未支持的表达式类型: %d", node->type);
            return REG_NONE;
    }
}

// =============================================================================
// 语句代码生成
// =============================================================================

static void generate_statement_code(CodeGenerator *codegen, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_COMPOUND_STMT:
            for (int i = 0; i < node->compound_stmt.statement_count; i++) {
                generate_statement_code(codegen, node->compound_stmt.statements[i]);
            }
            break;
            
        case AST_EXPRESSION_STMT:
            if (node->type != AST_COMPOUND_STMT) {
                RiscVRegister reg = generate_expression_code(codegen, node);
                if (reg != REG_NONE) {
                    free_register(codegen, reg);
                }
            }
            break;
            
        case AST_IF_STMT: {
            char *else_label = generate_label(codegen);
            char *end_label = generate_label(codegen);
            
            // 计算条件
            RiscVRegister cond_reg = generate_expression_code(codegen, node->if_stmt.condition);
            
            // 条件为假时跳转到else分支
            emit_instruction(codegen, INST_BEQ, cond_reg, REG_ZERO, else_label);
            free_register(codegen, cond_reg);
            
            // then分支
            generate_statement_code(codegen, node->if_stmt.then_stmt);
            emit_instruction(codegen, INST_J, end_label);
            
            // else分支标签
            // 这里需要输出标签，简化处理
            
            // else分支
            if (node->if_stmt.else_stmt) {
                generate_statement_code(codegen, node->if_stmt.else_stmt);
            }
            
            // 结束标签
            
            free(else_label);
            free(end_label);
            break;
        }
        
        case AST_WHILE_STMT: {
            char *loop_label = generate_label(codegen);
            char *end_label = generate_label(codegen);
            
            // 循环开始标签
            
            // 计算条件
            RiscVRegister cond_reg = generate_expression_code(codegen, node->while_stmt.condition);
            
            // 条件为假时跳出循环
            emit_instruction(codegen, INST_BEQ, cond_reg, REG_ZERO, end_label);
            free_register(codegen, cond_reg);
            
            // 循环体
            generate_statement_code(codegen, node->while_stmt.body);
            
            // 跳回循环开始
            emit_instruction(codegen, INST_J, loop_label);
            
            // 循环结束标签
            
            free(loop_label);
            free(end_label);
            break;
        }
        
        case AST_FOR_STMT: {
            char *loop_label = generate_label(codegen);
            char *continue_label = generate_label(codegen);
            char *end_label = generate_label(codegen);
            
            // 初始化
            if (node->for_stmt.init) {
                RiscVRegister reg = generate_expression_code(codegen, node->for_stmt.init);
                if (reg != REG_NONE) free_register(codegen, reg);
            }
            
            // 循环开始标签
            
            // 条件检查
            if (node->for_stmt.condition) {
                RiscVRegister cond_reg = generate_expression_code(codegen, node->for_stmt.condition);
                emit_instruction(codegen, INST_BEQ, cond_reg, REG_ZERO, end_label);
                free_register(codegen, cond_reg);
            }
            
            // 循环体
            generate_statement_code(codegen, node->for_stmt.body);
            
            // 递增表达式
            if (node->for_stmt.increment) {
                RiscVRegister reg = generate_expression_code(codegen, node->for_stmt.increment);
                if (reg != REG_NONE) free_register(codegen, reg);
            }
            
            // 跳回循环开始
            emit_instruction(codegen, INST_J, loop_label);
            
            // 循环结束标签
            
            free(loop_label);
            free(continue_label);
            free(end_label);
            break;
        }
        
        case AST_PARALLEL_FOR_STMT: {
            // 并行for循环代码生成
            // 这里需要生成多线程代码，简化处理
            warning("并行for循环代码生成未完全实现");
            
            // 暂时按普通for循环处理
            char *loop_label = generate_label(codegen);
            char *end_label = generate_label(codegen);
            
            if (node->parallel_for_stmt.init) {
                RiscVRegister reg = generate_expression_code(codegen, node->parallel_for_stmt.init);
                if (reg != REG_NONE) free_register(codegen, reg);
            }
            
            if (node->parallel_for_stmt.condition) {
                RiscVRegister cond_reg = generate_expression_code(codegen, node->parallel_for_stmt.condition);
                emit_instruction(codegen, INST_BEQ, cond_reg, REG_ZERO, end_label);
                free_register(codegen, cond_reg);
            }
            
            generate_statement_code(codegen, node->parallel_for_stmt.body);
            
            if (node->parallel_for_stmt.increment) {
                RiscVRegister reg = generate_expression_code(codegen, node->parallel_for_stmt.increment);
                if (reg != REG_NONE) free_register(codegen, reg);
            }
            
            emit_instruction(codegen, INST_J, loop_label);
            
            free(loop_label);
            free(end_label);
            break;
        }
        
        case AST_RETURN_STMT: {
            if (node->return_stmt.value) {
                RiscVRegister value_reg = generate_expression_code(codegen, node->return_stmt.value);
                if (value_reg != REG_A0) {
                    emit_instruction(codegen, INST_MV, REG_A0, value_reg);
                }
                free_register(codegen, value_reg);
            }
            emit_instruction(codegen, INST_RET);
            break;
        }
        
        case AST_BREAK_STMT:
            // 需要跳转到循环结束标签
            warning("break语句需要上下文信息");
            break;
            
        case AST_CONTINUE_STMT:
            // 需要跳转到循环继续标签
            warning("continue语句需要上下文信息");
            break;
            
        case AST_BARRIER_STMT:
            // 内存屏障
            emit_instruction(codegen, INST_FENCE);
            break;
            
        default:
            // 其他语句类型按表达式处理
            RiscVRegister reg = generate_expression_code(codegen, node);
            if (reg != REG_NONE) {
                free_register(codegen, reg);
            }
            break;
    }
}

// =============================================================================
// 函数代码生成
// =============================================================================

static void generate_function_code(CodeGenerator *codegen, ASTNode *node) {
    if (!node || node->type != AST_FUNCTION_DEF) return;
    
    // 函数序言
    emit_instruction(codegen, INST_ADDI, REG_SP, REG_SP, -16); // 为局部变量预留空间
    emit_instruction(codegen, INST_SW, REG_RA, REG_SP, 12);    // 保存返回地址
    emit_instruction(codegen, INST_SW, REG_S0, REG_SP, 8);     // 保存帧指针
    emit_instruction(codegen, INST_ADDI, REG_S0, REG_SP, 16);  // 设置帧指针
    
    // 函数体
    if (node->function_def.body) {
        generate_statement_code(codegen, node->function_def.body);
    }
    
    // 函数结语（如果没有显式return）
    emit_instruction(codegen, INST_LI, REG_A0, 0);           // 默认返回0
    emit_instruction(codegen, INST_LW, REG_RA, REG_SP, 12);  // 恢复返回地址
    emit_instruction(codegen, INST_LW, REG_S0, REG_SP, 8);   // 恢复帧指针
    emit_instruction(codegen, INST_ADDI, REG_SP, REG_SP, 16); // 恢复栈指针
    emit_instruction(codegen, INST_RET);                     // 返回
}

// =============================================================================
// 主代码生成函数
// =============================================================================

bool generate_code(CodeGenerator *codegen, ASTNode *ast, FILE *output) {
    if (!codegen || !ast || !output) {
        return false;
    }
    
    if (ast->type != AST_PROGRAM) {
        error("根节点必须是程序节点");
        return false;
    }
    
    // 生成汇编文件头部
    fprintf(output, ".text\n");
    fprintf(output, ".globl main\n\n");
    
    // 遍历所有声明，生成代码
    for (int i = 0; i < ast->program.declaration_count; i++) {
        ASTNode *decl = ast->program.declarations[i];
        
        if (decl->type == AST_FUNCTION_DEF) {
            // 函数标签
            fprintf(output, "%s:\n", decl->function_def.name);
            
            // 生成函数代码
            generate_function_code(codegen, decl);
            
            // 输出指令
            Instruction *inst = codegen->instructions;
            while (inst) {
                fprintf(output, "\t%s", instruction_names[inst->opcode]);
                
                // 输出操作数
                for (int j = 0; j < inst->operand_count; j++) {
                    if (j > 0) fprintf(output, ",");
                    fprintf(output, " ");
                    
                    Operand *op = &inst->operands[j];
                    switch (op->type) {
                        case OPERAND_REGISTER:
                            fprintf(output, "%s", register_names[op->reg]);
                            break;
                        case OPERAND_IMMEDIATE:
                            fprintf(output, "%d", op->immediate);
                            break;
                        case OPERAND_LABEL:
                            fprintf(output, "%s", op->label);
                            break;
                        case OPERAND_MEMORY:
                            fprintf(output, "%d(%s)", op->memory.offset, 
                                   register_names[op->memory.base]);
                            break;
                        case OPERAND_OFFSET:
                            fprintf(output, "%d", op->immediate);
                            break;
                    }
                }
                
                if (inst->comment) {
                    fprintf(output, "\t# %s", inst->comment);
                }
                fprintf(output, "\n");
                
                inst = inst->next;
            }
            
            fprintf(output, "\n");
            
            // 清空指令列表，准备下一个函数
            inst = codegen->instructions;
            while (inst) {
                Instruction *next = inst->next;
                free(inst->comment);
                free(inst);
                inst = next;
            }
            codegen->instructions = NULL;
            codegen->last_instruction = NULL;
            codegen->instruction_count = 0;
        }
    }
    
    return true;
}

// =============================================================================
// 工具函数实现
// =============================================================================

const char *register_to_string(RiscVRegister reg) {
    if (reg >= 0 && reg < REG_COUNT) {
        return register_names[reg];
    }
    return "unknown";
}

// =============================================================================
// 优化函数（简化版）
// =============================================================================

void optimize_instructions(CodeGenerator *codegen) {
    if (!codegen->optimize_instructions) return;
    
    // 简单的窥孔优化
    Instruction *inst = codegen->instructions;
    while (inst && inst->next) {
        // 删除冗余的move指令
        if (inst->opcode == INST_MV && 
            inst->operands[0].reg == inst->operands[1].reg) {
            // 删除这条指令
            // 这里需要更复杂的链表操作
        }
        
        inst = inst->next;
    }
}

void optimize_registers(CodeGenerator *codegen) {
    if (!codegen->optimize_registers) return;
    
    // 寄存器分配优化
    // 这里可以实现图着色算法等
    warning("寄存器优化未实现");
}

void optimize_parallel(CodeGenerator *codegen) {
    if (!codegen->optimize_parallel) return;
    
    // 并行代码优化
    warning("并行优化未实现");
}
