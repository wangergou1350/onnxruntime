/*
 * X86/X64 并行 C 编译器 - 主程序入口
 * 提供命令行接口和编译器主逻辑
 */

#include "x86_cc.h"

int main(int argc, char **argv) {
    return compile_main(argc, argv);
}
