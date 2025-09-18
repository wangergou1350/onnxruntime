#include "riscv_cc.h"

int main(int argc, char *argv[]) {
    // 创建编译器实例
    Compiler *compiler = create_compiler();
    if (!compiler) {
        fprintf(stderr, "编译器初始化失败\n");
        return EXIT_FAILURE;
    }
    
    // 解析命令行参数
    int parse_result = parse_arguments(argc, argv, &compiler->options);
    if (parse_result != 0) {
        destroy_compiler(compiler);
        return parse_result == -1 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    
    // 执行编译
    bool success = compile_file(compiler, 
                               compiler->options.input_filename, 
                               compiler->options.output_filename);
    
    // 获取错误统计
    int error_count = compiler->options.error_count;
    int warning_count = compiler->options.warning_count;
    
    // 清理资源
    destroy_compiler(compiler);
    
    // 返回适当的退出码
    if (!success || error_count > 0) {
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
