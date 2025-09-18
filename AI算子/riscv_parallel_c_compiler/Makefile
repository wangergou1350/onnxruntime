# RISC-V并行C编译器构建系统
# 作者: AI助手
# 版本: 1.0.0

# 编译器设置
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -Isrc
LDFLAGS = 
LIBS = 

# 目录设置
SRC_DIR = src
BUILD_DIR = build
EXAMPLES_DIR = examples
TESTS_DIR = tests

# 源文件
SOURCES = $(SRC_DIR)/main.c \
          $(SRC_DIR)/lexer.c \
          $(SRC_DIR)/parser.c \
          $(SRC_DIR)/semantic.c \
          $(SRC_DIR)/codegen.c \
          $(SRC_DIR)/utils.c

# 目标文件
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# 主目标
TARGET = riscv-cc

# 默认目标
.PHONY: all
all: $(TARGET)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译目标文件
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo "编译 $<"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 链接主程序
$(TARGET): $(OBJECTS)
	@echo "链接 $@"
	$(CC) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $@
	@echo "构建完成: $@"

# 清理
.PHONY: clean
clean:
	@echo "清理构建文件"
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)
	rm -f $(EXAMPLES_DIR)/*.s
	rm -f $(TESTS_DIR)/*.s

# 安装
.PHONY: install
install: $(TARGET)
	@echo "安装到 /usr/local/bin"
	sudo cp $(TARGET) /usr/local/bin/
	sudo chmod +x /usr/local/bin/$(TARGET)

# 卸载
.PHONY: uninstall
uninstall:
	@echo "从 /usr/local/bin 卸载"
	sudo rm -f /usr/local/bin/$(TARGET)

# 运行测试
.PHONY: test
test: $(TARGET)
	@echo "运行测试用例"
	@for test_file in $(TESTS_DIR)/*.c; do \
		if [ -f "$$test_file" ]; then \
			echo "测试: $$test_file"; \
			./$(TARGET) "$$test_file" || echo "测试失败: $$test_file"; \
		fi; \
	done

# 构建示例
.PHONY: examples
examples: $(TARGET)
	@echo "编译示例程序"
	@for example in $(EXAMPLES_DIR)/*.c; do \
		if [ -f "$$example" ]; then \
			echo "编译示例: $$example"; \
			./$(TARGET) "$$example" || echo "示例编译失败: $$example"; \
		fi; \
	done

# 调试版本
.PHONY: debug
debug: CFLAGS += -DDEBUG -O0
debug: $(TARGET)

# 发布版本
.PHONY: release
release: CFLAGS += -DNDEBUG -O3
release: clean $(TARGET)

# 代码检查
.PHONY: check
check:
	@echo "运行代码检查"
	cppcheck --enable=all --std=c99 $(SRC_DIR)/ || echo "cppcheck未安装"

# 代码格式化
.PHONY: format
format:
	@echo "格式化代码"
	@for file in $(SOURCES) $(SRC_DIR)/*.h; do \
		if [ -f "$$file" ]; then \
			clang-format -i "$$file" || echo "clang-format未安装"; \
		fi; \
	done

# 生成文档
.PHONY: docs
docs:
	@echo "生成文档"
	doxygen Doxyfile || echo "doxygen未安装"

# 显示帮助
.PHONY: help
help:
	@echo "RISC-V并行C编译器构建系统"
	@echo ""
	@echo "可用目标:"
	@echo "  all        - 构建编译器 (默认)"
	@echo "  clean      - 清理构建文件"
	@echo "  install    - 安装到系统"
	@echo "  uninstall  - 从系统卸载"
	@echo "  test       - 运行测试用例"
	@echo "  examples   - 编译示例程序"
	@echo "  debug      - 构建调试版本"
	@echo "  release    - 构建发布版本"
	@echo "  check      - 运行代码检查"
	@echo "  format     - 格式化代码"
	@echo "  docs       - 生成文档"
	@echo "  help       - 显示此帮助"
	@echo ""
	@echo "使用示例:"
	@echo "  make                # 构建编译器"
	@echo "  make clean          # 清理文件"
	@echo "  make test           # 运行测试"
	@echo "  make examples       # 编译示例"
	@echo "  make install        # 安装编译器"

# 依赖关系
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c $(SRC_DIR)/riscv_cc.h
$(BUILD_DIR)/lexer.o: $(SRC_DIR)/lexer.c $(SRC_DIR)/riscv_cc.h
$(BUILD_DIR)/parser.o: $(SRC_DIR)/parser.c $(SRC_DIR)/riscv_cc.h
$(BUILD_DIR)/semantic.o: $(SRC_DIR)/semantic.c $(SRC_DIR)/riscv_cc.h
$(BUILD_DIR)/codegen.o: $(SRC_DIR)/codegen.c $(SRC_DIR)/riscv_cc.h
$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.c $(SRC_DIR)/riscv_cc.h
