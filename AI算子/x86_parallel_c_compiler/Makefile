# X86/X64 并行 C 编译器 Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -Iinclude
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# 源文件
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/x86cc

# 默认目标
all: $(TARGET)

# 创建目录
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# 编译目标文件
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 链接生成可执行文件
$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(OBJECTS) -o $@

# 清理
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# 安装
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# 卸载
uninstall:
	rm -f /usr/local/bin/x86cc

# 测试
test: $(TARGET)
	@echo "Running compiler tests..."
	@cd tests && ./run_tests.sh

# 调试版本
debug: CFLAGS += -DDEBUG -O0
debug: $(TARGET)

# 发布版本
release: CFLAGS += -DNDEBUG -O3 -flto
release: clean $(TARGET)

# 代码格式化
format:
	find src include -name "*.c" -o -name "*.h" | xargs clang-format -i

# 静态分析
analyze:
	cppcheck --enable=all --std=c99 src/

# 生成文档
docs:
	doxygen Doxyfile

# 显示帮助
help:
	@echo "Available targets:"
	@echo "  all      - Build the compiler (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  test     - Run test suite"
	@echo "  debug    - Build debug version"
	@echo "  release  - Build optimized release version"
	@echo "  format   - Format source code"
	@echo "  analyze  - Run static analysis"
	@echo "  docs     - Generate documentation"

.PHONY: all clean install uninstall test debug release format analyze docs help
