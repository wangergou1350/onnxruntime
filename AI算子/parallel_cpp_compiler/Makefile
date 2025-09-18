# Parallel C++ Compiler Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
LDFLAGS = -lpthread
TARGET = pcpp
SRCDIR = src
OBJDIR = obj
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Default target
all: $(TARGET)

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(TARGET)

# Clean build files
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Install (copy to system directory)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Run tests
test: $(TARGET)
	@echo "Running basic tests..."
	@./test_compiler.sh

# Debug build
debug: CFLAGS += -DDEBUG -g3
debug: $(TARGET)

# Release build
release: CFLAGS += -DNDEBUG -O3
release: clean $(TARGET)

# Format code
format:
	clang-format -i $(SRCDIR)/*.c $(SRCDIR)/*.h

# Check memory leaks
valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET) examples/simple.cpp

# Generate documentation
docs:
	doxygen Doxyfile

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build the compiler (default)"
	@echo "  clean    - Remove build files"
	@echo "  install  - Install to system directory"
	@echo "  test     - Run tests"
	@echo "  debug    - Build debug version"
	@echo "  release  - Build optimized release version"
	@echo "  format   - Format source code"
	@echo "  valgrind - Check for memory leaks"
	@echo "  docs     - Generate documentation"
	@echo "  help     - Show this help"

# Phony targets
.PHONY: all clean install uninstall test debug release format valgrind docs help

# Dependencies
$(OBJDIR)/main.o: $(SRCDIR)/main.c $(SRCDIR)/pcpp.h
$(OBJDIR)/lexer.o: $(SRCDIR)/lexer.c $(SRCDIR)/pcpp.h
$(OBJDIR)/parser.o: $(SRCDIR)/parser.c $(SRCDIR)/pcpp.h
$(OBJDIR)/semantic.o: $(SRCDIR)/semantic.c $(SRCDIR)/pcpp.h
$(OBJDIR)/codegen.o: $(SRCDIR)/codegen.c $(SRCDIR)/pcpp.h
