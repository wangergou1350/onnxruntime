# Makefile for ParallelC Compiler

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -pthread
TARGET = pcc
SRCDIR = src
OBJDIR = obj
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Default target
all: $(TARGET)

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile source files to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Test the compiler
test: $(TARGET)
	@echo "Running compiler tests..."
	./$(TARGET) examples/vector_add.c -o test_output
	./test_output
	@echo "Tests completed successfully!"

# Install (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

.PHONY: all clean test install
