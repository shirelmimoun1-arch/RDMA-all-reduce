# Compiler and Flags
CC       := gcc
CFLAGS   := -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L
LDFLAGS  := -libverbs

# Target Binary and Source
TARGET   := ring_allreduce
SRC      := ring_allreduce_advanced.c
OBJ      := $(SRC:.c=.o)

# Default rule: build the binary
all: $(TARGET)

# Link object files to create the final executable
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Compile C source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)

# Run examples for quick testing
run-eager: $(TARGET)
	./$(TARGET) -myindex 01 -list localhost localhost -protocol eager -count 65536

run-rendezvous: $(TARGET)
	./$(TARGET) -myindex 01 -list localhost localhost -protocol rendezvous -count 65536

.PHONY: all clean run-eager run-rendezvous