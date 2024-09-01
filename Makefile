CC = gcc
CFLAGS = -Wall -Wextra -g
# -fsanitize=address
SRCDIR = .
TARGETDIR = target

# Source files
EXAMPLE_SRC = $(SRCDIR)/example.c
NN_SRC = $(SRCDIR)/train_nn.c
COMMON_SRCS = $(wildcard $(SRCDIR)/*.c)  # All other .c files as common sources

# Filter out specific main source files from COMMON_SRCS
COMMON_SRCS := $(filter-out $(EXAMPLE_SRC) $(NN_SRC), $(COMMON_SRCS))

# Object files
ENGINE_OBJS = $(EXAMPLE_SRC:.c=.o) $(COMMON_SRCS:.c=.o)
MODEL_OBJS = $(NN_SRC:.c=.o) $(COMMON_SRCS:.c=.o)

# Executable names
TARGET1 = $(TARGETDIR)/example
TARGET2 = $(TARGETDIR)/train_nn

.PHONY: all clean

# Build both executables
all: $(TARGET1) $(TARGET2)

# Rule for the first executable (runEngine.c)
$(TARGET1): $(ENGINE_OBJS) | $(TARGETDIR)
	$(CC) $(CFLAGS) $(ENGINE_OBJS) -o $@ -lm

# Rule for the second executable (runModel.c)
$(TARGET2): $(MODEL_OBJS) | $(TARGETDIR)
	$(CC) $(CFLAGS) $(MODEL_OBJS) -o $@ -lm

# Ensure the target directory exists
$(TARGETDIR):
	mkdir -p $(TARGETDIR)

# Rule to compile object files
%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -rf $(TARGETDIR) *.o
