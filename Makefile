CC = gcc
CFLAGS = -Wall -Wextra -g
# -fsanitize=address
SRCDIR = .
TARGETDIR = target

# Source files
ENGINE_SRC = $(SRCDIR)/runEngine.c
MODEL_SRC = $(SRCDIR)/runModel.c
COMMON_SRCS = $(wildcard $(SRCDIR)/*.c)  # All other .c files as common sources

# Filter out specific main source files from COMMON_SRCS
COMMON_SRCS := $(filter-out $(ENGINE_SRC) $(MODEL_SRC), $(COMMON_SRCS))

# Object files
ENGINE_OBJS = $(ENGINE_SRC:.c=.o) $(COMMON_SRCS:.c=.o)
MODEL_OBJS = $(MODEL_SRC:.c=.o) $(COMMON_SRCS:.c=.o)

# Executable names
TARGET1 = $(TARGETDIR)/runEngine
TARGET2 = $(TARGETDIR)/runModel

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
