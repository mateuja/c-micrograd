CC = gcc
CFLAGS = -Wall -Wextra -g
SRCDIR = .
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(SRCS:$(SRCDIR)/%.c=%.o)
TARGETDIR = target
TARGET = $(TARGETDIR)/c-micrograd

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(TARGETDIR)
	$(CC) $(CFLAGS) $^ -o $@ -lm

$(TARGETDIR):
	mkdir -p $(TARGETDIR)

%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGETDIR) *.o

