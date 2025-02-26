CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99
LDFLAGS = -lm

# Source files and object files
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

# Header files
HDRS = $(wildcard *.h)

# Executable name
TARGET = neural_network

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)