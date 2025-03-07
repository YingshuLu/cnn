CC = gcc

CFLAGS = -I. -I../
LDFLAGS = -lm

HDRS = $(wildcard *.h) $(wildcard ../*.h)

SRCS = $(wildcard *.c) $(wildcard ../*.c)

OBJS = $(SRCS:.c=.o)

TARGET = mnist_test

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)
	rm -f ${OBJS}

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) mnist_test
