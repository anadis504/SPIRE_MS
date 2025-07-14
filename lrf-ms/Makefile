CC = g++
CFLAGS = -std=c++20 -O3 -march=native -DNDEBUG -fopenmp -DLIBSAIS_OPENMP -funroll-loops -g

SRCS = libsais/src/libsais.o matching_statistics.cpp 
OBJS = $(SRCS:.cpp=.o)

TARGET = lrf_ms

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)