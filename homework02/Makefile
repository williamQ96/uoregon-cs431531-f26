CC := gcc
CXX := g++
CFLAGS := -O3 -g -std=c99 -Wall -Wextra -fopenmp -D_POSIX_C_SOURCE=200112L
CXXFLAGS := -O3 -g -std=c++17 -Wall -Wextra -Wpedantic -fopenmp
LDLIBS := -lm

.PHONY: all test clean
all: prefix benchmark

prefix: prefix.c prefix.h common.c common.h
	$(CC) $(CFLAGS) prefix.c common.c -o $@ $(LDLIBS)

build/prefix_library.o: prefix.c prefix.h common.h
	mkdir -p build
	$(CC) $(CFLAGS) -DPREFIX_SUM_NO_MAIN -c prefix.c -o $@

benchmark: bench/benchmark.cpp prefix.h build/prefix_library.o
	$(CXX) $(CXXFLAGS) bench/benchmark.cpp build/prefix_library.o -o $@

test: all
	./benchmark --test
	OMP_NUM_THREADS=4 ./prefix 10007 5

clean:
	rm -f prefix benchmark build/prefix_library.o
