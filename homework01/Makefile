CC = gcc
CFLAGS ?= -O3 -g -std=c99 -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp
LDLIBS ?= -lm
PYTHON ?= python3

.PHONY: all clean test report
all: pi

pi: pi.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

test: pi
	$(PYTHON) tests/test_pi.py --exe ./pi

report:
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(RM) pi pi.exe pi.o common.o
