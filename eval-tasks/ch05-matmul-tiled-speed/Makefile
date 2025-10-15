# Makefile
NVCC ?= nvcc
ARCH ?= -arch=sm_70
OPTS := -O3 $(ARCH) --use_fast_math

TILE ?= 16

.PHONY: all test run-student run-reference clean

all: test_student test_reference

student.o: student_kernel.cu
	$(NVCC) $(OPTS) -DTILE=$(TILE) -c $< -o $@

reference.o: reference_solution.cu
	$(NVCC) $(OPTS) -DTILE=$(TILE) -c $< -o $@

test_student: test_matmul_speed.cu student.o reference.o
	$(NVCC) $(OPTS) -DTILE=$(TILE) $^ -o $@

test: test_student
	./test_student

# Reference binary reuses the harness but shims matmul_student to the reference tiled kernel.
test_reference: test_matmul_speed.cu reference_solution.cu reference_student_wrapper.cu
	$(NVCC) $(OPTS) -DTILE=$(TILE) $^ -o $@

run-student: test_student
	./test_student

run-reference: test_reference
	./test_reference

clean:
	rm -f *.o test_student test_reference
