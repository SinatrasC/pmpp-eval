# Makefile for ch03-ex1b-matmul-col-per-thread

NVCC = nvcc
CUDA_FLAGS = -O2 -std=c++17 -Xcompiler -Wall

# Default target: build and run student implementation
all: test_student

# Reference implementation
test_reference: test_matmul_col.cu reference_solution.cu
	$(NVCC) $(CUDA_FLAGS) -o test_reference test_matmul_col.cu reference_solution.cu

# Student implementation
test_student: test_matmul_col.cu student_kernel.cu
	$(NVCC) $(CUDA_FLAGS) -o test_student test_matmul_col.cu student_kernel.cu

# Run student test
run_student: test_student
	./test_student

# Run reference test
run_reference: test_reference
	./test_reference

# Clean build artifacts
clean:
	rm -f test_student test_reference

.PHONY: all run_student run_reference clean