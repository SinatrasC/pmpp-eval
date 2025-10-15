# Build both student and reference tests
# Usage:
#   make test_student
#   make test_reference
#   make clean

NVCC ?= nvcc
GPU_ARCH ?= sm_70
CFLAGS = -O2 -std=c++17 -arch=$(GPU_ARCH)

# Student build
test_student: test_conv2d.o student_kernel.o
	$(NVCC) $(CFLAGS) -o $@ $^

student_kernel.o: student_kernel.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

# Reference build
test_reference: test_conv2d.o reference_solution.o
	$(NVCC) $(CFLAGS) -o $@ $^

reference_solution.o: reference_solution.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

test_conv2d.o: test_conv2d.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o test_student test_reference