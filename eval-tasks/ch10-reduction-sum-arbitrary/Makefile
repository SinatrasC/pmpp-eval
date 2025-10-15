# Makefile
NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17

all: test_reference test_student

test_reference: test_sum_arbitrary.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

test_student: test_sum_arbitrary.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student