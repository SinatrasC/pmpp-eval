# ch14-coo-to-csr-single / Makefile
NVCC ?= nvcc
CFLAGS := -O2 -std=c++17

all: test_reference test_student

test_reference: test_coo_to_csr.cu reference_solution.cu
	$(NVCC) $(CFLAGS) -o $@ $^

test_student: test_coo_to_csr.cu student_kernel.cu
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student