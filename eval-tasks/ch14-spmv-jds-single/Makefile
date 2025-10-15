# ch14-spmv-jds-single / Makefile
NVCC ?= nvcc
CFLAGS := -O2 -std=c++17

all: test_reference test_student

test_reference: test_spmv_jds.cu reference_solution.cu
	$(NVCC) $(CFLAGS) -o $@ $^

test_student: test_spmv_jds.cu student_kernel.cu
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student