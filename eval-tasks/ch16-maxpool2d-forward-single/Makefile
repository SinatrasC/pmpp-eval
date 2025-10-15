# ch16-maxpool2d-forward-single / Makefile
NVCC ?= nvcc
CFLAGS = -O2 -std=c++17 -arch=sm_70

all: test_reference test_student

test_reference: test_maxpool_forward.cu reference_solution.cu
	$(NVCC) $(CFLAGS) -o $@ $^

test_student: test_maxpool_forward.cu student_kernel.cu
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student

.PHONY: all clean