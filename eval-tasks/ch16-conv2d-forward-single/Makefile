# ch16-conv2d-forward-single / Makefile
NVCC ?= nvcc
CFLAGS = -O2 -std=c++17 -arch=sm_70

all: test_reference test_student

test_reference: test_conv2d_forward.cu reference_solution.cu
	$(NVCC) $(CFLAGS) -o $@ $^

test_student: test_conv2d_forward.cu student_kernel.cu
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student

.PHONY: all clean