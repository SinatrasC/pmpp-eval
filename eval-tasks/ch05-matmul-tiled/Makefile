NVCC ?= nvcc
ARCH ?= -arch=sm_70
CXXFLAGS := $(ARCH) -O2 --use_fast_math

all: test_reference test_student

test_reference: test_matmul_tiled.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

test_student: test_matmul_tiled.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student