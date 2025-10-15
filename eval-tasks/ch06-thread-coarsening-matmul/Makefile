NVCC      ?= nvcc
ARCH      ?= -arch=sm_70
CXXFLAGS  ?= -O2 -std=c++17 $(ARCH)
DEFS      ?= -DTILE_WIDTH=16 -DCOARSE_FACTOR=4

all: test_reference test_student

test_reference: test_matmul_coarsened.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) $(DEFS) -o $@ $^

test_student: test_matmul_coarsened.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) $(DEFS) -DUSE_STUDENT -o $@ $^

clean:
	rm -f test_reference test_student *.o