NVCC ?= nvcc
CXXFLAGS := -O2 -std=c++17

all: test_student test_reference

test_student: test_vecadd.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

test_reference: test_vecadd.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f test_student test_reference