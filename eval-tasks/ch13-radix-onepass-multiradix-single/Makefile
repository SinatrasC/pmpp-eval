NVCC ?= nvcc
CXXFLAGS := -O2 -std=c++17

all: test_reference test_student

test_reference: test_radix_onepass.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

test_student: test_radix_onepass.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student *.o