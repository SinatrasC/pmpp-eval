NVCC ?= nvcc
CXXFLAGS := -O2 -std=c++17

all: test_reference test_student

test_reference: test_merge_sort.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

test_student: test_merge_sort.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student *.o