NVCC ?= nvcc
CXXFLAGS := -O2 -std=c++17

all: test_student test_reference

test_student: test_rgb2gray.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

test_reference: test_rgb2gray.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f test_student test_reference