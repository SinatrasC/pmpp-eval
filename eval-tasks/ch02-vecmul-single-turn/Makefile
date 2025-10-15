NVCC ?= nvcc
CXXFLAGS := -O2

all: test_student test_reference

test_student: test_vecmul.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

test_reference: test_vecmul.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -o $@ $^

clean:
	rm -f test_student test_reference