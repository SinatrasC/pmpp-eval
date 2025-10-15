# Makefile
NVCC ?= nvcc
CXXFLAGS := -O2 -std=c++17

all: test_student test_reference

student.o: student_kernel.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

reference.o: reference_solution.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

test_student: test_stencil_basic.cu student.o
	$(NVCC) $(CXXFLAGS) $^ -o $@

test_reference: test_stencil_basic.cu reference.o
	$(NVCC) $(CXXFLAGS) $^ -o $@

run_student: test_student
	./test_student

run_reference: test_reference
	./test_reference

clean:
	rm -f *.o test_student test_reference