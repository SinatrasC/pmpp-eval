NVCC ?= nvcc
ARCH ?= sm_70
CFLAGS = -std=c++14 -O2 -Xcompiler -fno-exceptions -arch=$(ARCH)

all: test_reference test_student

test_reference: test_pack.o reference_solution.o
	$(NVCC) $(CFLAGS) -o $@ $^

test_student: test_pack.o student_kernel.o
	$(NVCC) $(CFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o test_reference test_student