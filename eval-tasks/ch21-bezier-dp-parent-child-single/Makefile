NVCC ?= nvcc
ARCH ?= sm_70
CFLAGS = -std=c++14 -O2 -Xcompiler -fno-exceptions -rdc=true -arch=$(ARCH)
LDFLAGS = -lcudadevrt

all: test_reference test_student

test_reference: test_bezier_dp.o reference_solution.o
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_student: test_bezier_dp.o student_kernel.o
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o test_reference test_student