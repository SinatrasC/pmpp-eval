# ch18-energy-scatter-single / Makefile

NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
# Allow override of SM; default works on Volta+; or drop -arch to let nvcc choose.
ARCH ?= -arch=sm_70
DEFS ?= -DCHUNK_SIZE=256
# Enable device linking across TUs; extra safety for device symbols.
RDC ?= -rdc=true

all: test_reference test_student

test_reference: test_energy_scatter.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) $(ARCH) $(DEFS) $(RDC) -o $@ $^

test_student: test_energy_scatter.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) $(ARCH) $(DEFS) $(RDC) -o $@ $^

clean:
	rm -f test_reference test_student *.o

.PHONY: all clean