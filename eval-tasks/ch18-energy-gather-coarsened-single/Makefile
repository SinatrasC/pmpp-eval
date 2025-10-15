# ch18-energy-gather-coarsened-single / Makefile

NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17
ARCH ?= -arch=sm_70
DEFS ?= -DCHUNK_SIZE=256 -DCOARSEN_FACTOR=8
# Enable device linking across TUs; extra safety for device symbols.
RDC ?= -rdc=true

all: test_reference test_student

test_reference: test_energy_gather_coarsened.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) $(ARCH) $(DEFS) $(RDC) -o $@ $^

test_student: test_energy_gather_coarsened.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) $(ARCH) $(DEFS) $(RDC) -o $@ $^

clean:
	rm -f test_reference test_student *.o

.PHONY: all clean