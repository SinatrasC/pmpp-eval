# Makefile for matmul-tiled-multiturn
NVCC ?= nvcc
ARCH ?= -arch=sm_70
CXXFLAGS := -O2 -std=c++17 $(ARCH)

# Binaries
STUDENT_BIN := test_student
REF_BIN     := test_reference

all: $(STUDENT_BIN) $(REF_BIN)

$(STUDENT_BIN): test_matmul.cu student_kernel.cu
	$(NVCC) $(CXXFLAGS) -DBUILD_STUDENT -o $@ $^

$(REF_BIN): test_matmul.cu reference_solution.cu
	$(NVCC) $(CXXFLAGS) -DBUILD_REFERENCE -o $@ $^

run-student: $(STUDENT_BIN)
	./$(STUDENT_BIN)

run-reference: $(REF_BIN)
	./$(REF_BIN)

clean:
	rm -f $(STUDENT_BIN) $(REF_BIN)

.PHONY: all run-student run-reference clean
