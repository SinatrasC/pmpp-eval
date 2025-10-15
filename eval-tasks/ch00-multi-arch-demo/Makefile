NVCC ?= nvcc
SM_LIST ?= 70 80 90
MAX_SM := $(word $(words $(SM_LIST)),$(SM_LIST))
GENCODE_FLAGS := $(foreach sm,$(SM_LIST),-gencode=arch=compute_$(sm),code=sm_$(sm))
GENCODE_FLAGS += -gencode=arch=compute_$(MAX_SM),code=compute_$(MAX_SM)
NVCCFLAGS := -O2 -std=c++17 $(GENCODE_FLAGS)

.PHONY: all clean test_reference test_student

all: test_reference test_student

test_reference: test_demo.cu reference_solution.cu student_kernel.cu
	$(NVCC) $(NVCCFLAGS) -DREFERENCE_BUILD -o $@ $^

test_student: test_demo.cu student_kernel.cu reference_solution.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student
