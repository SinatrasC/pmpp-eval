NVCC      := nvcc
CXXFLAGS  := -O2 -std=c++17

SRC_TEST    := test_device_props.cu
SRC_STUDENT := student_kernel.cu
SRC_REF     := reference_solution.cu

.PHONY: all run clean

all: test_reference test_student

test_reference: $(SRC_TEST) $(SRC_REF) reference_only_wrapper.cu student_kernel.cuh
	$(NVCC) $(CXXFLAGS) -o $@ $(SRC_TEST) $(SRC_REF) reference_only_wrapper.cu

test_student: $(SRC_TEST) $(SRC_STUDENT) $(SRC_REF) student_kernel.cuh
	$(NVCC) $(CXXFLAGS) -o $@ $(SRC_TEST) $(SRC_STUDENT) $(SRC_REF)

run: test_student
	./test_student

clean:
	rm -f test_reference test_student