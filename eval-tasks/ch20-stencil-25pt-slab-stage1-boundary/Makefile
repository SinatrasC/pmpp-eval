NVCC ?= nvcc
CXXFLAGS := -O2 -std=c++17

TARGET_REF := test_reference
TARGET_STU := test_student

COMMON := test_stage1.cu
REFSRC := reference_solution.cu
STUSRC := student_kernel.cu

all: $(TARGET_REF)

$(TARGET_REF): $(COMMON) $(REFSRC)
	$(NVCC) $(CXXFLAGS) -o $@ $^

$(TARGET_STU): $(COMMON) $(STUSRC)
	$(NVCC) $(CXXFLAGS) -o $@ $^

run_ref: $(TARGET_REF)
	./$(TARGET_REF)

run_student: $(TARGET_STU)
	./$(TARGET_STU)

clean:
	rm -f $(TARGET_REF) $(TARGET_STU) *.o