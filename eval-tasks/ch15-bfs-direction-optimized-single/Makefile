# ch15-bfs-direction-optimized-single / Makefile

CXX = nvcc
CXXFLAGS = -std=c++17 -O2 -arch=sm_70

TARGET_STUDENT = test_student
TARGET_REFERENCE = test_reference

all: $(TARGET_STUDENT) $(TARGET_REFERENCE)

$(TARGET_STUDENT): test_bfs_diropt.cu student_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TARGET_REFERENCE): test_bfs_diropt.cu reference_solution.cu
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET_STUDENT) $(TARGET_REFERENCE)

.PHONY: all clean