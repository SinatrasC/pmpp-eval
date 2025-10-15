# ch15-bfs-push-single / Makefile
NVCC ?= nvcc
CFLAGS := -O3 -std=c++17

all: test_reference test_student

test_reference: test_bfs_push.cu reference_solution.cu
	$(NVCC) $(CFLAGS) -o $@ $^

test_student: test_bfs_push.cu student_kernel.cu
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f test_reference test_student