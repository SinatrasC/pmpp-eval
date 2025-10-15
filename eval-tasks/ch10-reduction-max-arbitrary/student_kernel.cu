// student_kernel.cu
#include <cuda_runtime.h>
#include <limits>

// TODO: Implement arbitrary-length maximum reduction with grid-stride, shared
// memory, and a CAS-loop atomicMax for float. Initialize per-thread local max
// to -INFINITY when n==0 or no elements in its stride.
//
// IMPORTANT: Use the -INFINITY macro (not std::numeric_limits<float>::infinity())
// Example: float local_max = -INFINITY;
__device__ inline
void atomicMaxFloat(float* addr, float val) {
    // TODO: Implement via atomicCAS on int bit patterns
}

extern "C" __global__
void reduce_max_arbitrary(const float* in, float* out, int n) {
    // TODO
}