#include <cuda_runtime.h>

// TODO: Implement vector addition kernel
// Requirements:
// - Each thread computes one element of output array c
// - Use global thread index to identify which element to compute
// - Add bounds checking to handle array sizes not divisible by block size
// - Perform: c[i] = a[i] + b[i]
//
// Hints:
// - Calculate global index from blockIdx, blockDim, and threadIdx
// - Check: if (index < n) before accessing arrays
__global__ void vec_add_student(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    // TODO: Implement kernel body
}

// TODO: Implement kernel launcher function
// Requirements:
// - Configure grid and block dimensions
// - Launch vec_add_student kernel with appropriate dimensions
// - Ensure enough threads to cover all n elements
//
// Hints:
// - Common block size: 256 threads
// - Grid size: (n + blockSize - 1) / blockSize for full coverage
extern "C" void launch_student(const float* a, const float* b, float* c, int n) {
    // TODO: Configure dimensions and launch kernel
}
