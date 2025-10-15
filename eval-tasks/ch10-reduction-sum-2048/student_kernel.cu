// student_kernel.cu
#include <cuda_runtime.h>

// TODO: Implement convergent shared-memory reduction for exactly 2048 elements.
// Contract:
//  - gridDim.x == 1, blockDim.x == 1024
//  - Each thread loads two elements: in[tid] and in[tid + 1024]
//  - Reduce in shared memory (convergent pattern), write out[0] only.
extern "C" __global__
void reduce_sum_2048(const float* in, float* out) {
    // TODO
    // Suggested shape:
    // __shared__ float s[1024];
    // unsigned t = threadIdx.x;
    // float v = in[t] + in[t + 1024];
    // s[t] = v;
    // __syncthreads();
    // for (unsigned stride = blockDim.x/2; stride >= 1; stride >>= 1) { ... }
    // if (t == 0) out[0] = s[0];
}