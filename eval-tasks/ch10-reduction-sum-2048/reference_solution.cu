// reference_solution.cu
#include <cuda_runtime.h>

extern "C" __global__
void reduce_sum_2048(const float* __restrict__ in, float* __restrict__ out) {
    __shared__ float s[1024];
    const unsigned t = threadIdx.x;

    // 2 elements per thread; single block (1024 threads) → 2048 total
    float v = in[t] + in[t + 1024];
    s[t] = v;
    __syncthreads();

    // Convergent reduction
    for (unsigned stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        if (t < stride) {
            s[t] += s[t + stride];
        }
        __syncthreads();
        if (stride == 1) break; // avoid undefined behavior on shift to 0
    }

    if (t == 0) out[0] = s[0];
}