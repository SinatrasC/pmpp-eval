// reference_solution.cu
#include <cuda_runtime.h>

extern "C" __global__
void reduce_sum_arbitrary(const float* __restrict__ in,
                          float* __restrict__ out, int n)
{
    extern __shared__ float s[];
    const int tid = threadIdx.x;

    // 2-elements-per-thread grid-stride loop
    long long idx    = (long long)blockIdx.x * blockDim.x * 2 + tid;
    const long long stride = (long long)gridDim.x * blockDim.x * 2;

    float sum = 0.f;
    for (; idx < n; idx += stride) {
        sum += in[idx];
        long long idx2 = idx + blockDim.x;
        if (idx2 < n) sum += in[(int)idx2];
    }
    s[tid] = sum;
    __syncthreads();

    // Convergent shared-memory reduction
    for (int step = blockDim.x >> 1; step >= 1; step >>= 1) {
        if (tid < step) s[tid] += s[tid + step];
        __syncthreads();
        if (step == 1) break;
    }

    if (tid == 0) atomicAdd(out, s[0]);
}