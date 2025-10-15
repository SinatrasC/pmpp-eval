// student_kernel.cu
#include <cuda_runtime.h>

// TODO: Implement arbitrary-length sum with grid-stride loads, shared-memory
// reduction per block, and atomicAdd(out, block_sum). Use dynamic shared memory.
extern "C" __global__
void reduce_sum_arbitrary(const float* in, float* out, int n) {
    // TODO
    // Suggested shape:
    // extern __shared__ float s[];
    // int tid = threadIdx.x;
    // long long idx = blockIdx.x * (long long)blockDim.x * 2 + tid;
    // long long stride = (long long)gridDim.x * blockDim.x * 2;
    // float sum = 0.f;
    // for (; idx < n; idx += stride) {
    //   sum += in[idx];
    //   long long idx2 = idx + blockDim.x;
    //   if (idx2 < n) sum += in[idx2];
    // }
    // s[tid] = sum; __syncthreads();
    // for (int step = blockDim.x/2; step >= 1; step >>= 1) { ... }
    // if (tid==0) atomicAdd(out, s[0]);
}