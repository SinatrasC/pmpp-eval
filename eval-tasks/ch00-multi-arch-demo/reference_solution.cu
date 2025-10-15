#include <cuda_runtime.h>
#include <cstdio>

__global__ void vec_add_ref(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void launch_reference(const float* a, const float* b, float* c, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vec_add_ref<<<grid, block>>>(a, b, c, n);
}
