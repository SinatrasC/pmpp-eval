#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ unsigned char clamp_u8(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// BAD: Missing proper rounding (truncation instead)
__global__ void rgb2grayKernel(const unsigned char* R,
                               const unsigned char* G,
                               const unsigned char* B,
                               unsigned char* gray,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y = 0.299f * (float)R[i] + 0.587f * (float)G[i] + 0.114f * (float)B[i];
        int yi = (int)y;  // Truncation instead of floorf(y + 0.5f)
        gray[i] = clamp_u8(yi);
    }
}