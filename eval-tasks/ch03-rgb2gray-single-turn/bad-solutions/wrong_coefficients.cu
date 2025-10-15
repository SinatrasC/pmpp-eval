#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ unsigned char clamp_u8(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// BAD: Wrong RGB coefficients (equal weights instead of proper luminance)
__global__ void rgb2grayKernel(const unsigned char* R,
                               const unsigned char* G,
                               const unsigned char* B,
                               unsigned char* gray,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float y = 0.333f * (float)R[i] + 0.333f * (float)G[i] + 0.333f * (float)B[i];  // Wrong coefficients
        int yi = (int)floorf(y + 0.5f);
        gray[i] = clamp_u8(yi);
    }
}