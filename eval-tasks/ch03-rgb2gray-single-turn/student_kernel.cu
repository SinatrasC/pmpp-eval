#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ unsigned char clamp_u8(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__global__ void rgb2grayKernel(const unsigned char* R,
                               const unsigned char* G,
                               const unsigned char* B,
                               unsigned char* gray,
                               int n) {
    // TODO:
    // - Compute global index i
    // - If (i < n), compute:
    //     float y = 0.299f*R[i] + 0.587f*G[i] + 0.114f*B[i];
    //     int yi = (int)floorf(y + 0.5f);   // round to nearest
    //     gray[i] = clamp_u8(yi);
}