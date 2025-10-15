// students edit only this file
#include <cuda_runtime.h>
#include <cstdio>

#ifndef TILE
#define TILE 16
#endif

#ifndef MAX_RADIUS
#define MAX_RADIUS 8   // supports up to (2*8+1)=17x17 filters
#endif

// 1D constant buffer for filter coefficients (row-major)
// Size = (2*MAX_RADIUS+1)^2
__constant__ float c_filter[(2*MAX_RADIUS+1)*(2*MAX_RADIUS+1)];

extern "C" __host__ void setFilterConstant(const float* h_filter, int r) {
    const int K = 2*r + 1;
    cudaMemcpyToSymbol(c_filter, h_filter, K*K*sizeof(float), 0, cudaMemcpyHostToDevice);
}

// Students must implement this kernel.
// Requirements:
// - Shared-memory tiling with halo of +/-r (use zero padding for out-of-bounds loads)
// - Use c_filter (in constant memory) for filter coefficients
// - Each thread computes one output pixel (if in bounds)
// - Grid/block: 2D, blockDim=(TILE,TILE), gridDim=ceil(W/TILE) x ceil(H/TILE)
// - Inputs/outputs are float* (grayscale), shapes: in/out = H*W
// - r is runtime radius (<= MAX_RADIUS)
__global__ void conv2d_tiled_constant_kernel(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             int H, int W, int r)
{
    // TODO: Implement tiled 2D convolution with constant memory
    //
    // Key steps:
    // 1) Compute global (x,y) coordinates for this thread
    // 2) Declare shared memory tile with halo: extern __shared__ float smem[];
    //    Size needed: (TILE+2*r) * (TILE+2*r) * sizeof(float)
    // 3) Compute the tile's coverage region including halo
    // 4) Cooperatively load the entire tile+halo region into shared memory
    //    - Use zero padding for out-of-bounds pixels
    //    - May need nested loops for threads to cover entire shared memory region
    // 5) __syncthreads() to ensure all data is loaded
    // 6) If this thread's output pixel is in bounds, compute convolution:
    //    - Access input pixels from shared memory (with proper offsets)
    //    - Access filter coefficients from c_filter constant memory
    //    - Accumulate weighted sum
    // 7) Write result to global output memory
    //
    // Hints:
    // - Filter coefficients in c_filter are stored row-major: c_filter[(dy+r)*(2*r+1) + (dx+r)]
    // - Shared memory indexing: smem[sy * tileWidth + sx] where tileWidth = TILE+2*r
    // - Global input indexing: in[gy * W + gx]
    // - Consider boundary conditions carefully for both image edges and tile edges

    // Placeholder implementation (will fail until properly implemented):
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        // This is incorrect - just copies input to output
        out[y * W + x] = in[y * W + x];
    }
}