#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

#ifndef TILE
#define TILE 16
#endif

#ifndef MAX_RADIUS
#define MAX_RADIUS 8
#endif

__constant__ float c_filter[(2*MAX_RADIUS+1)*(2*MAX_RADIUS+1)];

extern "C" __host__ void setFilterConstant(const float* h_filter, int r) {
    const int K = 2*r + 1;
    cudaMemcpyToSymbol(c_filter, h_filter, K*K*sizeof(float), 0, cudaMemcpyHostToDevice);
}

static __device__ __forceinline__ int filterIndex(int r, int dy, int dx) {
    const int K = 2*r + 1;
    return (dy + r)*K + (dx + r);
}

__global__ void conv2d_tiled_constant_kernel(const float* __restrict__ in,
                                             float* __restrict__ out,
                                             int H, int W, int r)
{
    extern __shared__ float smem[]; // size = (TILE+2*r)*(TILE+2*r)
    const int K = 2*r + 1;
    const int tileW = TILE + 2*r;
    const int tileH = TILE + 2*r;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * TILE + tx;
    const int y = blockIdx.y * TILE + ty;

    // Top-left of the shared tile in global coordinates
    const int startX = blockIdx.x * TILE - r;
    const int startY = blockIdx.y * TILE - r;

    // Cooperative load of the full (tileH x tileW) region with strides
    for (int yy = ty; yy < tileH; yy += blockDim.y) {
        const int gy = startY + yy;
        for (int xx = tx; xx < tileW; xx += blockDim.x) {
            const int gx = startX + xx;
            float v = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                v = in[gy*W + gx];
            }
            smem[yy*tileW + xx] = v;
        }
    }

    __syncthreads();

    if (x < W && y < H) {
        float acc = 0.0f;
        const int sx = tx + r;
        const int sy = ty + r;

        // Convolution
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                const float w = c_filter[filterIndex(r, dy, dx)];
                const float v = smem[(sy+dy)*tileW + (sx+dx)];
                acc += w * v;
            }
        }
        out[y*W + x] = acc;
    }
}