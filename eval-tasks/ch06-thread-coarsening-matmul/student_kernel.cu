#include <cuda_runtime.h>
#include <cstdio>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

// Students: implement a coarsened, tiled GEMM C[M×K] = A[M×N] * B[N×K]
// Each block computes a tile: height TILE_WIDTH, width TILE_WIDTH*COARSE_FACTOR
// Use shared memory tiles for A and B; do safe (bounds-checked) loads.
// Row-major layout: elem(i,j) at base[i*ld + j].

__global__ void MatmulCoarsenedKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K)
{
    // TODO: Implement thread coarsening matrix multiplication
    //
    // Key requirements:
    // 1. Use shared memory tiles for A and B:
    //    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    //    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR];
    //
    // 2. Each thread computes COARSE_FACTOR output elements
    //    - Thread (tx,ty) computes elements at columns: colBase + c*TILE_WIDTH for c=0..COARSE_FACTOR-1
    //    - Use register array: float acc[COARSE_FACTOR];
    //
    // 3. Loop over tiles of the N dimension:
    //    - Load A tile (TILE_WIDTH x TILE_WIDTH)
    //    - Load B super-tile (TILE_WIDTH x TILE_WIDTH*COARSE_FACTOR) in COARSE_FACTOR stripes
    //    - __syncthreads() after loading
    //    - Compute partial products with triple nested loop (k, c)
    //    - __syncthreads() before next iteration
    //
    // 4. Write results with bounds checking
    //
    // Template structure:
    // const int ty = threadIdx.y;
    // const int tx = threadIdx.x;
    // const int row = blockIdx.y * TILE_WIDTH + ty;
    // const int colBase = blockIdx.x * (TILE_WIDTH * COARSE_FACTOR) + tx;
    //
    // float acc[COARSE_FACTOR];
    // for (int c = 0; c < COARSE_FACTOR; ++c) acc[c] = 0.0f;
    //
    // Loop over tiles...
    //
    // Write results...

    // Placeholder implementation (incorrect, will fail tests):
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        C[row * K + col] = 0.0f; // TODO: Replace with actual coarsened computation
    }
}

// Student launcher: choose grid/block and launch your kernel
extern "C" void launch_student(const float* A_d,
                               const float* B_d,
                               float* C_d,
                               int M, int N, int K)
{
    // TODO: Configure proper grid and block dimensions for thread coarsening
    //
    // Key points:
    // - Block size should be (TILE_WIDTH, TILE_WIDTH)
    // - Grid X dimension should account for COARSE_FACTOR: (K + TILE_WIDTH*COARSE_FACTOR - 1) / (TILE_WIDTH*COARSE_FACTOR)
    // - Grid Y dimension covers rows: (M + TILE_WIDTH - 1) / TILE_WIDTH

    // Placeholder launch (incorrect dimensions):
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((K + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    MatmulCoarsenedKernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
}