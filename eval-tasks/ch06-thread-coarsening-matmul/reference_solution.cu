#include <cuda_runtime.h>
#include <cstdio>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

// Correct, coarsened, tiled row-major GEMM with boundary-safe loads/stores.
// C[M×K] = A[M×N] * B[N×K]
__global__ void MatmulCoarsenedKernelRef(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int blockRow = blockIdx.y * TILE_WIDTH;
    const int blockCol = blockIdx.x * (TILE_WIDTH * COARSE_FACTOR);

    const int row = blockRow + ty;
    const int colBase = blockCol + tx;

    float acc[COARSE_FACTOR];
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) acc[c] = 0.0f;

    const int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE_WIDTH + tx;
        const int aRow = row;

        // Load A tile
        As[ty][tx] = (aRow < M && aCol < N) ? A[aRow * N + aCol] : 0.0f;

        // Load B super-tile (COARSE_FACTOR stripes)
        const int bRow = t * TILE_WIDTH + ty;

        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            const int bCol = colBase + c * TILE_WIDTH;
            Bs[ty][tx + c * TILE_WIDTH] =
                (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;
        }

        __syncthreads();

        // Multiply-accumulate
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            const float aVal = As[ty][k];
            #pragma unroll
            for (int c = 0; c < COARSE_FACTOR; ++c) {
                acc[c] += aVal * Bs[k][tx + c * TILE_WIDTH];
            }
        }
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        const int col = colBase + c * TILE_WIDTH;
        if (row < M && col < K) {
            C[row * K + col] = acc[c];
        }
    }
}

extern "C" void launch_reference(const float* A_d,
                                 const float* B_d,
                                 float* C_d,
                                 int M, int N, int K)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    const int tileW = TILE_WIDTH * COARSE_FACTOR;
    dim3 grid((K + tileW - 1) / tileW,
              (M + TILE_WIDTH - 1) / TILE_WIDTH,
              1);

    MatmulCoarsenedKernelRef<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
}