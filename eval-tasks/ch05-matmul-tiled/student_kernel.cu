#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cstdio>

#ifndef TILE
#define TILE 16
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

// TODO: Implement shared-memory tiled matrix multiplication kernel
//
// Requirements:
//  - Implement GEMM: P = M * N where M is m×n, N is n×o, P is m×o
//  - Use shared memory tiling to reduce global memory accesses
//  - Declare two shared memory tiles (size TILE×TILE) for M and N submatrices
//  - Process matrix multiplication in phases, loading one tile at a time
//  - Use __syncthreads() to ensure all threads have loaded data before computing
//  - Handle edge cases with bounds checking (matrices may not align to TILE boundaries)
//  - Do NOT modify inputs M or N (they are const)
//
// Algorithm outline:
//  1. Calculate output row and column for this thread
//  2. Loop over tiles along the shared dimension (k-dimension):
//     a. Load one TILE×TILE submatrix from M into shared memory
//     b. Load one TILE×TILE submatrix from N into shared memory
//     c. Use bounds checks: load 0.0f if out of matrix bounds
//     d. Synchronize threads (__syncthreads())
//     e. Compute partial dot product using shared memory tiles
//     f. Synchronize threads again before next tile
//  3. Write accumulated result to global memory P
//
// Hints:
//  - TILE is defined as a compile-time constant (default 16)
//  - Use threadIdx for within-block indexing, blockIdx for block position
//  - Shared memory declarations: __shared__ float tile_M[TILE][TILE];
//  - Number of phases: (n + TILE - 1) / TILE
//  - Matrix indexing: M[row * n + k], N[k * o + col], P[row * o + col]

__global__ void TiledMatMulKernel(const float* __restrict__ M,
                                  const float* __restrict__ N,
                                  float* __restrict__ P,
                                  int m, int n, int o)
{
    // TODO: Implement tiled matrix multiplication
}

// TODO: Implement kernel launcher
//
// Requirements:
//  - Allocate device memory for M, N, and P matrices
//  - Copy input matrices M and N from host to device
//  - Configure grid and block dimensions for tiled execution
//  - Launch TiledMatMulKernel with appropriate parameters
//  - Copy result matrix P from device back to host
//  - Free all device allocations
//
// Hints:
//  - Handle edge case: if m==0 || n==0 || o==0, return early
//  - Memory size: M needs m*n*sizeof(float), N needs n*o*sizeof(float), P needs m*o*sizeof(float)
//  - Use gpuErrchk() macro to wrap all CUDA API calls for error checking
//  - Block dimensions: dim3 block(TILE, TILE) for 2D thread blocks
//  - Grid dimensions: ensure enough blocks to cover entire output matrix
//  - Grid size formula: (dimension + TILE - 1) / TILE
//  - Remember to synchronize after kernel launch to catch execution errors

extern "C"
void launch_tiled_matmul(const float* M_h, const float* N_h, float* P_h,
                         int m, int n, int o)
{
    // TODO: Implement launcher
    (void)M_h; (void)N_h; (void)P_h; (void)m; (void)n; (void)o;
}