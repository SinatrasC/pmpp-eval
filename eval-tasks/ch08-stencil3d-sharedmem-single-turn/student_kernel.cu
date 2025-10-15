// student_kernel.cu
#include <cuda_runtime.h>
#include <cstdio>

// Tile parameters for this task
#ifndef IN_TILE_DIM
#define IN_TILE_DIM 8          // threads per dim that load (with halo)
#endif
#define OUT_TILE_DIM (IN_TILE_DIM-2)

__global__ void stencil3d_shared_student(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6)
{
    // TODO:
    // - Launch with block=(IN_TILE_DIM,IN_TILE_DIM,IN_TILE_DIM)
    // - Each block loads a IN_TILE_DIM^3 tile (with halo) into shared memory
    // - Only threads with local coords in [1..IN_TILE_DIM-2] compute outputs
    //   for the corresponding global interior coordinates
    // - Copy-through boundary cells (same rule as basic)
    // Hints:
    //   Shared array: __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    //   Global coords start at (blockIdx * OUT_TILE_DIM) - 1 (to include halo)
    //   Guard global loads (row/col/depth) that fall outside [0..N-1]
}