#include <cuda_runtime.h>

// TODO: Implement Stage-2 interior update for the 25-point stencil (R=4).
// Local z extent = dimz + 8. Owned z = [4 .. 4+dimz-1].
// Stage-2 interior planes: k ∈ [8 .. (4+dimz-1)-4].
// For i/j edges (i<4 || i>=dimx-4 || j<4 || j>=dimy-4) copy-through.
// Do not touch halos or Stage-1 planes.
//
// STENCIL WEIGHTS: Use distance-based weighted stencil (NOT simple averaging!)
// - w0 = 0.5   (center point)
// - w1 = 0.10  (distance 1 neighbors: ±1 in each axis)
// - w2 = 0.05  (distance 2 neighbors: ±2 in each axis)
// - w3 = 0.025 (distance 3 neighbors: ±3 in each axis)
// - w4 = 0.0125 (distance 4 neighbors: ±4 in each axis)
//
// Formula: out[i,j,k] = w0 * in[i,j,k]
//                     + w1 * (in[i±1,j,k] + in[i,j±1,k] + in[i,j,k±1])
//                     + w2 * (in[i±2,j,k] + in[i,j±2,k] + in[i,j,k±2])
//                     + w3 * (in[i±3,j,k] + in[i,j±3,k] + in[i,j,k±3])
//                     + w4 * (in[i±4,j,k] + in[i,j±4,k] + in[i,j,k±4])

static inline __device__ size_t idx3(int i,int j,int k,int dx,int dy){
    return (size_t(k)*dy + j)*dx + i;
}

__global__ void stencil25_stage2_kernel(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int dimx,int dimy,int dimz)
{
    // TODO
}

extern "C" void stencil25_stage2_interior(const float* d_in, float* d_out,
                                          int dimx,int dimy,int dimz)
{
    // TODO: launch 3D grid and synchronize
}