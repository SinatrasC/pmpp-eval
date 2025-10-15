#include <cuda_runtime.h>

// TODO: Implement a single-GPU axis-aligned 25-point stencil with radius R=4.
// Contract:
//  - Input/Output are dense 3D grids (dimx*dimy*dimz), row-major:
//      idx(i,j,k) = (k*dimy + j)*dimx + i
//  - For interior cells (i,j,k ∈ [4 .. dim-1-4]) compute:
//      out = w0*center + Σ_{d=1..4} w[d] * (±d along x + ±d along y + ±d along z)
//    with weights: w0=0.5, w1=0.10, w2=0.05, w3=0.025, w4=0.0125
//  - Boundary cells (within 4 of any face) must be copy-through: out=in
//  - Reasonable 3D launch config and synchronization
//
// Suggested steps:
//  1) write idx3 helper
//  2) in-kernel boundary test => copy-through
//  3) accumulate using a small unrolled loop d=1..4
//  4) host wrapper launches kernel

static inline __device__ size_t idx3(int i,int j,int k,int dx,int dy){
    return (size_t(k)*dy + j)*dx + i;
}

__global__ void stencil25_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int dimx, int dimy, int dimz)
{
    // TODO
}

extern "C" void stencil25_single_gpu(const float* d_in, float* d_out,
                                     int dimx, int dimy, int dimz)
{
    // TODO: choose block/grid and launch kernel, then cudaDeviceSynchronize()
}