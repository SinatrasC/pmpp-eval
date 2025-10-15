// student_kernel.cu
#include <cuda_runtime.h>
#include <cstdio>

// Implement a single explicit 7-point heat step.
// in  : N*N*N input (flattened, row-major: i*N*N + j*N + k)
// out : N*N*N output (same layout)
// N   : grid dimension
// alpha, dt, dx: physical parameters; r = alpha*dt/(dx*dx)
// Boundary policy: copy boundary through (out = in) if any neighbor would be OOB.
__global__ void heat_step_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 unsigned int N,
                                 float alpha, float dt, float dx)
{
    // TODO:
    // 1) Compute (i,j,k) from block and thread indices
    // 2) If any of i,j,k is 0 or N-1 => boundary: out = in and return
    // 3) Else compute r = alpha*dt/(dx*dx) and 7-point update:
    //    out = in + r*(sum six neighbors - 6*in)
    // Guard small N (N<3) by simply copying (no interior exists).
}