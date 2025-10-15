// student_kernel.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void stencil3d_basic_student(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6)
{
    // TODO:
    // - Each thread computes OUT(i,j,k) for one grid point
    // - Use 7-point stencil on INTERIOR points: (1..N-2) in each dim
    // - For boundary points (i==0 || i==N-1 || ...), copy through: out = in
    // - Guard for N==0 or N==1 safely
    // Hints:
    //   int i = blockIdx.z * blockDim.z + threadIdx.z;
    //   int j = blockIdx.y * blockDim.y + threadIdx.y;
    //   int k = blockIdx.x * blockDim.x + threadIdx.x;
    //   int idx = (i * N + j) * N + k;
    //   int L = ((i) * N + j) * N + (k-1); // k-1, etc.
}