// reference_solution.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void stencil3d_basic_student(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6)
{
    if (N <= 0) return;

    int k = blockIdx.x * blockDim.x + threadIdx.x; // x -> k
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y -> j
    int i = blockIdx.z * blockDim.z + threadIdx.z; // z -> i
    if (i >= N || j >= N || k >= N) return;

    auto idx = [N](int I, int J, int K){ return (I * N + J) * N + K; };

    bool interior = (i > 0 && i < N-1) && (j > 0 && j < N-1) && (k > 0 && k < N-1);
    if (!interior) {
        out[idx(i,j,k)] = in[idx(i,j,k)];
        return;
    }

    float ctr = in[idx(i,  j,  k  )];
    float xm  = in[idx(i,  j,  k-1)];
    float xp  = in[idx(i,  j,  k+1)];
    float ym  = in[idx(i,  j-1,k  )];
    float yp  = in[idx(i,  j+1,k  )];
    float zm  = in[idx(i-1,j,  k  )];
    float zp  = in[idx(i+1,j,  k  )];

    out[idx(i,j,k)] = c0*ctr + c1*xm + c2*xp + c3*ym + c4*yp + c5*zm + c6*zp;
}