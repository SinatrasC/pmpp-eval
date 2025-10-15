// reference_solution.cu
#include <cuda_runtime.h>

static __device__ __forceinline__
size_t idx(unsigned i, unsigned j, unsigned k, unsigned N) {
    return (size_t)i * N * N + (size_t)j * N + (size_t)k;
}

__global__ void heat_step_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 unsigned int N,
                                 float alpha, float dt, float dx)
{
    unsigned i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N || k >= N) return;

    // For tiny grids there is no interior
    if (N < 3 || i == 0 || j == 0 || k == 0 || i == N-1 || j == N-1 || k == N-1) {
        out[idx(i,j,k,N)] = in[idx(i,j,k,N)];
        return;
    }

    const float r = alpha * dt / (dx * dx);

    const float c  = in[idx(i,  j,  k,  N)];
    const float xm = in[idx(i-1,j,  k,  N)];
    const float xp = in[idx(i+1,j,  k,  N)];
    const float ym = in[idx(i,  j-1,k,  N)];
    const float yp = in[idx(i,  j+1,k,  N)];
    const float zm = in[idx(i,  j,  k-1,N)];
    const float zp = in[idx(i,  j,  k+1,N)];

    out[idx(i,j,k,N)] = c + r * ((xm + xp + ym + yp + zm + zp) - 6.0f * c);
}