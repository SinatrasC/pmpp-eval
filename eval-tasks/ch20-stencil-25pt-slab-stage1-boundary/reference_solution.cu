#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

static inline __device__ size_t idx3(int i,int j,int k,int dx,int dy){
    return (size_t(k)*dy + j)*dx + i;
}

// Owned region local z: [4 .. 4+dimz-1]
// Stage-1 planes: [4..7] and [4+dimz-4 .. 4+dimz-1]
__global__ void stencil25_stage1_kernel_ref(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int dimx,int dimy,int dimz)
{
    const int R=4;
    const int zOwnedBeg = 4;
    const int zOwnedEnd = 4 + dimz - 1;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z; // local z including halos

    // total local z extent = dimz + 8
    if(i>=dimx || j>=dimy || k>=dimz+8) return;

    bool inStage1 = ( (k >= zOwnedBeg && k <= zOwnedBeg+3) ||
                      (k >= zOwnedEnd-3 && k <= zOwnedEnd) );

    if(!inStage1){
        // Do not touch other planes
        return;
    }

    const float w0=0.5f, w1=0.10f, w2=0.05f, w3=0.025f, w4=0.0125f;
    const float w[5]={w0,w1,w2,w3,w4};

    const bool interiorXY = (i>=R && i<dimx-R && j>=R && j<dimy-R);
    const size_t p = idx3(i,j,k,dimx,dimy);

    if(!interiorXY){
        // Copy-through for edge i/j within Stage-1 planes
        out[p] = in[p];
        return;
    }

    float acc = w[0]*in[p];
    #pragma unroll
    for(int d=1; d<=4; ++d){
        acc += w[d] * ( in[idx3(i-d,j,k,dimx,dimy)] + in[idx3(i+d,j,k,dimx,dimy)]
                      + in[idx3(i,j-d,k,dimx,dimy)] + in[idx3(i,j+d,k,dimx,dimy)]
                      + in[idx3(i,j,k-d,dimx,dimy)] + in[idx3(i,j,k+d,dimx,dimy)] );
    }
    out[p] = acc;
}

// Host wrapper: arrays sized for (dimx*dimy*(dimz+8))
extern "C" void stencil25_stage1_boundary(const float* d_in, float* d_out,
                                          int dimx,int dimy,int dimz)
{
    dim3 block(8,8,8);
    dim3 grid( (dimx+block.x-1)/block.x,
               (dimy+block.y-1)/block.y,
               ((dimz+8)+block.z-1)/block.z );
    stencil25_stage1_kernel_ref<<<grid,block>>>(d_in, d_out, dimx,dimy,dimz);
    cudaDeviceSynchronize();
}