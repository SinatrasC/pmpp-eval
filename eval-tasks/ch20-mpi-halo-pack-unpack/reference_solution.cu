#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cassert>

static inline __host__ __device__
size_t idx3(int i,int j,int k,int dx,int dy){ return (size_t(k)*dy + j)*dx + i; }

static inline __host__ __device__
size_t pack_idx(int p,int j,int i,int dx,int dy){ return (size_t(p)*dy + j)*dx + i; }

__global__ void k_pack(const float* __restrict__ grid,
                       int dimx,int dimy,int dimz,
                       float* __restrict__ left_send,
                       float* __restrict__ right_send)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i>=dimx || j>=dimy) return;

    const int zOwnedBeg = 4;
    const int zOwnedEnd = 4 + dimz - 1;

    // pack 4 planes on each side
    #pragma unroll
    for(int p=0;p<4;++p){
        int kL = zOwnedBeg + p;
        int kR = (zOwnedEnd - 3) + p;

        size_t srcL = idx3(i,j,kL,dimx,dimy);
        size_t srcR = idx3(i,j,kR,dimx,dimy);
        size_t dst  = pack_idx(p,j,i,dimx,dimy);

        left_send [dst] = grid[srcL];
        right_send[dst] = grid[srcR];
    }
}

__global__ void k_unpack(float* __restrict__ grid,
                         int dimx,int dimy,int dimz,
                         const float* __restrict__ left_recv,
                         const float* __restrict__ right_recv)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i>=dimx || j>=dimy) return;

    const int zHaloL = 0;
    const int zHaloR = dimz + 4; // first right halo plane

    #pragma unroll
    for(int p=0;p<4;++p){
        int kL = zHaloL + p;       // 0..3
        int kR = zHaloR + p;       // dimz+4 .. dimz+7

        size_t dstL = idx3(i,j,kL,dimx,dimy);
        size_t dstR = idx3(i,j,kR,dimx,dimy);
        size_t src  = pack_idx(p,j,i,dimx,dimy);

        grid[dstL] = left_recv [src];
        grid[dstR] = right_recv[src];
    }
}

extern "C" void halo_pack_boundaries(const float* d_grid,
                                     int dimx,int dimy,int dimz,
                                     float* d_left_send,
                                     float* d_right_send)
{
    dim3 block(16,16);
    dim3 grid( (dimx+block.x-1)/block.x, (dimy+block.y-1)/block.y );
    k_pack<<<grid,block>>>(d_grid, dimx,dimy,dimz, d_left_send, d_right_send);
    cudaDeviceSynchronize();
}

extern "C" void halo_unpack_to_halos(float* d_grid,
                                     int dimx,int dimy,int dimz,
                                     const float* d_left_recv,
                                     const float* d_right_recv)
{
    dim3 block(16,16);
    dim3 grid( (dimx+block.x-1)/block.x, (dimy+block.y-1)/block.y );
    k_unpack<<<grid,block>>>(d_grid, dimx,dimy,dimz, d_left_recv, d_right_recv);
    cudaDeviceSynchronize();
}