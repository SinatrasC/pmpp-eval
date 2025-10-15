#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cstdio>

static inline __host__ __device__
size_t idx3(int i,int j,int k,int dx,int dy){ return (size_t(k)*dy + j)*dx + i; }
static void ck(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

// Stage 1/2, pack/unpack launchers (provided)
extern "C" void stencil25_stage1_boundary(const float* d_in, float* d_out,
                                          int dimx,int dimy,int dz_local,
                                          int z_global_beg, int dimz_total);
extern "C" void stencil25_stage2_interior(const float* d_in, float* d_out,
                                          int dimx,int dimy,int dz_local);
extern "C" void halo_pack_boundaries(const float* d_slab_out,
                                     int dimx,int dimy,int dz_local,
                                     float* d_left_send, float* d_right_send);
extern "C" void halo_unpack_to_halos(float* d_slab_out,
                                     int dimx,int dimy,int dz_local,
                                     const float* d_left_recv, const float* d_right_recv);

// CUDA-aware sendrecv wrapper (device->device)
extern "C" void mpi_cudaaware_sendrecv_device(const float* d_sendbuf, int sendcount,
                                              float* d_recvbuf, int recvcount);

// scatter/gather kernels (provided)
__global__ void k_scatter_from_full(const float* __restrict__ d_in_full,
                                    float* __restrict__ d_slab_in,
                                    int dimx,int dimy,int z0,int dz)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int t=blockIdx.z*blockDim.z+threadIdx.z;
    if(i>=dimx||j>=dimy||t>=dz) return;
    int k_local = 4 + t;
    int k_full  = z0 + t;
    d_slab_in[idx3(i,j,k_local,dimx,dimy)] =
        d_in_full[idx3(i,j,k_full,dimx,dimy)];
}
__global__ void k_gather_to_full(const float* __restrict__ d_slab_out,
                                 float* __restrict__ d_out_full,
                                 int dimx,int dimy,int z0,int dz)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int t=blockIdx.z*blockDim.z+threadIdx.z;
    if(i>=dimx||j>=dimy||t>=dz) return;
    int k_local = 4 + t;
    int k_full  = z0 + t;
    d_out_full[idx3(i,j,k_full,dimx,dimy)] =
        d_slab_out[idx3(i,j,k_local,dimx,dimy)];
}

extern "C" void mpi_stencil_pipeline_cudaaware(const float* d_in_full,
                                               float* d_out_full,
                                               int dimx,int dimy,int dimz_total,
                                               int procs)
{
    // TODO: Implement CUDA-aware MPI pipeline for multi-slab 25-point stencil:
    //
    // Requirements:
    // - Partition z-dimension into `procs` slabs (each owns dimz_total/procs planes)
    // - Each slab needs halo regions (4 planes on each side for R=4 stencil)
    // - Allocate per-slab buffers with halo space and exchange buffers
    // - Scatter input data from full array to per-slab buffers (use k_scatter_from_full)
    // - Compute boundary planes with stage1 kernel
    // - Pack boundary data for halo exchange
    // - Use CUDA-aware MPI to exchange halos directly between GPU buffers (mpi_cudaaware_sendrecv_device)
    // - Unpack received halo data into neighbor regions
    // - Compute interior planes with stage2 kernel
    // - Gather results back to full array (use k_gather_to_full)
    // - Free all allocated buffers
    //
    // Hints:
    // - CUDA-aware MPI allows direct GPU-to-GPU transfers without host staging
    // - Use provided helper kernels: k_scatter_from_full, k_gather_to_full
    // - Use provided packing functions: halo_pack_boundaries, halo_unpack_to_halos
    // - mpi_cudaaware_sendrecv_device handles bidirectional exchange between neighbors
    (void)d_in_full; (void)d_out_full; (void)dimx; (void)dimy; (void)dimz_total; (void)procs;
}