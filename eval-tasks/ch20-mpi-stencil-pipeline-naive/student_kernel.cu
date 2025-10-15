#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cstdio>

// ====== Utilities ======
static inline __host__ __device__
size_t idx3(int i,int j,int k,int dx,int dy){ return (size_t(k)*dy + j)*dx + i; }
static inline void ck(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

// ====== Provided stencil kernels & pack/unpack launchers ======
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

// ====== Small helpers to scatter/gather between full & slab memory ======
__global__ void k_scatter_from_full(const float* __restrict__ d_in_full,
                                    float* __restrict__ d_slab_in,
                                    int dimx,int dimy,int z0,int dz)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int t=blockIdx.z*blockDim.z+threadIdx.z; // local owned z [0..dz-1]
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
    int t=blockIdx.z*blockDim.z+threadIdx.z; // local owned z [0..dz-1]
    if(i>=dimx||j>=dimy||t>=dz) return;
    int k_local = 4 + t;
    int k_full  = z0 + t;
    d_out_full[idx3(i,j,k_full,dimx,dimy)] =
        d_slab_out[idx3(i,j,k_local,dimx,dimy)];
}

// ====== YOUR TASK ======
extern "C" void mpi_stencil_pipeline_naive(const float* d_in_full,
                                           float* d_out_full,
                                           int dimx,int dimy,int dimz_total,
                                           int procs)
{
    // TODO: Implement naive MPI-style pipeline for multi-slab 25-point stencil:
    //
    // Requirements:
    // - Partition z-dimension into `procs` slabs (each slab owns dimz_total/procs planes)
    // - Each slab needs halo regions (4 planes on each side for R=4 stencil)
    // - Allocate per-slab buffers with halo space: dimx * dimy * (local_dz + 8)
    // - Scatter input data from full array to per-slab buffers (use k_scatter_from_full kernel)
    // - Compute boundary planes with stage1 kernel (updates 4 planes on each end)
    // - Pack boundary data and exchange halos between neighboring slabs (simulate MPI with cudaMemcpy)
    // - Unpack received halo data into neighbor regions
    // - Compute interior planes with stage2 kernel
    // - Gather results from per-slab buffers back to full array (use k_gather_to_full kernel)
    // - Free all allocated buffers
    //
    // Hints:
    // - Use provided helper kernels: k_scatter_from_full, k_gather_to_full
    // - Use provided packing functions: halo_pack_boundaries, halo_unpack_to_halos
    // - Stage1 kernel handles boundary planes, Stage2 handles interior
    // - Halo exchange is device-to-device cudaMemcpy (simulates MPI)
    (void)d_in_full; (void)d_out_full; (void)dimx; (void)dimy; (void)dimz_total; (void)procs;
}