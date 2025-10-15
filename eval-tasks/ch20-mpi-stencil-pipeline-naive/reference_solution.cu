#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

static inline __host__ __device__
size_t idx3(int i,int j,int k,int dx,int dy){ return (size_t(k)*dy + j)*dx + i; }

// ====================== Stencil Kernels (25-pt, R=4) ======================
__global__ void k_stage1_boundary(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int dimx,int dimy,int dz_local,
                                  int z_global_beg, int dimz_total)
{
    const int R=4;
    const int zOwnedBeg = 4;
    const int zOwnedEnd = 4 + dz_local - 1;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z; // local z including halos
    if(i>=dimx || j>=dimy || k>=dz_local+8) return;

    // boundary planes: first 4 and last 4 owned planes
    bool isBoundaryZ = (k>=zOwnedBeg && k<=zOwnedBeg+3) ||
                       (k>=zOwnedEnd-3 && k<=zOwnedEnd);
    if(!isBoundaryZ) return;

    // copy-through on x/y faces always
    if(i< R || i>=dimx-R || j< R || j>=dimy-R){
        out[idx3(i,j,k,dimx,dimy)] = in[idx3(i,j,k,dimx,dimy)];
        return;
    }

    // global z index for this local k
    int z_global = z_global_beg + (k - 4);

    // if the 4-neighborhood in z would leave the global domain => copy-through
    if(z_global - 4 < 0 || z_global + 4 >= dimz_total){
        out[idx3(i,j,k,dimx,dimy)] = in[idx3(i,j,k,dimx,dimy)];
        return;
    }

    const float w0=0.5f, w1=0.10f, w2=0.05f, w3=0.025f, w4=0.0125f;
    const float w[5]={w0,w1,w2,w3,w4};

    size_t p = idx3(i,j,k,dimx,dimy);
    float acc = w[0]*in[p];
    #pragma unroll
    for(int d=1; d<=4; ++d){
        acc += w[d]*( in[idx3(i-d,j,k,dimx,dimy)] + in[idx3(i+d,j,k,dimx,dimy)]
                    + in[idx3(i,j-d,k,dimx,dimy)] + in[idx3(i,j+d,k,dimx,dimy)]
                    + in[idx3(i,j,k-d,dimx,dimy)] + in[idx3(i,j,k+d,dimx,dimy)] );
    }
    out[p] = acc;
}

__global__ void k_stage2_interior(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int dimx,int dimy,int dz_local)
{
    const int R=4;
    const int zOwnedBeg = 4;
    const int zOwnedEnd = 4 + dz_local - 1;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i>=dimx || j>=dimy || k>=dz_local+8) return;

    // interior only (skip 4 planes near each local z end)
    if(!(k >= zOwnedBeg+4 && k <= zOwnedEnd-4)) return;

    if(i< R || i>=dimx-R || j< R || j>=dimy-R){
        out[idx3(i,j,k,dimx,dimy)] = in[idx3(i,j,k,dimx,dimy)];
        return;
    }

    const float w0=0.5f, w1=0.10f, w2=0.05f, w3=0.025f, w4=0.0125f;
    const float w[5]={w0,w1,w2,w3,w4};

    size_t p = idx3(i,j,k,dimx,dimy);
    float acc = w[0]*in[p];
    #pragma unroll
    for(int d=1; d<=4; ++d){
        acc += w[d]*( in[idx3(i-d,j,k,dimx,dimy)] + in[idx3(i+d,j,k,dimx,dimy)]
                    + in[idx3(i,j-d,k,dimx,dimy)] + in[idx3(i,j+d,k,dimx,dimy)]
                    + in[idx3(i,j,k-d,dimx,dimy)] + in[idx3(i,j,k+d,dimx,dimy)] );
    }
    out[p] = acc;
}

static void launch_stage1(const float* d_in, float* d_out,
                          int dimx,int dimy,int dz_local,
                          int z_global_beg, int dimz_total)
{
    dim3 block(8,8,8);
    dim3 grid((dimx+7)/8, (dimy+7)/8, ((dz_local+8)+7)/8);
    k_stage1_boundary<<<grid,block>>>(d_in, d_out, dimx,dimy,dz_local, z_global_beg, dimz_total);
}

static void launch_stage2(const float* d_in, float* d_out,
                          int dimx,int dimy,int dz_local)
{
    dim3 block(8,8,8);
    dim3 grid((dimx+7)/8, (dimy+7)/8, ((dz_local+8)+7)/8);
    k_stage2_interior<<<grid,block>>>(d_in, d_out, dimx,dimy,dz_local);
}

// ====================== Pack / Unpack (4 planes per side) ==================
static inline __host__ __device__
size_t pack_idx(int p,int j,int i,int dx,int dy){ return (size_t(p)*dy + j)*dx + i; }

__global__ void k_pack(const float* __restrict__ slab_out,
                       int dimx,int dimy,int dz_local,
                       float* __restrict__ left_send,
                       float* __restrict__ right_send)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i>=dimx || j>=dimy) return;

    const int zOwnedBeg = 4;
    const int zOwnedEnd = 4 + dz_local - 1;

    #pragma unroll
    for(int p=0;p<4;++p){
        int kL = zOwnedBeg + p;
        int kR = (zOwnedEnd - 3) + p;
        size_t sL = idx3(i,j,kL,dimx,dimy);
        size_t sR = idx3(i,j,kR,dimx,dimy);
        size_t d  = pack_idx(p,j,i,dimx,dimy);
        left_send [d] = slab_out[sL];
        right_send[d] = slab_out[sR];
    }
}

__global__ void k_unpack(float* __restrict__ slab_out,
                         int dimx,int dimy,int dz_local,
                         const float* __restrict__ left_recv,
                         const float* __restrict__ right_recv)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i>=dimx || j>=dimy) return;

    const int kHaloL = 0;
    const int kHaloR = dz_local + 4;

    #pragma unroll
    for(int p=0;p<4;++p){
        size_t s = pack_idx(p,j,i,dimx,dimy);
        if (left_recv)  slab_out[idx3(i,j,kHaloL+p,dimx,dimy)] = left_recv [s];
        if (right_recv) slab_out[idx3(i,j,kHaloR+p,dimx,dimy)] = right_recv[s];
    }
}

void stencil25_stage1_boundary(const float* d_in, float* d_out,
                               int dimx,int dimy,int dz_local,
                               int z_global_beg, int dimz_total)
{
    launch_stage1(d_in, d_out, dimx,dimy,dz_local, z_global_beg, dimz_total);
    cudaDeviceSynchronize();
}
void stencil25_stage2_interior(const float* d_in, float* d_out,
                               int dimx,int dimy,int dz_local)
{
    launch_stage2(d_in, d_out, dimx,dimy,dz_local);
    cudaDeviceSynchronize();
}
void halo_pack_boundaries(const float* d_slab_out,
                          int dimx,int dimy,int dz_local,
                          float* d_left_send, float* d_right_send)
{
    dim3 b(16,16); dim3 g((dimx+15)/16,(dimy+15)/16);
    k_pack<<<g,b>>>(d_slab_out, dimx,dimy,dz_local, d_left_send,d_right_send);
    cudaDeviceSynchronize();
}
void halo_unpack_to_halos(float* d_slab_out,
                          int dimx,int dimy,int dz_local,
                          const float* d_left_recv, const float* d_right_recv)
{
    dim3 b(16,16); dim3 g((dimx+15)/16,(dimy+15)/16);
    k_unpack<<<g,b>>>(d_slab_out, dimx,dimy,dz_local, d_left_recv,d_right_recv);
    cudaDeviceSynchronize();
}

// ====================== Pipeline Orchestration (reference) =================
static void ck(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

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

extern "C" void mpi_stencil_pipeline_naive(const float* d_in_full,
                                           float* d_out_full,
                                           int dimx,int dimy,int dimz_total,
                                           int procs)
{
    assert(procs>=1 && dimz_total%procs==0);
    int dz = dimz_total / procs;
    size_t Nplane = size_t(dimx)*dimy;
    size_t Nslab  = Nplane*(dz+8);
    size_t Npack  = Nplane*4;

    // Per-slab buffers
    std::vector<float*> d_in (procs,nullptr), d_out(procs,nullptr);
    std::vector<float*> d_Ls (procs,nullptr), d_Rs(procs,nullptr);
    std::vector<float*> d_Lr (procs,nullptr), d_Rr(procs,nullptr);

    for(int r=0;r<procs;++r){
        ck(cudaMalloc(&d_in[r],  Nslab*sizeof(float)),"malloc slab in");
        ck(cudaMalloc(&d_out[r], Nslab*sizeof(float)),"malloc slab out");
        ck(cudaMalloc(&d_Ls[r],  Npack*sizeof(float)),"malloc Ls");
        ck(cudaMalloc(&d_Rs[r],  Npack*sizeof(float)),"malloc Rs");
        ck(cudaMalloc(&d_Lr[r],  Npack*sizeof(float)),"malloc Lr");
        ck(cudaMalloc(&d_Rr[r],  Npack*sizeof(float)),"malloc Rr");
        // Seed both in/out with input copy for pass-through faces
        ck(cudaMemset(d_in[r],  0, Nslab*sizeof(float)),"memset in");
        ck(cudaMemset(d_out[r], 0, Nslab*sizeof(float)),"memset out");
    }

    // Scatter global input into local slabs (owned planes at k_local=[4..])
    {
        dim3 b(8,8,8);
        for(int r=0;r<procs;++r){
            int z0 = r*dz;
            dim3 g((dimx+7)/8, (dimy+7)/8, (dz+7)/8);
            k_scatter_from_full<<<g,b>>>(d_in_full, d_in[r], dimx,dimy, z0,dz);
        }
        ck(cudaDeviceSynchronize(),"sync scatter");
        // Copy in -> out initially (so untouched positions remain pass-through)
        for(int r=0;r<procs;++r){
            ck(cudaMemcpy(d_out[r], d_in[r], Nslab*sizeof(float), cudaMemcpyDeviceToDevice), "seed out");
        }
    }

    // --- Fill halos in IN FIRST (pack/exchange/unpack) ---
    // Pack boundary planes FROM IN (owned edge planes k_local=4..7 and k_local=dz-3..dz)
    for(int r=0;r<procs;++r){
        halo_pack_boundaries(d_in[r], dimx,dimy,dz, d_Ls[r], d_Rs[r]);
    }

    // Simulate MPI exchange (device->device copies)
    for(int r=0;r<procs;++r){
        int left  = (r>0)        ? r-1 : -1;
        int right = (r<procs-1)  ? r+1 : -1;
        if(left  >=0){ ck(cudaMemcpy(d_Rr[left], d_Ls[r], Npack*sizeof(float), cudaMemcpyDeviceToDevice),"xchg L->left.Rr"); }
        if(right >=0){ ck(cudaMemcpy(d_Lr[right],d_Rs[r], Npack*sizeof(float), cudaMemcpyDeviceToDevice),"xchg R->right.Lr"); }
    }

    // Unpack RECV buffers into IN halos; only sides that exist
    for(int r=0;r<procs;++r){
        const float* L = (r>0)        ? d_Lr[r] : nullptr;
        const float* R = (r<procs-1)  ? d_Rr[r] : nullptr;
        halo_unpack_to_halos(d_in[r], dimx,dimy,dz, L, R);
    }

    // --- Now Stage-1: boundary update on OUT, reading halos from IN ---
    for(int r=0;r<procs;++r){
        int z0 = r*dz;
        stencil25_stage1_boundary(d_in[r], d_out[r], dimx,dimy,dz, z0, dimz_total);
    }

    // Stage-2 interior update on OUT (using IN as input)
    for(int r=0;r<procs;++r){
        stencil25_stage2_interior(d_in[r], d_out[r], dimx,dimy,dz);
    }

    // Gather owned planes from OUT back to d_out_full
    {
        dim3 b(8,8,8);
        for(int r=0;r<procs;++r){
            int z0=r*dz; dim3 g((dimx+7)/8,(dimy+7)/8,(dz+7)/8);
            k_gather_to_full<<<g,b>>>(d_out[r], d_out_full, dimx,dimy, z0,dz);
        }
        ck(cudaDeviceSynchronize(),"sync gather");
    }

    for(int r=0;r<procs;++r){
        cudaFree(d_in[r]); cudaFree(d_out[r]); cudaFree(d_Ls[r]); cudaFree(d_Rs[r]); cudaFree(d_Lr[r]); cudaFree(d_Rr[r]);
    }
}