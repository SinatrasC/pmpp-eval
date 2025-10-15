#include <cuda_runtime.h>

// TODOs:
// Implement two host wrappers that launch simple 2D grids:
//
// 1) halo_pack_boundaries(d_grid, dimx,dimy,dimz, d_left_send, d_right_send)
//    - Read 4 owned boundary planes and pack into left/right buffers.
//      Owned z range is [4 .. 4+dimz-1]. Pack planes:
//         left : k = 4 + p,           p ∈ [0..3]
//         right: k = (4+dimz-4) + p,  p ∈ [0..3]
//    - Packed layout is plane-major then row-major:
//         pack_idx(p,j,i) = (p*dimy + j)*dimx + i
//
// 2) halo_unpack_to_halos(d_grid, dimx,dimy,dimz, d_left_recv, d_right_recv)
//    - Write left_recv to left halo k = 0..3  (k = 0 + p)
//    - Write right_recv to right halo k = dimz+4 .. dimz+7 (k = dimz+4 + p)
//    - Same pack_idx layout for sources.
//
// Keep the rest of the grid untouched.

static inline __device__ size_t idx3(int i,int j,int k,int dx,int dy){
    return (size_t(k)*dy + j)*dx + i;
}
static inline __device__ size_t pack_idx(int p,int j,int i,int dx,int dy){
    return (size_t(p)*dy + j)*dx + i;
}

__global__ void k_pack_student(const float* __restrict__ grid,
                               int dimx,int dimy,int dimz,
                               float* __restrict__ left_send,
                               float* __restrict__ right_send)
{
    // TODO: implement (mirror reference description above)
}

__global__ void k_unpack_student(float* __restrict__ grid,
                                 int dimx,int dimy,int dimz,
                                 const float* __restrict__ left_recv,
                                 const float* __restrict__ right_recv)
{
    // TODO: implement (mirror reference description above)
}

extern "C" void halo_pack_boundaries(const float* d_grid,
                                     int dimx,int dimy,int dimz,
                                     float* d_left_send,
                                     float* d_right_send)
{
    // TODO: choose block(16,16) grid(ceil) and launch k_pack_student, then sync
}

extern "C" void halo_unpack_to_halos(float* d_grid,
                                     int dimx,int dimy,int dimz,
                                     const float* d_left_recv,
                                     const float* d_right_recv)
{
    // TODO: choose block(16,16) grid(ceil) and launch k_unpack_student, then sync
}