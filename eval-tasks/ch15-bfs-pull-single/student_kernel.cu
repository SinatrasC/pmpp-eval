// ch15-bfs-pull-single / student_kernel.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels_pull(int* __restrict__ level, int V, int src){
    // TODO: Initialize the level array and set the source distance to zero.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        level[i] = INF_LVL;
    }
    if (src >= 0 && src < V && i == src) {
        level[i] = 0;
    }
}

__global__ void _clear_bitmap(unsigned char* __restrict__ bm, int V){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        bm[i] = 0;
    }
}

__global__ void _set_single(unsigned char* __restrict__ bm, int idx){
    if (threadIdx.x == 0 && blockIdx.x == 0 && idx >= 0) {
        bm[idx] = 1;
    }
}

__global__ void bfs_pull_kernel(const int* __restrict__ row_ptr,
                                const int* __restrict__ col_idx,
                                const unsigned char* __restrict__ in_frontier,
                                unsigned char* __restrict__ out_frontier,
                                int* __restrict__ level,
                                int cur_level,
                                int V,
                                int* __restrict__ next_count)
{
    // TODO: For undiscovered vertices, scan neighbors to detect if any reside
    // in the current frontier bitmap.
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < V) {
        (void)row_ptr;
        (void)col_idx;
        (void)in_frontier;
        (void)out_frontier;
        (void)level;
        (void)cur_level;
        (void)next_count;
    }
}

extern "C" void bfs_pull_gpu(const int* d_row_ptr,
                             const int* d_col_idx,
                             int V, int E,
                             int src,
                             int* d_level)
{
    (void)d_row_ptr;
    (void)d_col_idx;
    (void)E;

    if (V <= 0) {
        return;
    }

    dim3 b(256), g((V + b.x - 1) / b.x);
    _init_levels_pull<<<g, b>>>(d_level, V, src);
    // TODO: Allocate bitmaps, iterate levels until no new vertices are found.
}
