// ch15-bfs-edge-centric-single / student_kernel.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels_edge(int* __restrict__ level, int V, int src){
    // TODO: Initialize the level array and seed the source vertex.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        level[i] = INF_LVL;
    }
    if (i == src) {
        level[i] = 0;
    }
}

__global__ void bfs_edge_centric_kernel(const int* __restrict__ row_ptr,
                                        const int* __restrict__ col_idx,
                                        int* __restrict__ level,
                                        int cur_level,
                                        int E,
                                        int* __restrict__ active_found)
{
    // TODO: Iterate one edge per thread, discover newly reached vertices, and
    // signal progress via active_found.
    if (blockIdx.x * blockDim.x + threadIdx.x < E) {
        (void)row_ptr;
        (void)col_idx;
        (void)level;
        (void)cur_level;
        (void)active_found;
    }
}

extern "C" void bfs_edge_centric_gpu(const int* d_row_ptr,
                                     const int* d_col_idx,
                                     int V, int E,
                                     int src,
                                     int* d_level)
{
    // TODO: Alternate between edge-centric relaxations until no progress is
    // made.  Maintain device-side flags and handle empty graphs gracefully.
    if (V <= 0 || E < 0) {
        return;
    }

    dim3 b(128);
    dim3 g((V + b.x - 1) / b.x);
    _init_levels_edge<<<g, b>>>(d_level, V, src);
}
