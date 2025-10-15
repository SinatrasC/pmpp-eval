// ch15-bfs-direction-optimized-single / student_kernel.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels(int* __restrict__ level, int V){
    // TODO: Initialize levels to INF_LVL.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        level[i] = INF_LVL;
    }
}

__global__ void _clear_bitmap(unsigned char* __restrict__ bm, int V){
    // TODO: Zero bitmap entries in parallel.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        bm[i] = 0;
    }
}

__global__ void _mark_bitmap_from_list(const int* __restrict__ list, int n,
                                       unsigned char* __restrict__ bm){
    // TODO: Mark current frontier vertices in the bitmap.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        (void)list[i];
        (void)bm;
    }
}

__global__ void k_push(const int* __restrict__ row_ptr,
                       const int* __restrict__ col_idx,
                       const int* __restrict__ frontier,
                       int frontier_size,
                       int* __restrict__ next_frontier,
                       int* __restrict__ next_size,
                       int* __restrict__ level,
                       int cur_level)
{
    // TODO: Push-based relaxation using atomicCAS/atomicAdd.
    if (row_ptr && col_idx && frontier && next_frontier && next_size && level) {
        (void)frontier_size;
        (void)cur_level;
    }
}

__global__ void k_pull(const int* __restrict__ row_ptr,
                       const int* __restrict__ col_idx,
                       const unsigned char* __restrict__ in_frontier,
                       int* __restrict__ next_frontier,
                       int* __restrict__ next_size,
                       int* __restrict__ level,
                       int cur_level,
                       int V)
{
    // TODO: Pull-based relaxation that scans undiscovered vertices.
    if (row_ptr && col_idx && in_frontier && next_frontier && next_size && level) {
        (void)cur_level;
        (void)V;
    }
}

extern "C" void bfs_direction_optimized_gpu(const int* d_row_ptr,
                                            const int* d_col_idx,
                                            int V, int E,
                                            int src,
                                            int* d_level)
{
    // TODO: Alternate between push and pull phases based on the frontier size
    // thresholds described in the README.
    if (V <= 0 || E < 0) {
        return;
    }

    dim3 b(256), g((V + b.x - 1) / b.x);
    _init_levels<<<g, b>>>(d_level, V);
    if (src >= 0 && src < V) {
        cudaMemcpy(d_level + src, &src, sizeof(int), cudaMemcpyHostToDevice);
    }
}
