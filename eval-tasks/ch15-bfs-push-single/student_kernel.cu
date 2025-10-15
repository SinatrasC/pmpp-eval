// ch15-bfs-push-single / student_kernel.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels(int* __restrict__ level, int V, int src){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        level[i] = INF_LVL;
    }
    if (src >= 0 && src < V && i == src) {
        level[i] = 0;
    }
}

__global__ void bfs_push_kernel(const int* __restrict__ row_ptr,
                                const int* __restrict__ col_idx,
                                const int* __restrict__ frontier,
                                int frontier_size,
                                int* __restrict__ next_frontier,
                                int* __restrict__ next_frontier_size,
                                int* __restrict__ level,
                                int cur_level)
{
    // TODO: Visit outgoing edges of the frontier, atomically claim newly
    // discovered vertices, and append them to next_frontier.
    if (row_ptr && col_idx && frontier && next_frontier && next_frontier_size && level) {
        (void)frontier_size;
        (void)cur_level;
    }
}

extern "C" void bfs_push_gpu(const int* d_row_ptr,
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
    _init_levels<<<g, b>>>(d_level, V, src);
    // TODO: Allocate frontier buffers, iterate until the frontier is empty, and
    // free any resources allocated on the device.
}
