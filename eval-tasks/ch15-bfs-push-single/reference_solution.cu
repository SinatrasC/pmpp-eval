// ch15-bfs-push-single / reference_solution.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels_ref(int* __restrict__ level, int V, int src){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) level[i] = INF_LVL;
    if (i == 0 && src >= 0 && src < V) { /* src set below explicitly */ }
}

__global__ void bfs_push_kernel_ref(const int* __restrict__ row_ptr,
                                    const int* __restrict__ col_idx,
                                    const int* __restrict__ frontier,
                                    int frontier_size,
                                    int* __restrict__ next_frontier,
                                    int* __restrict__ next_frontier_size,
                                    int* __restrict__ level,
                                    int cur_level)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= frontier_size) return;

    int u = frontier[t];
    int beg = row_ptr[u];
    int end = row_ptr[u+1];
    for (int e = beg; e < end; ++e) {
        int v = col_idx[e];
        if (atomicCAS(&level[v], INF_LVL, cur_level + 1) == INF_LVL) {
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = v;
        }
    }
}

extern "C" void bfs_push_gpu(const int* d_row_ptr,
                             const int* d_col_idx,
                             int V, int E,
                             int src,
                             int* d_level)
{
    if (V <= 0) return;

    dim3 b(256), g((V + b.x - 1)/b.x);
    _init_levels_ref<<<g,b>>>(d_level, V, src);
    cudaDeviceSynchronize();
    int zero = 0;
    cudaMemcpy(d_level + src, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int *d_frontier = nullptr, *d_next_frontier = nullptr;
    int *d_next_size = nullptr;
    cudaMalloc(&d_frontier,      V * sizeof(int));
    cudaMalloc(&d_next_frontier, V * sizeof(int));
    cudaMalloc(&d_next_size,     sizeof(int));

    cudaMemcpy(d_frontier, &src, sizeof(int), cudaMemcpyHostToDevice);

    int h_frontier_size = 1;
    int cur_level = 0;

    while (h_frontier_size > 0) {
        cudaMemset(d_next_size, 0, sizeof(int));

        dim3 kb(256), kg((h_frontier_size + kb.x - 1) / kb.x);
        bfs_push_kernel_ref<<<kg, kb>>>(d_row_ptr, d_col_idx,
                                        d_frontier, h_frontier_size,
                                        d_next_frontier, d_next_size,
                                        d_level, cur_level);
        cudaDeviceSynchronize();

        int h_next = 0;
        cudaMemcpy(&h_next, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);

        std::swap(d_frontier, d_next_frontier);
        h_frontier_size = h_next;
        ++cur_level;
    }

    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_size);
}