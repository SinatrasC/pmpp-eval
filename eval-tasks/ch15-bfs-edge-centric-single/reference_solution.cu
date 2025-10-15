// ch15-bfs-edge-centric-single / reference_solution.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels_edge_ref(int* __restrict__ level, int V, int src){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) level[i] = INF_LVL;
    if (i == 0 && src >= 0 && src < V) {
        level[src] = 0;
    }
}

// Precompute edge-to-vertex mapping for efficiency
__global__ void _build_edge_to_vertex_ref(const int* __restrict__ row_ptr,
                                          int* __restrict__ edge_to_vertex,
                                          int V, int E)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    int beg = row_ptr[u];
    int end = row_ptr[u+1];
    for (int e = beg; e < end; ++e) {
        edge_to_vertex[e] = u;
    }
}

__global__ void bfs_edge_centric_kernel_ref(const int* __restrict__ row_ptr,
                                            const int* __restrict__ col_idx,
                                            const int* __restrict__ edge_to_vertex,
                                            int* __restrict__ level,
                                            int cur_level,
                                            int E,
                                            int* __restrict__ active_found)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= E) return;

    int u = edge_to_vertex[edge];
    int v = col_idx[edge];

    // Check if this edge crosses the frontier
    if (level[u] == cur_level && level[v] == INF_LVL) {
        level[v] = cur_level + 1;
        atomicMax(active_found, 1);
    }
}

extern "C" void bfs_edge_centric_gpu(const int* d_row_ptr,
                                     const int* d_col_idx,
                                     int V, int E,
                                     int src,
                                     int* d_level)
{
    if (V <= 0 || E <= 0) return;

    dim3 b(256), g((V + b.x - 1)/b.x);

    // Initialize levels
    _init_levels_edge_ref<<<g,b>>>(d_level, V, src);
    cudaDeviceSynchronize();

    // Build edge-to-vertex mapping for efficiency
    int *d_edge_to_vertex = nullptr;
    cudaMalloc(&d_edge_to_vertex, E * sizeof(int));
    _build_edge_to_vertex_ref<<<g,b>>>(d_row_ptr, d_edge_to_vertex, V, E);
    cudaDeviceSynchronize();

    // Edge-centric requires one thread per edge
    dim3 eb(256), eg((E + eb.x - 1)/eb.x);

    int *d_active = nullptr;
    cudaMalloc(&d_active, sizeof(int));

    int cur_level = 0;
    while (true) {
        cudaMemset(d_active, 0, sizeof(int));

        bfs_edge_centric_kernel_ref<<<eg,eb>>>(d_row_ptr, d_col_idx,
                                               d_edge_to_vertex,
                                               d_level, cur_level, E,
                                               d_active);
        cudaDeviceSynchronize();

        int h_active = 0;
        cudaMemcpy(&h_active, d_active, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_active == 0) break;

        ++cur_level;
    }

    cudaFree(d_active);
    cudaFree(d_edge_to_vertex);
}