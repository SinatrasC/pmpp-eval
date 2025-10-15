// ch15-bfs-direction-optimized-single / reference_solution.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels_ref(int* __restrict__ level, int V){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) level[i] = INF_LVL;
}

__global__ void _clear_bitmap_ref(unsigned char* __restrict__ bm, int V){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) bm[i] = 0;
}

__global__ void _mark_bitmap_from_list_ref(const int* __restrict__ list, int n,
                                           unsigned char* __restrict__ bm){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = list[i];
        bm[v] = 1;
    }
}

__global__ void k_push_ref(const int* __restrict__ row_ptr,
                           const int* __restrict__ col_idx,
                           const int* __restrict__ frontier,
                           int frontier_size,
                           int* __restrict__ next_frontier,
                           int* __restrict__ next_size,
                           int* __restrict__ level,
                           int cur_level)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= frontier_size) return;

    int u = frontier[t];
    int beg = row_ptr[u], end = row_ptr[u+1];
    for (int e = beg; e < end; ++e) {
        int v = col_idx[e];
        if (atomicCAS(&level[v], INF_LVL, cur_level + 1) == INF_LVL) {
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

__global__ void k_pull_ref(const int* __restrict__ row_ptr,
                           const int* __restrict__ col_idx,
                           const unsigned char* __restrict__ in_frontier,
                           int* __restrict__ next_frontier,
                           int* __restrict__ next_size,
                           int* __restrict__ level,
                           int cur_level,
                           int V)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    if (level[v] != INF_LVL) return;

    int beg = row_ptr[v], end = row_ptr[v+1];
    for (int e = beg; e < end; ++e) {
        int u = col_idx[e];
        if (in_frontier[u]) {
            level[v] = cur_level + 1;
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
            break;
        }
    }
}

extern "C" void bfs_direction_optimized_gpu(const int* d_row_ptr,
                                            const int* d_col_idx,
                                            int V, int E,
                                            int src,
                                            int* d_level)
{
    if (V <= 0) return;

    dim3 b(256), g((V + b.x - 1)/b.x);

    _init_levels_ref<<<g,b>>>(d_level, V);
    cudaDeviceSynchronize();
    int zero = 0;
    cudaMemcpy(d_level + src, &zero, sizeof(int), cudaMemcpyHostToDevice);

    int *d_frontier=nullptr, *d_next_frontier=nullptr;
    int *d_next_size=nullptr;
    cudaMalloc(&d_frontier,      V*sizeof(int));
    cudaMalloc(&d_next_frontier, V*sizeof(int));
    cudaMalloc(&d_next_size,     sizeof(int));

    unsigned char *d_in_bm=nullptr;
    cudaMalloc(&d_in_bm, V*sizeof(unsigned char));
    _clear_bitmap_ref<<<g,b>>>(d_in_bm, V);

    cudaMemcpy(d_frontier, &src, sizeof(int), cudaMemcpyHostToDevice);
    int h_frontier_size = 1;
    bool mode_is_pull = false;
    int cur_level = 0;

    while (h_frontier_size > 0) {
        cudaMemset(d_next_size, 0, sizeof(int));

        if (!mode_is_pull) {
            dim3 kb(256), kg((h_frontier_size + kb.x - 1)/kb.x);
            k_push_ref<<<kg,kb>>>(d_row_ptr, d_col_idx,
                                  d_frontier, h_frontier_size,
                                  d_next_frontier, d_next_size,
                                  d_level, cur_level);
            cudaDeviceSynchronize();
        } else {
            _clear_bitmap_ref<<<g,b>>>(d_in_bm, V);
            dim3 mb(256), mg((h_frontier_size + mb.x - 1)/mb.x);
            _mark_bitmap_from_list_ref<<<mg,mb>>>(d_frontier, h_frontier_size, d_in_bm);
            cudaDeviceSynchronize();

            k_pull_ref<<<g,b>>>(d_row_ptr, d_col_idx,
                                d_in_bm,
                                d_next_frontier, d_next_size,
                                d_level, cur_level, V);
            cudaDeviceSynchronize();
        }

        int h_next = 0;
        cudaMemcpy(&h_next, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);

        if (!mode_is_pull && h_next > V/16) {
            mode_is_pull = true;
        } else if (mode_is_pull && h_next < V/64) {
            mode_is_pull = false;
        }

        std::swap(d_frontier, d_next_frontier);
        h_frontier_size = h_next;
        ++cur_level;
    }

    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_size);
    cudaFree(d_in_bm);
}