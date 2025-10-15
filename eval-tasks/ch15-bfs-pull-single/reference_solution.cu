// ch15-bfs-pull-single / reference_solution.cu
#include <cuda_runtime.h>
#include <limits.h>

#ifndef INF_LVL
#define INF_LVL 0x3f3f3f3f
#endif

__global__ void _init_levels_pull_ref(int* __restrict__ level, int V, int src){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) level[i] = INF_LVL;
    if (i == 0 && src>=0 && src<V) { /* src set below */ }
}

__global__ void _clear_bitmap_ref(unsigned char* __restrict__ bm, int V){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) bm[i] = 0;
}

__global__ void _set_single_ref(unsigned char* __restrict__ bm, int idx){
    if (threadIdx.x==0 && blockIdx.x==0) bm[idx] = 1;
}

__global__ void bfs_pull_kernel_ref(const int* __restrict__ row_ptr,
                                    const int* __restrict__ col_idx,
                                    const unsigned char* __restrict__ in_frontier,
                                    unsigned char* __restrict__ out_frontier,
                                    int* __restrict__ level,
                                    int cur_level,
                                    int V,
                                    int* __restrict__ next_count)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    if (level[v] != INF_LVL) return;

    int beg = row_ptr[v];
    int end = row_ptr[v+1];
    for (int e = beg; e < end; ++e) {
        int u = col_idx[e];
        if (in_frontier[u]) {
            level[v] = cur_level + 1;
            out_frontier[v] = 1;
            atomicAdd(next_count, 1);
            break;
        }
    }
}

extern "C" void bfs_pull_gpu(const int* d_row_ptr,
                             const int* d_col_idx,
                             int V, int E,
                             int src,
                             int* d_level)
{
    if (V <= 0) return;

    dim3 b(256), g((V + b.x - 1)/b.x);
    _init_levels_pull_ref<<<g,b>>>(d_level, V, src);
    cudaDeviceSynchronize();
    int zero = 0;
    cudaMemcpy(d_level + src, &zero, sizeof(int), cudaMemcpyHostToDevice);

    unsigned char *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in,  V*sizeof(unsigned char));
    cudaMalloc(&d_out, V*sizeof(unsigned char));
    _clear_bitmap_ref<<<g,b>>>(d_in, V);
    _clear_bitmap_ref<<<g,b>>>(d_out, V);
    _set_single_ref<<<1,1>>>(d_in, src);

    int *d_next_count=nullptr; cudaMalloc(&d_next_count, sizeof(int));

    int cur_level=0;
    while (true) {
        cudaMemset(d_next_count, 0, sizeof(int));
        _clear_bitmap_ref<<<g,b>>>(d_out, V);

        bfs_pull_kernel_ref<<<g,b>>>(d_row_ptr, d_col_idx,
                                     d_in, d_out,
                                     d_level, cur_level, V,
                                     d_next_count);
        cudaDeviceSynchronize();

        int h_next=0; cudaMemcpy(&h_next, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_next==0) break;

        std::swap(d_in, d_out);
        ++cur_level;
    }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_next_count);
}