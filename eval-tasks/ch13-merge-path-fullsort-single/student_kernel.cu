// ch13-merge-path-fullsort-single / student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>

// CONTRACT:
// Implement stable GPU merge sort using iterative merge passes.
// You must implement:
//   - merge_path_search (device): diagonal search
//   - merge_path_kernel (global): merges a slice [d0,d1)
//   - gpu_merge_sort (host): doubles width and ping-pongs buffers until sorted

__device__ int merge_path_search(const uint32_t* A, int nA,
                                 const uint32_t* B, int nB,
                                 int d)
{
    // TODO: diagonal binary search; return i (then j=d-i)
    return 0;
}

__global__ void merge_path_kernel(const uint32_t* __restrict__ A, int nA,
                                  const uint32_t* __restrict__ B, int nB,
                                  uint32_t* __restrict__ C)
{
    // TODO:
    //  - P = total threads
    //  - segment size seg = ceil((nA+nB)/P)
    //  - each thread t merges its slice [d0,d1)
    //  - compute (i0,j0) & (i1,j1) via merge_path_search
    //  - sequentially merge into C[d0..d1)
}

extern "C" void gpu_merge_sort(const uint32_t* d_in, uint32_t* d_out, int n)
{
    // TODO:
    //  - width = 1; ping-pong buffers
    //  - for width < n:
    //      * launch merges of adjacent runs [k..k+width) and [k+width..k+2*width)
    //  - final result copied to d_out
    (void)d_in; (void)d_out; (void)n;
}