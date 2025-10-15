// student_kernel.cu
#include <cuda_runtime.h>

// TODO: Implement parallel merge using merge-path (diagonal partition).
// Contract summary:
//  - Stable: on ties, choose A first
//  - Partition per thread using diagonals; then sequentially merge that slice
//  - Inputs A,B are sorted ascending; write C of length nA+nB

__device__ inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Find (i,j) on diagonal d (i+j = d) satisfying merge-path conditions.
// Returns i; j = d - i.
// Invariants:
//   lo = max(0, d - nB), hi = min(d, nA)
// Stable tie-breaking: A[i-1] <= B[j]  (and B[j-1] < A[i])
__device__ __forceinline__
int merge_path_search(const int* __restrict__ A, int nA,
                      const int* __restrict__ B, int nB,
                      int d)
{
    // TODO: Implement binary search to find merge-path coordinates
    // Return i such that (i, d-i) satisfies merge-path conditions
    return 0; // placeholder
}

extern "C" __global__
void merge_path_kernel(const int* __restrict__ A, int nA,
                       const int* __restrict__ B, int nB,
                       int* __restrict__ C)
{
    // TODO: Implement merge-path parallel merge
    // 1. Calculate thread's diagonal range [d0, d1)
    // 2. Find merge coordinates (i0,j0) and (i1,j1) using merge_path_search
    // 3. Sequentially merge A[i0..i1) and B[j0..j1) into C[d0..d1)
}