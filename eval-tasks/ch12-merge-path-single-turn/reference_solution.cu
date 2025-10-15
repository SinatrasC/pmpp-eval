// reference_solution.cu
#include <cuda_runtime.h>
#include <limits.h>

__device__ __forceinline__
int merge_path_search(const int* __restrict__ A, int nA,
                      const int* __restrict__ B, int nB,
                      int d)
{
    int lo = max(0, d - nB);
    int hi = min(d, nA);

    while (lo < hi) {
        int i = (lo + hi) >> 1;
        int j = d - i;

        int Ai_1 = (i > 0)  ? A[i - 1] : INT_MIN;
        int Bj   = (j < nB) ? B[j]     : INT_MAX;

        if (Ai_1 > Bj) {
            hi = i;
        } else {
            lo = i + 1;
        }
    }

    return lo;
}

extern "C" __global__
void merge_path_kernel(const int* __restrict__ A, int nA,
                       const int* __restrict__ B, int nB,
                       int* __restrict__ C)
{
    const int P = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int nC = nA + nB;
    if (P <= 0) return;

    const int seg = (nC + P - 1) / P;
    int d0 = tid * seg;
    if (d0 >= nC) return;
    int d1 = min(d0 + seg, nC);

    int i0 = merge_path_search(A, nA, B, nB, d0);
    int j0 = d0 - i0;
    int i1 = merge_path_search(A, nA, B, nB, d1);
    int j1 = d1 - i1;

    // Sequential merge of assigned ranges
    int i = i0, j = j0, k = d0;
    while (k < d1) {
        bool take_A;
        if (i >= i1) {
            take_A = false;
        } else if (j >= j1) {
            take_A = true;
        } else {
            // Both valid, stable comparison: A[i] <= B[j] means take A
            take_A = (A[i] <= B[j]);
        }

        if (take_A) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
}