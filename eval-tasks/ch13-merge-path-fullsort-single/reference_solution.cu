// ch13-merge-path-fullsort-single / reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cassert>

static void ck(cudaError_t e,const char*m){
    if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2); }
}

// Stable merge-path diagonal search:
// Find i on diagonal d (i + j = d) such that:
//   A[i-1] <= B[j]    and    B[j-1] < A[i]
// (with sentinels A[-1] = MIN, A[nA] = MAX, same for B)
__device__ int merge_path_search(const uint32_t* A, int nA,
                                 const uint32_t* B, int nB,
                                 int d)
{
    int lo = max(0, d - nB);
    int hi = min(d, nA);

    while (lo <= hi) {
        int i = (lo + hi) >> 1;
        int j = d - i;

        // Sentinels
        uint32_t Ai_1 = (i > 0   ? A[i-1]      : 0u);
        uint32_t Ai   = (i < nA  ? A[i]        : 0xffffffffu);
        uint32_t Bj_1 = (j > 0   ? B[j-1]      : 0u);
        uint32_t Bj   = (j < nB  ? B[j]        : 0xffffffffu);

        // Stable boundary: left side uses <= , right side uses <
        if (Ai_1 <= Bj && Bj_1 < Ai) {
            return i;
        }
        // Move left if A[i-1] > B[j], else move right
        if (Ai_1 > Bj) {
            hi = i - 1;
        } else {
            lo = i + 1;
        }
    }
    // Fallback (shouldn't be reached with valid inputs)
    return lo;
}

__global__ void merge_path_kernel(const uint32_t* __restrict__ A, int nA,
                                  const uint32_t* __restrict__ B, int nB,
                                  uint32_t* __restrict__ C)
{
    int P   = gridDim.x * blockDim.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int nC  = nA + nB;
    if(P==0 || tid>=P) return;

    int seg = (nC + P - 1)/P;
    int d0  = tid*seg;
    if(d0>=nC) return;
    int d1  = min(d0+seg, nC);

    int i0 = merge_path_search(A,nA,B,nB,d0);
    int j0 = d0 - i0;
    int i1 = merge_path_search(A,nA,B,nB,d1);
    int j1 = d1 - i1;

    int i=i0, j=j0, k=d0;
    while(k<d1){
        bool takeA;
        if(i>=i1) takeA=false;
        else if(j>=j1) takeA=true;
        else           takeA = (A[i] <= B[j]); // stable
        C[k++] = takeA ? A[i++] : B[j++];
    }
}

static void launch_merge(const uint32_t* A, int nA,
                         const uint32_t* B, int nB,
                         uint32_t* C)
{
    if(nA==0 && nB==0) return;
    dim3 block(256);
    int nC = nA + nB;
    int P  = (nC + 63)/64;            // ~64 out elems per thread
    dim3 grid( (P + block.x - 1)/block.x );
    merge_path_kernel<<<grid,block>>>(A,nA,B,nB,C);
    ck(cudaGetLastError(),"merge");
}

extern "C" void gpu_merge_sort(const uint32_t* d_in, uint32_t* d_out, int n)
{
    if(n<=0){ return; }
    // two buffers for ping-pong
    uint32_t *bufA=nullptr, *bufB=nullptr;
    ck(cudaMalloc(&bufA, n*sizeof(uint32_t)), "malloc A");
    ck(cudaMalloc(&bufB, n*sizeof(uint32_t)), "malloc B");
    ck(cudaMemcpy(bufA, d_in, n*sizeof(uint32_t), cudaMemcpyDeviceToDevice), "copy in");

    for(int width=1; width<n; width<<=1){
        // merge runs of [k..k+width) and [k+width..k+2*width)
        for(int k=0; k<n; k += 2*width){
            int nA = min(width, n-k);
            int nB = min(width, max(0, n - (k+width)));
            const uint32_t* A = bufA + k;
            const uint32_t* B = bufA + k + width;
            uint32_t*       C = bufB + k;
            launch_merge(A,nA,B,nB,C);
        }
        ck(cudaDeviceSynchronize(),"sync pass");
        std::swap(bufA, bufB);
    }
    ck(cudaMemcpy(d_out, bufA, n*sizeof(uint32_t), cudaMemcpyDeviceToDevice), "copy out");
    cudaFree(bufA); cudaFree(bufB);
}