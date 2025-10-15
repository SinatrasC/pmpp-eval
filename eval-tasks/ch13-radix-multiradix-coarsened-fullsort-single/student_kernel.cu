// student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

static inline void CK(cudaError_t e,const char* m){
    if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

#ifndef RADIX_BITS
#define RADIX_BITS 4
#endif
#ifndef COARSENING_FACTOR
#define COARSENING_FACTOR 8
#endif
#ifndef BLOCK
#define BLOCK 256
#endif

// TODO: Implement a stable 4-bit LSD radix sort with thread coarsening (COARSENING_FACTOR).
// Sort must be in-place on `data` (you may use a temp buffer internally).

// Choose your radix size and coarsening factor
#define RADIX_SIZE (1 << RADIX_BITS)  // 2^RADIX_BITS buckets
#define RADIX_MASK (RADIX_SIZE - 1)

extern "C" __global__
void radix_sort_coarsened_kernel(unsigned int* __restrict__ data,
                                unsigned int* __restrict__ temp,
                                int n,
                                int shift)
{
    // TODO: Implement coarsened multi-radix sort pass
    // 1. Each thread loads COARSENING_FACTOR elements
    // 2. Count elements for each bucket using coarsened loading
    // 3. Compute prefix sums to find output positions for each bucket
    // 4. Scatter elements to correct positions based on radix value
    // 5. Ensure stable sorting and efficient memory access patterns
}

extern "C"
void radix_sort_coarsened_host(unsigned int* data, int n)
{
    if(n <= 1) return;
    // Placeholder: shallow copy (intentionally insufficient so tests fail until implemented)
    unsigned int* tmp=nullptr;
    CK(cudaMalloc(&tmp, n*sizeof(unsigned int)), "malloc tmp");
    CK(cudaMemcpy(tmp, data, n*sizeof(unsigned int), cudaMemcpyDeviceToDevice), "copy to tmp");
    CK(cudaMemcpy(data, tmp, n*sizeof(unsigned int), cudaMemcpyDeviceToDevice), "copy back");
    cudaFree(tmp);
}