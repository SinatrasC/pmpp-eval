// reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <algorithm>

static inline void CK(cudaError_t e, const char* m){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e));
        std::exit(2);
    }
}

#ifndef RADIX_BITS
#define RADIX_BITS 4
#endif
static_assert(RADIX_BITS >= 1 && RADIX_BITS <= 8, "RADIX_BITS must be in [1,8]");
constexpr int RADIX_SIZE = (1 << RADIX_BITS);      // 16 buckets for 4-bit
constexpr uint32_t RADIX_MASK = RADIX_SIZE - 1;

#ifndef BLOCK
#define BLOCK 256
#endif

// Kernel 1: Per-block, compute per-thread local rank (exclusive) for its bucket,
// and per-block bucket totals. Stable within block via shared 2D prefix scan.
__global__ void kLocalRanksAndCounts(const uint32_t* __restrict__ in,
                                     int n, int passShift,
                                     uint32_t* __restrict__ localRank,      // [n]
                                     uint32_t* __restrict__ blockCounts)    // [grid.x * RADIX_SIZE]
{
    __shared__ uint32_t s_flag[RADIX_SIZE][BLOCK]; // 16 * 256 * 4B = 16 KB
    int g0 = blockIdx.x * blockDim.x;
    int i  = g0 + threadIdx.x;

    // Initialize flags to 0
    for (int b=0; b<RADIX_SIZE; ++b) {
        s_flag[b][threadIdx.x] = 0u;
    }
    __syncthreads();

    // Set flag for my bucket if in-range
    uint32_t myBucket = 0;
    if (i < n) {
        uint32_t x = in[i];
        myBucket = (x >> passShift) & RADIX_MASK;
        s_flag[myBucket][threadIdx.x] = 1u;
    }
    __syncthreads();

    // Hillis–Steele scan per bucket across the block (inclusive)
    for (int ofs=1; ofs<blockDim.x; ofs <<= 1) {
        for (int b=0; b<RADIX_SIZE; ++b) {
            uint32_t add = (threadIdx.x >= ofs) ? s_flag[b][threadIdx.x - ofs] : 0u;
            __syncthreads();
            s_flag[b][threadIdx.x] += add;
            __syncthreads();
        }
    }

    // local exclusive rank for my element
    if (i < n) {
        uint32_t incl = s_flag[myBucket][threadIdx.x];  // inclusive
        localRank[i] = incl - 1u;                       // exclusive
    }
    // block totals per bucket (last lane writes)
    if (threadIdx.x == blockDim.x - 1) {
        for (int b=0; b<RADIX_SIZE; ++b) {
            // for partial tail blocks the scan already treated out-of-range as 0
            blockCounts[blockIdx.x * RADIX_SIZE + b] = s_flag[b][threadIdx.x];
        }
    }
}

// Kernel 2: Scatter to global positions using blockOffsets (global base for each block/bucket).
__global__ void kScatter(const uint32_t* __restrict__ in,
                         int n, int passShift,
                         const uint32_t* __restrict__ localRank,       // [n]
                         const uint32_t* __restrict__ blockOffsets,    // [grid.x * RADIX_SIZE], full global offsets
                         uint32_t* __restrict__ out)
{
    int g0 = blockIdx.x * blockDim.x;
    int i  = g0 + threadIdx.x;
    if (i >= n) return;

    uint32_t x = in[i];
    uint32_t bucket = (x >> passShift) & RADIX_MASK;
    uint32_t base   = blockOffsets[blockIdx.x * RADIX_SIZE + bucket];
    uint32_t pos    = base + localRank[i]; // stable within block
    out[pos] = x;
}

extern "C"
void radix_sort_multiradix_host(unsigned int* data, int n)
{
    if (n <= 1) return;

    // allocate temp buffer (same size) and per-pass scratch arrays
    uint32_t *d_tmp = nullptr;
    CK(cudaMalloc(&d_tmp, n * sizeof(uint32_t)), "malloc tmp");

    uint32_t *d_localRank = nullptr;
    CK(cudaMalloc(&d_localRank, n * sizeof(uint32_t)), "malloc localRank");

    // Launch config
    dim3 block(BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK);

    // Per-pass block counts and offsets (grid.x * RADIX_SIZE)
    const int B = grid.x;
    uint32_t *d_blockCounts  = nullptr;
    uint32_t *d_blockOffsets = nullptr;
    CK(cudaMalloc(&d_blockCounts,  B * RADIX_SIZE * sizeof(uint32_t)), "malloc blockCounts");
    CK(cudaMalloc(&d_blockOffsets, B * RADIX_SIZE * sizeof(uint32_t)), "malloc blockOffsets");

    // Host scratch
    std::vector<uint32_t> h_counts(B * RADIX_SIZE);
    std::vector<uint32_t> h_offsets(B * RADIX_SIZE);

    uint32_t* d_src = data;
    uint32_t* d_dst = d_tmp;

    const int passes = (32 + RADIX_BITS - 1) / RADIX_BITS; // e.g., 8 for 4-bit
    for (int pass = 0; pass < passes; ++pass) {
        int shift = pass * RADIX_BITS;

        // 1) Per-block local ranks and block bucket counts
        kLocalRanksAndCounts<<<grid, block>>>(d_src, n, shift, d_localRank, d_blockCounts);
        CK(cudaGetLastError(), "kLocalRanksAndCounts");
        CK(cudaDeviceSynchronize(), "sync ranks+counts");

        // 2) Build full global blockOffsets on host:
        CK(cudaMemcpy(h_counts.data(), d_blockCounts,
                      h_counts.size()*sizeof(uint32_t),
                      cudaMemcpyDeviceToHost), "D2H blockCounts");

        // totals per bucket
        uint32_t totals[RADIX_SIZE] = {0};
        for (int b=0; b<RADIX_SIZE; ++b){
            uint32_t t=0;
            for (int blk=0; blk<B; ++blk) t += h_counts[blk*RADIX_SIZE + b];
            totals[b] = t;
        }
        // base offset per bucket (prefix of totals)
        uint32_t base[RADIX_SIZE] = {0};
        for (int b=1; b<RADIX_SIZE; ++b) base[b] = base[b-1] + totals[b-1];

        // per-block offsets: base[b] + sum of counts for bucket b in all prior blocks
        for (int b=0; b<RADIX_SIZE; ++b){
            uint32_t run = base[b];
            for (int blk=0; blk<B; ++blk){
                h_offsets[blk*RADIX_SIZE + b] = run;
                run += h_counts[blk*RADIX_SIZE + b];
            }
        }

        CK(cudaMemcpy(d_blockOffsets, h_offsets.data(),
                      h_offsets.size()*sizeof(uint32_t),
                      cudaMemcpyHostToDevice), "H2D blockOffsets");

        // 3) Scatter (stable across entire grid)
        kScatter<<<grid, block>>>(d_src, n, shift, d_localRank, d_blockOffsets, d_dst);
        CK(cudaGetLastError(), "kScatter");
        CK(cudaDeviceSynchronize(), "sync scatter");

        // 4) Ping-pong buffers
        std::swap(d_src, d_dst);
    }

    // If final data not in-place, copy back
    if (d_src != data) {
        CK(cudaMemcpy(data, d_src, n*sizeof(uint32_t), cudaMemcpyDeviceToDevice), "copy back");
    }

    cudaFree(d_tmp);
    cudaFree(d_localRank);
    cudaFree(d_blockCounts);
    cudaFree(d_blockOffsets);
}