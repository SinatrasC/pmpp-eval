// ch13-radix-onepass-multiradix-single / reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

static void ck(cudaError_t e, const char* m){
    if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);}
}

// Extract r-bit digits from keys starting at bit position `shift`
__global__ void k_extract_digits(const uint32_t* __restrict__ keys,
                                 uint32_t* __restrict__ digits,
                                 int n, int mask, int shift)
{
    // Use contiguous tiling (each block covers a contiguous slice)
    int base = blockIdx.x * blockDim.x;
    int i = base + threadIdx.x;
    if(i < n){
        digits[i] = (keys[i] >> shift) & mask;
    }
}

// Per-block histogram using register-based per-thread histogram + shared reduction
__global__ void k_block_hist(const uint32_t* __restrict__ digits, int n,
                             int numBuckets,
                             uint32_t* __restrict__ blockHist) // [grid.x * numBuckets]
{
    extern __shared__ uint32_t sh_hist[];

    // Use contiguous tiling per block
    int base = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Initialize shared histogram
    for(int b = tid; b < numBuckets; b += blockDim.x) {
        sh_hist[b] = 0;
    }
    __syncthreads();

    // Each thread uses register-based histogram for its elements
    uint32_t reg_hist[16] = {0}; // Support up to 4-bit radix (16 buckets)
    assert(numBuckets <= 16);

    // Count elements in this block's range using registers
    for(int i = base + tid; i < n && i < base + blockDim.x; i += blockDim.x) {
        uint32_t digit = digits[i];
        reg_hist[digit]++;
    }

    // Reduce register histograms to shared memory
    for(int b = 0; b < numBuckets; b++) {
        atomicAdd(&sh_hist[b], reg_hist[b]);
    }
    __syncthreads();

    // Write block histogram to global memory
    if(tid < numBuckets) {
        blockHist[blockIdx.x * numBuckets + tid] = sh_hist[tid];
    }
}

// Compute stable local offsets using wave-by-wave scans (eliminates per-element atomics)
__global__ void k_local_offsets(const uint32_t* __restrict__ digits,
                                int n, int numBuckets,
                                uint32_t* __restrict__ local,
                                uint32_t* __restrict__ blockCounts) // [grid.x * numBuckets]
{
    extern __shared__ uint32_t sh_scan[];
    // sh_scan[numBuckets] - running count per bucket

    int base = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int elemCount = min(blockDim.x, n - base);

    // Initialize shared counters
    for(int b = tid; b < numBuckets; b += blockDim.x) {
        sh_scan[b] = 0;
    }
    __syncthreads();

    // Process elements wave-by-wave to maintain stability
    // Each wave processes blockDim.x elements simultaneously
    for(int wave = 0; wave < elemCount; wave += blockDim.x) {
        int i = base + wave + tid;
        uint32_t digit = 0;
        bool valid = (i < n && wave + tid < elemCount);

        if(valid) {
            digit = digits[i];
        }

        // Phase 1: All threads read current counter values
        uint32_t my_offset = 0;
        if(valid) {
            my_offset = sh_scan[digit];
        }
        __syncthreads();

        // Phase 2: Increment counters for this wave (one thread per bucket)
        if(tid < numBuckets) {
            uint32_t increment = 0;
            // Count how many elements in this wave belong to bucket tid
            for(int j = 0; j < blockDim.x && wave + j < elemCount; j++) {
                int idx = base + wave + j;
                if(idx < n && digits[idx] == tid) {
                    increment++;
                }
            }
            sh_scan[tid] += increment;
        }
        __syncthreads();

        // Phase 3: Compute stable local offset for each element
        if(valid) {
            // Count how many elements with same bucket appear before me in this wave
            uint32_t wave_offset = 0;
            for(int j = 0; j < tid && wave + j < elemCount; j++) {
                int idx = base + wave + j;
                if(idx < n && digits[idx] == digit) {
                    wave_offset++;
                }
            }
            local[i] = my_offset + wave_offset;
        }
        __syncthreads();
    }

    // Write final block counts
    if(tid < numBuckets) {
        blockCounts[blockIdx.x * numBuckets + tid] = sh_scan[tid];
    }
}

// Scatter elements to final positions
__global__ void k_scatter(const uint32_t* __restrict__ keys,
                          const uint32_t* __restrict__ digits,
                          const uint32_t* __restrict__ local,
                          int n, int numBuckets,
                          const uint32_t* __restrict__ globalBase,   // [numBuckets]
                          const uint32_t* __restrict__ blockBase,    // [grid.x * numBuckets]
                          uint32_t* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        uint32_t digit = digits[i];
        uint32_t pos = globalBase[digit] + blockBase[blockIdx.x * numBuckets + digit] + local[i];
        out[pos] = keys[i];
    }
}

extern "C" void radix_onepass_multiradix(
    const uint32_t* keys_d, uint32_t* out_d,
    int n, int r, int shift)
{
    if(n <= 0) return;
    assert(r == 1 || r == 2 || r == 4);
    int numBuckets = 1 << r;
    int mask = numBuckets - 1;

    // Use contiguous tiling: each block processes a contiguous slice
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    grid.x = std::min(grid.x, 1024u); // Cap grid size

    // Allocate scratch arrays
    uint32_t *digits_d = nullptr, *local_d = nullptr;
    uint32_t *blockHist_d = nullptr, *blockCounts_d = nullptr, *blockBase_d = nullptr;
    ck(cudaMalloc(&digits_d,      n * sizeof(uint32_t)), "malloc digits");
    ck(cudaMalloc(&local_d,       n * sizeof(uint32_t)), "malloc local");
    ck(cudaMalloc(&blockHist_d,   grid.x * numBuckets * sizeof(uint32_t)), "malloc blockHist");
    ck(cudaMalloc(&blockCounts_d, grid.x * numBuckets * sizeof(uint32_t)), "malloc blockCounts");
    ck(cudaMalloc(&blockBase_d,   grid.x * numBuckets * sizeof(uint32_t)), "malloc blockBase");

    // 1) Extract digits
    k_extract_digits<<<grid, block>>>(keys_d, digits_d, n, mask, shift);
    ck(cudaGetLastError(), "extract");

    // 2) Compute per-block histograms
    k_block_hist<<<grid, block, numBuckets * sizeof(uint32_t)>>>(digits_d, n, numBuckets, blockHist_d);
    ck(cudaGetLastError(), "hist");

    // 3) Compute global and per-block base offsets on host
    std::vector<uint32_t> h_blockHist(grid.x * numBuckets);
    std::vector<uint32_t> h_globalBase(numBuckets);
    std::vector<uint32_t> h_blockBase(grid.x * numBuckets);

    ck(cudaMemcpy(h_blockHist.data(), blockHist_d,
                  h_blockHist.size() * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost), "D2H blockHist");

    // Compute global base (exclusive scan of total counts per bucket)
    uint32_t total_processed = 0;
    for(int b = 0; b < numBuckets; b++) {
        h_globalBase[b] = total_processed;
        uint32_t bucket_total = 0;
        for(unsigned blk = 0; blk < grid.x; blk++) {
            bucket_total += h_blockHist[blk * numBuckets + b];
        }
        total_processed += bucket_total;
    }

    // Compute per-block base offsets (exclusive scan within each bucket)
    for(int b = 0; b < numBuckets; b++) {
        uint32_t running_count = 0;
        for(unsigned blk = 0; blk < grid.x; blk++) {
            h_blockBase[blk * numBuckets + b] = running_count;
            running_count += h_blockHist[blk * numBuckets + b];
        }
    }

    // Upload base arrays to device
    uint32_t *globalBase_d = nullptr;
    ck(cudaMalloc(&globalBase_d, numBuckets * sizeof(uint32_t)), "malloc globalBase");
    ck(cudaMemcpy(globalBase_d, h_globalBase.data(),
                  numBuckets * sizeof(uint32_t),
                  cudaMemcpyHostToDevice), "H2D globalBase");
    ck(cudaMemcpy(blockBase_d, h_blockBase.data(),
                  h_blockBase.size() * sizeof(uint32_t),
                  cudaMemcpyHostToDevice), "H2D blockBase");

    // 4) Compute stable local offsets
    k_local_offsets<<<grid, block, numBuckets * sizeof(uint32_t)>>>(
        digits_d, n, numBuckets, local_d, blockCounts_d);
    ck(cudaGetLastError(), "local_offsets");

    // 5) Scatter to final positions
    k_scatter<<<grid, block>>>(keys_d, digits_d, local_d, n, numBuckets,
                               globalBase_d, blockBase_d, out_d);
    ck(cudaGetLastError(), "scatter");
    ck(cudaDeviceSynchronize(), "sync");

    // Cleanup
    cudaFree(digits_d);
    cudaFree(local_d);
    cudaFree(blockHist_d);
    cudaFree(blockCounts_d);
    cudaFree(blockBase_d);
    cudaFree(globalBase_d);
}