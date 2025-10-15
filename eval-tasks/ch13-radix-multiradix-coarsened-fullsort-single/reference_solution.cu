// reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <algorithm>

static inline void CK(cudaError_t e, const char* m){
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

static_assert(RADIX_BITS >= 1 && RADIX_BITS <= 8, "RADIX_BITS must be in [1,8]");
constexpr int RADIX_SIZE = (1 << RADIX_BITS);
constexpr uint32_t RADIX_MASK = RADIX_SIZE - 1;

// -------------------------------------------
// Kernel 1: Per-block bucket counts (coarsened)
// -------------------------------------------
__global__ void kBlockCounts(const uint32_t* __restrict__ in,
                             int n, int shiftBits,
                             uint32_t* __restrict__ blockCounts /* [grid.x * RADIX_SIZE] */)
{
    __shared__ uint32_t s_counts[RADIX_SIZE][BLOCK]; // 16*256*4B=16KB
    const int tid  = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    const int base = gtid * COARSENING_FACTOR;

    // zero per-thread counts
    for(int b=0;b<RADIX_SIZE;++b) s_counts[b][tid] = 0u;
    __syncthreads();

    // count my up to K items
    for(int k=0;k<COARSENING_FACTOR;++k){
        int idx = base + k;
        if(idx < n){
            uint32_t v = in[idx];
            uint32_t bucket = (v >> shiftBits) & RADIX_MASK;
            s_counts[bucket][tid] += 1u;
        }
    }
    __syncthreads();

    // thread 0 reduces per-bucket counts across threads
    if(tid == 0){
        for(int b=0;b<RADIX_SIZE;++b){
            uint32_t sum = 0u;
            for(int t=0;t<blockDim.x;++t) sum += s_counts[b][t];
            blockCounts[blockIdx.x * RADIX_SIZE + b] = sum;
        }
    }
}

// -----------------------------------------------------------
// Kernel 2: Stable scatter using host-computed blockOffsets
// -----------------------------------------------------------
__global__ void kScatterCoarsened(const uint32_t* __restrict__ in,
                                  int n, int shiftBits,
                                  const uint32_t* __restrict__ blockOffsets, // [grid.x*RADIX_SIZE]
                                  uint32_t* __restrict__ out)
{
    __shared__ uint32_t s_counts[RADIX_SIZE][BLOCK]; // per-thread counts
    const int tid  = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    const int base = gtid * COARSENING_FACTOR;

    // 1) per-thread counts per bucket
    for(int b=0;b<RADIX_SIZE;++b) s_counts[b][tid] = 0u;
    __syncthreads();

    uint32_t vals[COARSENING_FACTOR];
    uint32_t buckets[COARSENING_FACTOR];
    bool     valid[COARSENING_FACTOR];

    for(int k=0;k<COARSENING_FACTOR;++k){
        int idx = base + k;
        bool vld = (idx < n);
        valid[k] = vld;
        if(vld){
            uint32_t v = in[idx];
            vals[k]    = v;
            uint32_t b = (v >> shiftBits) & RADIX_MASK;
            buckets[k] = b;
            s_counts[b][tid] += 1u;
        }
    }
    __syncthreads();

    // 2) per-bucket scan across threads to get exclusive base per thread
    // Hillis–Steele inclusive scan on s_counts[b][*]
    for(int ofs=1; ofs < blockDim.x; ofs <<= 1){
        for(int b=0; b<RADIX_SIZE; ++b){
            uint32_t add = (tid >= ofs) ? s_counts[b][tid - ofs] : 0u;
            __syncthreads();
            s_counts[b][tid] += add;
            __syncthreads();
        }
    }
    // exclusive base for this thread per bucket = inclusive - myCount
    uint32_t myCount[RADIX_SIZE];
    for(int b=0;b<RADIX_SIZE;++b){
        // retrieve my own count by re-counting my local array (cheap)
        uint32_t cnt=0;
        for(int k=0;k<COARSENING_FACTOR;++k) if(valid[k] && buckets[k]==(uint32_t)b) ++cnt;
        myCount[b] = cnt;
        s_counts[b][tid] = s_counts[b][tid] - cnt; // now exclusive base per thread
    }
    __syncthreads();

    // 3) stable within-thread offsets: running local counters per bucket
    uint32_t localSeen[RADIX_SIZE]; for(int b=0;b<RADIX_SIZE;++b) localSeen[b]=0u;

    // 4) scatter in original index order (stability)
    for(int k=0;k<COARSENING_FACTOR;++k){
        if(!valid[k]) break;
        uint32_t b   = buckets[k];
        uint32_t baseGlobal = blockOffsets[blockIdx.x * RADIX_SIZE + b];
        uint32_t baseThread = s_counts[b][tid];           // exclusive base of this thread for bucket b
        uint32_t posInThread= localSeen[b]++;             // running order within-thread
        uint32_t finalPos   = baseGlobal + baseThread + posInThread;
        out[finalPos] = vals[k];
    }
}

// -------------------------------------------
// Host driver: stable LSD, 4-bit, coarsened
// -------------------------------------------
extern "C"
void radix_sort_coarsened_host(unsigned int* data, int n)
{
    if(n <= 1) return;

    // Launch config (threads cover ceil(n/K))
    const int threads = BLOCK;
    const int totalThreads = (n + COARSENING_FACTOR - 1) / COARSENING_FACTOR;
    const int blocks = (totalThreads + threads - 1) / threads;

    uint32_t *d_tmp=nullptr;
    CK(cudaMalloc(&d_tmp, n*sizeof(uint32_t)), "malloc tmp");

    // Per-pass scratch
    uint32_t *d_blockCounts=nullptr, *d_blockOffsets=nullptr;
    CK(cudaMalloc(&d_blockCounts,  blocks*RADIX_SIZE*sizeof(uint32_t)), "malloc blockCounts");
    CK(cudaMalloc(&d_blockOffsets, blocks*RADIX_SIZE*sizeof(uint32_t)), "malloc blockOffsets");

    std::vector<uint32_t> h_counts(blocks*RADIX_SIZE);
    std::vector<uint32_t> h_offsets(blocks*RADIX_SIZE);

    uint32_t* d_src = data;
    uint32_t* d_dst = d_tmp;

    const int passes = (32 + RADIX_BITS - 1) / RADIX_BITS; // 8 for 4-bit
    for(int p=0; p<passes; ++p){
        int shift = p * RADIX_BITS;

        // 1) per-block counts
        kBlockCounts<<<blocks, threads>>>(d_src, n, shift, d_blockCounts);
        CK(cudaGetLastError(), "kBlockCounts");
        CK(cudaDeviceSynchronize(), "sync counts");

        // 2) build global offsets on host
        CK(cudaMemcpy(h_counts.data(), d_blockCounts,
                      h_counts.size()*sizeof(uint32_t),
                      cudaMemcpyDeviceToHost),
           "D2H counts");

        // totals per bucket
        uint32_t totals[RADIX_SIZE]={0};
        for(int b=0;b<RADIX_SIZE;++b){
            uint32_t t=0;
            for(int blk=0; blk<blocks; ++blk) t += h_counts[blk*RADIX_SIZE + b];
            totals[b]=t;
        }
        // base per bucket
        uint32_t base[RADIX_SIZE]={0};
        for(int b=1;b<RADIX_SIZE;++b) base[b]=base[b-1]+totals[b-1];

        // prefix across blocks for each bucket
        for(int b=0;b<RADIX_SIZE;++b){
            uint32_t run=base[b];
            for(int blk=0; blk<blocks; ++blk){
                h_offsets[blk*RADIX_SIZE + b] = run;
                run += h_counts[blk*RADIX_SIZE + b];
            }
        }
        CK(cudaMemcpy(d_blockOffsets, h_offsets.data(),
                      h_offsets.size()*sizeof(uint32_t),
                      cudaMemcpyHostToDevice),
           "H2D offsets");

        // 3) stable scatter
        kScatterCoarsened<<<blocks, threads>>>(d_src, n, shift, d_blockOffsets, d_dst);
        CK(cudaGetLastError(), "kScatterCoarsened");
        CK(cudaDeviceSynchronize(), "sync scatter");

        std::swap(d_src, d_dst);
    }

    if(d_src != data){
        CK(cudaMemcpy(data, d_src, n*sizeof(uint32_t), cudaMemcpyDeviceToDevice), "copy back");
    }

    cudaFree(d_tmp);
    cudaFree(d_blockCounts);
    cudaFree(d_blockOffsets);
}