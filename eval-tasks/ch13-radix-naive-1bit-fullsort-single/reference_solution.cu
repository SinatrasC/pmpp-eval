// reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <algorithm>

static inline void CK(cudaError_t e, const char* m){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int BLOCK = 256;

// K1: flagsZero[i] = 1 if ((x >> bit) & 1) == 0 else 0
__global__ void kFlagZeros(const uint32_t* __restrict__ in,
                           int n, int bit,
                           uint32_t* __restrict__ flagsZero) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = in[i];
    uint32_t b = (x >> bit) & 1u;
    flagsZero[i] = (b ^ 1u);
}

// K2: per-block inclusive scan (Hillis–Steele), write exclusive result,
//     also write block sum.
__global__ void kBlockExclusiveScan(const uint32_t* __restrict__ in,
                                    uint32_t* __restrict__ exScan,
                                    uint32_t* __restrict__ blockSums,
                                    int n) {
    __shared__ uint32_t sh[BLOCK];

    int g0 = blockIdx.x * blockDim.x;
    int i  = g0 + threadIdx.x;

    uint32_t v = (i < n) ? in[i] : 0u;
    sh[threadIdx.x] = v;
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        uint32_t t = 0u;
        if (threadIdx.x >= offset) t = sh[threadIdx.x - offset];
        __syncthreads();
        sh[threadIdx.x] += t;
        __syncthreads();
    }

    if (i < n) exScan[i] = sh[threadIdx.x] - v;

    if (threadIdx.x == BLOCK - 1) {
        int last = min(BLOCK, n - g0);
        uint32_t blkSum = (last > 0) ? sh[last - 1] : 0u;
        blockSums[blockIdx.x] = blkSum;
    }
}

// K3: add per-block offsets to make exScan global
__global__ void kAddBlockOffsets(uint32_t* __restrict__ exScan,
                                 const uint32_t* __restrict__ blockOffsets,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t off = blockOffsets[blockIdx.x];
    exScan[i] += off;
}

// K4: stable scatter by current bit
__global__ void kScatter(const uint32_t* __restrict__ in,
                         const uint32_t* __restrict__ exScanZero,
                         uint32_t totalZeros,
                         int n, int bit,
                         uint32_t* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = in[i];
    uint32_t zBefore = exScanZero[i];
    uint32_t b = (x >> bit) & 1u;

    uint32_t pos = (b == 0u) ? zBefore
                             : (totalZeros + (uint32_t)i - zBefore);
    out[pos] = x;
}

extern "C" void radix_sort_1bit_host(unsigned int* data, int n) {
    if (n <= 0) return;

    uint32_t *bufA = nullptr, *bufB = nullptr;
    CK(cudaMalloc(&bufA, n * sizeof(uint32_t)), "malloc bufA");
    CK(cudaMalloc(&bufB, n * sizeof(uint32_t)), "malloc bufB");
    CK(cudaMemcpy(bufA, data, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice), "copy input");

    uint32_t *d_flagsZero = nullptr;
    uint32_t *d_exScan    = nullptr;
    CK(cudaMalloc(&d_flagsZero, n * sizeof(uint32_t)), "malloc flags");
    CK(cudaMalloc(&d_exScan,    n * sizeof(uint32_t)), "malloc exScan");

    int numBlocks = (n + BLOCK - 1) / BLOCK;
    uint32_t *d_blockSums = nullptr, *d_blockOffsets = nullptr;
    CK(cudaMalloc(&d_blockSums,    numBlocks * sizeof(uint32_t)), "malloc blockSums");
    CK(cudaMalloc(&d_blockOffsets, numBlocks * sizeof(uint32_t)), "malloc blockOffsets");

    std::vector<uint32_t> h_block(numBlocks);

    for (int bit = 0; bit < 32; ++bit) {
        // flags
        kFlagZeros<<<numBlocks, BLOCK>>>(bufA, n, bit, d_flagsZero);
        CK(cudaGetLastError(), "kFlagZeros");

        // per-block exscan
        kBlockExclusiveScan<<<numBlocks, BLOCK>>>(d_flagsZero, d_exScan, d_blockSums, n);
        CK(cudaGetLastError(), "kBlockExclusiveScan");

        // host scan of block sums
        CK(cudaMemcpy(h_block.data(), d_blockSums, numBlocks*sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H blockSums");
        uint32_t totalZeros = 0, run = 0;
        for (int b = 0; b < numBlocks; ++b) {
            uint32_t s = h_block[b];
            h_block[b] = run;   // exclusive
            run += s;
        }
        totalZeros = run;
        CK(cudaMemcpy(d_blockOffsets, h_block.data(), numBlocks*sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D blockOffsets");

        // add offsets
        kAddBlockOffsets<<<numBlocks, BLOCK>>>(d_exScan, d_blockOffsets, n);
        CK(cudaGetLastError(), "kAddBlockOffsets");

        // scatter (stable)
        kScatter<<<numBlocks, BLOCK>>>(bufA, d_exScan, totalZeros, n, bit, bufB);
        CK(cudaGetLastError(), "kScatter");

        CK(cudaDeviceSynchronize(), "sync pass");
        std::swap(bufA, bufB);
    }

    CK(cudaMemcpy(data, bufA, n*sizeof(uint32_t), cudaMemcpyDeviceToDevice), "copy result");

    cudaFree(bufA); cudaFree(bufB);
    cudaFree(d_flagsZero); cudaFree(d_exScan);
    cudaFree(d_blockSums); cudaFree(d_blockOffsets);
}