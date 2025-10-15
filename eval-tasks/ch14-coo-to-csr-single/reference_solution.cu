// ch14-coo-to-csr-single / reference_solution.cu
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>

extern "C" __global__
void k_hist_rows(const int* __restrict__ row, int nnz, int m, int* __restrict__ rowCounts)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nnz;
         i += blockDim.x * gridDim.x)
    {
        int r = row[i];
        if (0 <= r && r < m) {
            atomicAdd(&rowCounts[r], 1);
        }
    }
}

extern "C" __global__
void k_stable_scatter_single(const int* __restrict__ row,
                             const int* __restrict__ col,
                             const float* __restrict__ val,
                             int nnz, int m,
                             int* __restrict__ rowNext,
                             int* __restrict__ colCSR,
                             float* __restrict__ valCSR)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < nnz; ++i) {
            int r = row[i];
            if (0 <= r && r < m) {
                int pos = rowNext[r]++;
                colCSR[pos] = col[i];
                valCSR[pos] = val[i];
            }
        }
    }
}

extern "C" void coo_to_csr(const int* d_row, const int* d_col, const float* d_val,
                           int nnz, int m, int /*n*/,
                           int* d_rowPtr, int* d_colCSR, float* d_valCSR)
{
    if (m < 0 || nnz < 0) return;
    if (m <= 0) {
        int zero = 0;
        cudaMemcpy(d_rowPtr, &zero, sizeof(int), cudaMemcpyHostToDevice);
        return;
    }

    int* d_rowCounts = nullptr;
    cudaMalloc(&d_rowCounts, m * sizeof(int));
    cudaMemset(d_rowCounts, 0, m * sizeof(int));

    dim3 block(256);
    auto cdiv = [](int a, int b){ return (a + b - 1)/b; };
    int gx = nnz > 0 ? std::max(1, cdiv(nnz, (int)block.x)) : 1;
    gx = std::min(gx, 65535);
    dim3 grid(gx);

    // 1) histogram
    k_hist_rows<<<grid, block>>>(d_row, nnz, m, d_rowCounts);
    cudaDeviceSynchronize();

    // 2) exclusive scan on host
    std::vector<int> h_counts(m);
    cudaMemcpy(h_counts.data(), d_rowCounts, m*sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> h_rowPtr(m+1, 0);
    for (int i = 0; i < m; ++i) h_rowPtr[i+1] = h_rowPtr[i] + h_counts[i];
    cudaMemcpy(d_rowPtr, h_rowPtr.data(), (m+1)*sizeof(int), cudaMemcpyHostToDevice);

    // 3) stable scatter (single-thread kernel)
    int* d_rowNext = nullptr;
    cudaMalloc(&d_rowNext, m * sizeof(int));
    cudaMemcpy(d_rowNext, h_rowPtr.data(), m*sizeof(int), cudaMemcpyHostToDevice);

    k_stable_scatter_single<<<1, 1>>>(d_row, d_col, d_val, nnz, m, d_rowNext, d_colCSR, d_valCSR);
    cudaDeviceSynchronize();

    cudaFree(d_rowCounts);
    cudaFree(d_rowNext);
}