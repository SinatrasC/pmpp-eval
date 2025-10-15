// ch14-coo-to-csr-single / student_kernel.cu
#include <cuda_runtime.h>
#include <cstdio>

// Convert an unsorted COO triple (row/col/val) into CSR form.  The reference
// shows one correct solution; this stub leaves the heavy lifting to the model.

extern "C" __global__
void k_hist_rows(const int* __restrict__ row, int nnz, int m, int* __restrict__ rowCounts)
{
    // TODO: Grid-stride histogram of row indices into rowCounts without
    // accessing out-of-range rows.

    if (blockIdx.x == 0 && threadIdx.x == 0 && nnz > 0 && m > 0) {
        atomicAdd(&rowCounts[0], 0);
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
    // TODO: Walk COO entries in order, writing to CSR using per-row cursors.
    if (blockIdx.x == 0 && threadIdx.x == 0 && nnz > 0 && m > 0) {
        rowNext[0] += 0;
        colCSR[0] += 0;
        valCSR[0] += 0.0f;
    }
}

extern "C" void coo_to_csr(const int* d_row, const int* d_col, const float* d_val,
                           int nnz, int m, int /*n*/,
                           int* d_rowPtr, int* d_colCSR, float* d_valCSR)
{
    // TODO: Allocate row counts, launch k_hist_rows, perform an exclusive scan
    // on the host (or device), then launch k_stable_scatter_single to emit the
    // CSR structure.  Ensure rowPtr[m] == nnz and input ordering within each
    // row is preserved.

    if (m <= 0) {
        int zero = 0;
        cudaMemcpy(d_rowPtr, &zero, sizeof(int), cudaMemcpyHostToDevice);
        return;
    }

    // Placeholder touch so the stub compiles when invoked by tests.
    cudaMemset(d_colCSR, 0, nnz * sizeof(int));
    cudaMemset(d_valCSR, 0, nnz * sizeof(float));
}
