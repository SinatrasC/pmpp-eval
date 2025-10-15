// reference_student_wrapper.cu
// Provides a matmul_student shim that forwards to the reference tiled kernel.
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

extern "C" void matmul_ref_tiled(const float* dA, const float* dB, float* dC,
                                 int M, int N, int K, int tile);

extern "C" void matmul_student(const float* dA, const float* dB, float* dC,
                                int M, int N, int K, int tile) {
    // Delegate to the reference implementation so the speed harness
    // exercises identical logic during reference validation runs.
    matmul_ref_tiled(dA, dB, dC, M, N, K, tile);
}
