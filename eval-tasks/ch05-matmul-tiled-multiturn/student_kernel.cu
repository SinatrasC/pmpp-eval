// student_kernel.cu
// Implement a shared-memory tiled matrix multiply kernel:
//   C[M x K] = A[M x N] * B[N x K]
// TILE size is 16x16. Handle non-multiple sizes and out-of-bounds safely.

#include <cuda_runtime.h>

extern "C" void launch_student(const float* A, const float* B, float* C,
                               int M, int N, int K, int blockSize);

// TODO: Implement this kernel
__global__ void matmul_tiled_student(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K)
{
    // TODO: Implement shared-memory tiled matrix multiplication
    // REQUIRED: TILE = 16
    // 
    // Steps to implement:
    // 1. Define TILE size (16)
    // 2. Calculate 2D thread coordinates (row, col) in output matrix C
    // 3. Declare shared memory tiles for A and B submatrices  
    // 4. Initialize accumulator
    // 5. Loop over tiles along the inner dimension N:
    //    a. Cooperatively load A tile and B tile into shared memory
    //    b. Guard against out-of-bounds accesses (pad with zeros)
    //    c. Synchronize threads (__syncthreads())
    //    d. Compute partial products using shared memory tiles
    //    e. Synchronize threads again
    // 6. Write final result to global memory (with bounds checking)
}

extern "C" void launch_student(const float* A, const float* B, float* C,
                               int M, int N, int K, int /*blockSize*/)
{
    // TODO: Set up proper grid and block dimensions
    // Hint: Use 16x16 thread blocks, calculate grid size based on output dimensions
    
    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (M + 15) / 16);
    matmul_tiled_student<<<grid, block>>>(A, B, C, M, N, K);
}