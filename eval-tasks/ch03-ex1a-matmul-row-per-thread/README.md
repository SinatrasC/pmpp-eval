# Chapter 3 - Exercise 1a: Matrix Multiplication (Row-per-Thread)

## Problem Statement

Implement a CUDA kernel where each thread computes one complete row of the output matrix. This is a basic matrix multiplication approach where thread parallelism is organized at the row level.

## Task Description

Complete the `matrixMulRowKernel` function in `student_kernel.cu`. Each thread should:
1. Calculate its row index using `blockIdx.x * blockDim.x + threadIdx.x`
2. Guard against out-of-bounds access with `if (row < size)`
3. For each column in the output row, compute the dot product of the input row with the corresponding column
4. Write the result to the output matrix at the correct position

## Matrix Multiplication Formula
C[i,j] = Σ(k=0 to size-1) A[i,k] * B[k,j]

Where each thread computes all j values for a fixed i (row).

## Files

- `student_kernel.cu` - Your implementation (complete the TODO)
- `reference_solution.cu` - Working reference implementation
- `test_matmul_row.cu` - Comprehensive test harness with CPU oracle
- `Makefile` - Build configuration

## Usage

```bash
# Build and run student implementation
make run_student

# Build and run reference implementation  
make run_reference

# Clean build artifacts
make clean
```

## Test Coverage

The test harness validates:
- Multiple matrix sizes (0, 1, 5, 17, 31, 64, 65, 96)
- Various block sizes (64, 128, 256)
- Input immutability (inputs should not be modified)
- Numerical correctness against CPU oracle
- Proper handling of edge cases (empty matrices)

## Expected Output

```
RowKernel n=64 block=128 ... OK
RowKernel n=65 block=256 ... OK
...
```

All tests should pass with "OK" status for a correct implementation.