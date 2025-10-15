# Chapter 3 - Exercise 1b: Matrix Multiplication (Column-per-Thread)

## Problem Statement

Implement a CUDA kernel where each thread computes one complete column of the output matrix. This is an alternative matrix multiplication approach where thread parallelism is organized at the column level.

## Task Description

Complete the `matrixMulColKernel` function in `student_kernel.cu`. Each thread should:
1. Calculate its column index using `blockIdx.x * blockDim.x + threadIdx.x`
2. Guard against out-of-bounds access with `if (col < size)`
3. For each row in the output column, compute the dot product of the corresponding input row with the input column
4. Write the result to the output matrix at the correct position

## Matrix Multiplication Formula
C[i,j] = Σ(k=0 to size-1) A[i,k] * B[k,j]

Where each thread computes all i values for a fixed j (column).

## Files

- `student_kernel.cu` - Your implementation (complete the TODO)
- `reference_solution.cu` - Working reference implementation
- `test_matmul_col.cu` - Comprehensive test harness with CPU oracle
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
ColKernel n=64 block=128 ... OK
ColKernel n=65 block=256 ... OK
...
```

All tests should pass with "OK" status for a correct implementation.