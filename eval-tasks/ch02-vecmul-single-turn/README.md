# Vector Multiplication CUDA Kernel - Single Turn Task

## Problem Statement

Implement the CUDA kernel `vecMulKernel` so that each thread computes one element of the output vector:
`C[i] = A[i] * B[i]` for `0 ≤ i < n`.

```cpp
__global__ void vecMulKernel(float* A, float* B, float* C, int n);
```

**Requirements:**
- Each thread computes one element (index `i`) if `i < n`
- `A`, `B`, and `C` are device pointers to `float`
- `n` is the number of elements
- Must handle sizes not divisible by block size
- Numerical tolerance: absolute error ≤ 1e-6

## Files

- `student_kernel.cu`: Starter file for students to edit (contains empty kernel)
- `test_vecmul.cu`: Complete test harness with 5 test cases
- `reference_solution.cu`: Reference implementation (keep on evaluator side only)
- `Makefile`: Build automation

## Usage

### Test with student implementation
```bash
make test_empty    # Should fail with empty implementation
```

### Test with reference solution  
```bash
make test_reference    # Should pass all tests
```

### Manual build and run
```bash
nvcc -O2 -o test_vecmul student_kernel.cu test_vecmul.cu
./test_vecmul
```

## Expected Output (Reference Solution)

```
Test n=0 ... OK
Test n=1 ... OK
Test n=17 ... OK
Test n=256 ... OK
Test n=3000 ... OK
```

**Exit code 0** = pass, **Exit code non-zero** = fail

## Test Cases

- **n=0**: Edge case (empty vectors)
- **n=1**: Single element
- **n=17**: Small non-multiple of block size (256)
- **n=256**: Exact multiple of block size  
- **n=3000**: Large non-multiple of block size

## Environment Requirements

- CUDA toolkit (tested with CUDA 12.0)
- `nvcc` compiler in PATH
- NVIDIA GPU with compute capability support