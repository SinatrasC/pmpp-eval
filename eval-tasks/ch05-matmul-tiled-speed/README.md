# Chapter 5 – Tiled Matrix Multiplication (Correctness + Speed Eval)

This task evaluates both **correctness** and **performance** of a tiled GEMM implementation.

## Files

- `student_kernel.cu` — **TODO** starter; implement a shared-memory tiled GEMM
- `reference_solution.cu` — Known-good **naive** and **tiled** reference kernels
- `test_matmul_speed.cu` — Test harness:
  - correctness: student vs reference-tiled (elementwise tolerance)
  - speed: student vs reference-tiled (cudaEvent timing, averaged)
  - optional timing of naive reference (context only)
- `Makefile`

## Build & Run

```bash
make            # builds test_student and test_reference
# or
make test       # builds + runs ./test_student

./test_student  # run your implementation (after make)
./test_reference # run reference mirror
```

Adjust tile size at build:

```bash
make TILE=16
```

## Pass Criteria

* **Correctness:** Student output ≈ reference tiled (rtol=1e-3, atol=1e-4)
* **Performance:** For large sizes (≥ 512 in all dims), student must be **≤ 1.25×** reference tiled runtime on average.


## What You Implement

Open **student_kernel.cu** and complete the `matmul_tiled_student_kernel` function:

- Use `__shared__` memory tiles for both input matrices A and B
- Implement proper boundary checking for non-tile-multiple dimensions
- Use `__syncthreads()` for proper thread synchronization
- Accumulate partial results across tile phases
- Write final result to global memory

## Expected Output Format

```
M=   0 N=   0 K=   0 | REF   0.000 ms | STU   0.000 ms | NAIVE   0.000 ms | OK
M=   1 N=   1 K=   1 | REF   0.045 ms | STU   0.000 ms | NAIVE   0.046 ms | FAIL
M=  63 N=  63 K=  61 | REF   0.234 ms | STU   0.000 ms | NAIVE   0.678 ms | FAIL
...
M=1024 N=1024 K=1024 | REF  45.123 ms | STU   0.000 ms | NAIVE 234.567 ms | FAIL

Correctness: FAIL
Performance (enforced only for >=512 dims): FAIL
```

The student implementation will show `FAIL` until properly implemented with shared memory tiling.
