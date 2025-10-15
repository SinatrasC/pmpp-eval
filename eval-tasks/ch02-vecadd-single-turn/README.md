# Vector Addition — Single-Turn Task

## Goal
Implement `vecAddKernel` so that for each valid index `i` it computes:

```
C[i] = A[i] + B[i]
```

## Files
- `student_kernel.cu` — edit this file only
- `reference_solution.cu` — working solution
- `test_vecadd.cu` — test harness
- `Makefile`

## Build & Run
```bash
make           # builds test_student and test_reference
./test_reference   # should PASS all tests
./test_student     # your implementation should PASS all tests
```

## What the tests check

* Correctness vs CPU oracle (multiple sizes)
* Guarding `i < n` (out-of-bounds protection)
* Canaries left/right of buffers untouched
* CUDA error checks + varied block sizes (128, 256)