# RGB → Grayscale — Single-Turn Task

## Goal
Implement the kernel that converts separate R/G/B 8-bit planes into an 8-bit grayscale plane:

```
gray[i] = clamp(round(0.299*R[i] + 0.587*G[i] + 0.114*B[i]), 0, 255)
```

for all `i < n`.

## Files
- `student_kernel.cu` — edit this file only
- `reference_solution.cu`
- `test_rgb2gray.cu`
- `Makefile`

## Build & Run
```bash
make
./test_reference   # should PASS
./test_student     # your implementation should PASS
```

## What the tests check

* Exact match to CPU oracle (same rounding/clamp)
* OOB safety (padding canaries)
* Inputs not modified
* CUDA error checks + varied block sizes