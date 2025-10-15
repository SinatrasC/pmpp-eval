# merge-path-single-turn (single-turn)

Implement a **parallel merge** of two *sorted* int arrays `A` (len `nA`) and `B` (len `nB`)
into `C` (len `nA+nB`) using the **merge-path (diagonal partition)** method.

## Contract
- Kernel signature:
  ```c++
  __global__ void merge_path_kernel(const int* A, int nA,
                                    const int* B, int nB,
                                    int* C)
  ```

- Stable merge: on ties, **take from A first** (i.e., `<=`).
- Each thread computes a diagonal range `[d0, d1)` with:
  - `P = gridDim.x * blockDim.x`
  - `seg = ceil((nA + nB)/P)`
  - `d0 = min(t * seg, nA + nB)`, `d1 = min(d0 + seg, nA + nB)` where `t` is global thread id
- For each `d` (d0 and d1), find `(i,j)` s.t. `i+j=d` and:
  - `i in [max(0, d - nB), min(d, nA)]`
  - `A[i-1] <= B[j]` and `B[j-1] < A[i]` (with bounds checks)
- Then **sequentially merge** from `(i0,j0)` to fill `C[d0..d1)`.

## Build & Run
```bash
make test_reference   # passes all tests
make test_student     # fails until student implements kernel
```

The test harness checks:
- Correctness vs CPU oracle
- Input immutability (A,B unchanged)
- Guard canaries (detect OOB writes)
- Many edge/adversarial cases (empty, imbalanced sizes, duplicates)