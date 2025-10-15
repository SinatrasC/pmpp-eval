# Chapter 6 — Thread-Coarsening Matrix Multiplication (Eval Task)

Each thread computes multiple output elements (coarsening) while using shared-memory tiling.
This task is functionally verifiable (no environment dependencies beyond CUDA).

## Files
- `student_kernel.cu` – starter with a TODO kernel + launcher students must implement
- `reference_solution.cu` – correct coarsened tiled implementation
- `test_matmul_coarsened.cu` – CPU oracle, adversarial sizes, input immutability, sentinels
- `Makefile` – builds student and reference test binaries

## Build
```bash
make            # builds two binaries: test_reference and test_student
```

## Run

```bash
./test_reference   # should PASS all tests
./test_student     # will FAIL until student_kernel.cu is completed
```

## What to implement (student)

Open `student_kernel.cu` and complete:

1. `__global__ void MatmulCoarsenedKernel(...)`
2. `extern "C" void launch_student(...)` to configure grid/block and launch

Keep these constants (edit only if instructed):

* `TILE_WIDTH` (default 16)
* `COARSE_FACTOR` (default 4)

## Notes

* Tensors use row-major: A [M×N], B [N×K], C [M×K].
* Grid dims: X covers `TILE_WIDTH*COARSE_FACTOR` columns; Y covers `TILE_WIDTH` rows.
* Boundary-safe loads and stores are required.