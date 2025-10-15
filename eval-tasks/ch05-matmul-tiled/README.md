# Chapter 5 — Tiled Matrix Multiplication (CUDA)

Implement a tiled, shared-memory matrix-matrix multiplication kernel.

## Task
Edit **student_kernel.cu** so that `launch_tiled_matmul()` launches a correct
shared-memory tiled kernel that computes P = M × N for arbitrary (m, n, o).

Your kernel must:
- Work for non-multiples of TILE (handle boundaries safely).
- Keep inputs immutable (do not write to M or N).
- Produce results that match a CPU oracle within epsilon (1e-4).

We compile and test you with a deterministic harness:
- Multiple sizes (0/1/small/large; non-multiples of TILE)
- Adversarial value patterns
- Input immutability checks
- Output guard bands for OOB detection

## Build & Run
```bash
# Build and run against the reference (should pass)
make test_reference && ./test_reference

# Build and run against the student starter (will fail until implemented)
make test_student && ./test_student
```

## What you implement

Open **student_kernel.cu** and complete:

* `__global__ void TiledMatMulKernel(...)`
* `void launch_tiled_matmul(...)`

Use TILE = 16 (already defined). You must use shared memory and `__syncthreads()`.

## Notes

* We time nothing here—this is a *correctness* task.
* Don't modify function signatures or test harness logic.
* Keep inputs `const float*` and never write to them.