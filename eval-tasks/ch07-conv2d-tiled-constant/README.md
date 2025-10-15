# Chapter 7 — 2D Convolution (Tiled + Constant) — Evaluation Task

This task evaluates a **textbook** CUDA implementation of **2D convolution using shared-memory tiling + constant memory**.

It mirrors the approach in your Chapter 7 notes (constant memory for filter; shared-memory tile with halo; zero padding) and aligns with your Python testing harness ideas.

## Files

- `student_kernel.cu` — **edit this**: implement `conv2d_tiled_constant_kernel`
- `reference_solution.cu` — working implementation (ground truth)
- `test_conv2d.cu` — CPU oracle, adversarial inputs, sentinels, immutability checks
- `Makefile` — builds `test_student` and `test_reference`

## Assumptions

- Grayscale float32 images (H×W), zero-padded borders
- Filter is square `(2r+1) × (2r+1)` (runtime `r`, with `r ≤ MAX_RADIUS`)
- Filter coefficients copied into **constant memory** via `setFilterConstant`
- Shared memory tile size: `(TILE + 2*r) × (TILE + 2*r)`

## Build & Run

```bash
# Reference: should pass all tests
make test_reference && ./test_reference

# Student (initially fails until you implement the kernel)
make test_student && ./test_student
```

## What's Graded

* Correctness vs CPU oracle (tight tolerance)
* Proper handling of image boundaries (zero padding)
* Input immutability (no unintended writes)
* Works for multiple sizes and radii (r ≤ MAX_RADIUS)

## Implementation Requirements

Students must implement the `conv2d_tiled_constant_kernel` function that:

1. **Uses shared memory tiling** with halo regions for boundary handling
2. **Accesses filter coefficients from constant memory** (`c_filter`)
3. **Implements zero padding** for out-of-bounds image access
4. **Cooperatively loads tiles** with all threads in the block
5. **Synchronizes properly** before computation phase
6. **Handles arbitrary image dimensions** and filter radii