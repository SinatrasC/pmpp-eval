# Chapter 9 — Shared-Memory Privatized Histogram (Single Turn)

**Goal:** Implement a CUDA kernel that builds a histogram from an input array of `int` values in `[0, num_bins)` using **shared memory privatization** with global memory atomics for final merging.

## Task Description

Implement a privatized histogram kernel using shared memory per-block accumulation:


- Kernel signature (must match exactly):
  ```cpp
  __global__ void histogram_kernel(const int* in, unsigned int* hist,
                                   size_t N, int num_bins);
  ```

- Algorithm:
  1. Allocate `extern __shared__ unsigned int s_hist[]` of size `num_bins`
  2. Each block zeros its shared histogram cooperatively
  3. Grid-stride loop: each thread accumulates into shared memory (no atomics needed)
  4. Synchronize block, then cooperatively merge shared histogram to global with atomics
  5. For every `i in [0, N)`, if `0 <= in[i] < num_bins`, increment histogram

- No writes to `in`.
- No out-of-bounds reads/writes.

## Files

- **Edit:** `student_kernel.cu` only
- **Test:** `make test_student` and `./test_student`
- **Reference:** `make test_reference` for validation

## Grading Criteria

- Exact equality vs CPU oracle (64-bit counting internally)
- Input immutability (value + guard canaries)
- No out-of-bounds access (guard canaries around input and histogram)
- Multiple sizes, bin counts, and block sizes
- Shared memory usage validation

## Test Coverage

- Bin counts: {1,2,7,128,256,1024}
- Array sizes: {0,1,17,257,4093,1048576}
- Block sizes: {63,128,256}
- Adversarial patterns: uniform, skewed, extremes, random-ish
- Guard canary validation for both input and output buffers

## Implementation Requirements

- Use `extern __shared__ unsigned int s_hist[]` of size `num_bins`
- Cooperative shared memory initialization (zero)
- Grid-stride loop with shared memory accumulation
- Block synchronization before merging
- Cooperative global memory merge with `atomicAdd(&hist[bin], s_hist[bin])`
- Handle cases where `num_bins > blockDim.x`
- Do not modify input array
