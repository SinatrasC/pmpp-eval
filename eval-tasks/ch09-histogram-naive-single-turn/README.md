# Chapter 9 — Naive Global-Atomic Histogram (Single Turn)

**Goal:** Implement a CUDA kernel that builds a histogram from an input array of `int` values in `[0, num_bins)` using **global memory atomics** (no shared memory).

## Task Description

Implement a naive histogram kernel using global memory atomics:

- Kernel signature (must match exactly):
  ```cpp
  __global__ void histogram_kernel(const int* in, unsigned int* hist,
                                   size_t N, int num_bins);
  ```

- For every `i in [0, N)`, if `0 <= in[i] < num_bins`, atomically increment `hist[in[i]]`.
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

## Test Coverage

- Bin counts: {1,2,7,128,256,1024}
- Array sizes: {0,1,17,257,4093,1048576}
- Block sizes: {63,128,256}
- Adversarial patterns: uniform, skewed, extremes, random-ish
- Guard canary validation for both input and output buffers

## Implementation Requirements

- Use global memory `atomicAdd(&hist[bin], 1u)`
- Grid-stride loop over N
- Ignore out-of-range bin indices
- No shared memory usage
- Do not modify input array