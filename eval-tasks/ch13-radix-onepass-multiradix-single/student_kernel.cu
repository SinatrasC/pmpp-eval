// ch13-radix-onepass-multiradix-single / student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>

// CONTRACT:
// Implement one *stable* multiradix pass over keys.
// - keys_d:  input keys (length n)
// - out_d:   output keys (length n)
// - n:       number of elements
// - r:       bits per pass (1, 2, or 4)
// - shift:   bit shift for the digit (e.g., 0, r, 2r, ...)
// Approach expected (typical):
//   1) extract digits (0..(2^r - 1))
//   2) per-block histogram -> global array [grid x buckets]
//   3) host exclusive scan to get global bases & per-block bucket bases
//   4) stable scatter into out_d using digit, globalBase[b], blockBase[block,b], and local offset within block
// NOTE: Stability means equal digits preserve the original order.

extern "C" void radix_onepass_multiradix(
    const uint32_t* keys_d, uint32_t* out_d,
    int n, int r, int shift);

// TODO: provide your implementation
extern "C" void radix_onepass_multiradix(
    const uint32_t* keys_d, uint32_t* out_d,
    int n, int r, int shift)
{
    // Implement kernels + host prefix here.
    // You may choose blockDim=256 and compute grid from n.
    // (Any correct stable implementation passes.)
    (void)keys_d; (void)out_d; (void)n; (void)r; (void)shift;
}