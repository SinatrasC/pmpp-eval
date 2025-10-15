# reduction-max-arbitrary (single-turn)

Implement an arbitrary-length **maximum** reduction with:
- Grid-stride input scanning per thread (local max)
- Shared-memory tree reduction per block
- Race-free float atomic max at the end

## Contract
- Kernel: `__global__ void reduce_max_arbitrary(const float* in, float* out, int n)`
- Test initializes `out[0]` to `-INFINITY` and checks exact equality with CPU max.
- Only write to `out[0]` (atomic max). Do not modify inputs.

## Note on atomic max for float
Use a CAS loop on the integer bit pattern to implement `atomicMax` for `float`.

## Build & Run
```bash
make test_reference
make test_student
```