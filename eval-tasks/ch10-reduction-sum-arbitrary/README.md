# reduction-sum-arbitrary (single-turn)

Implement an arbitrary-length sum reduction with:
- Grid-stride input accumulation per thread
- Shared-memory tree reduction per block
- `atomicAdd(out, block_sum)` to combine blocks

## Contract
- Kernel: `__global__ void reduce_sum_arbitrary(const float* in, float* out, int n)`
- Test picks various `n`, block sizes {128,256,512}, and grid sizes automatically.
- Only write to `out[0]` (atomic adds). Do not modify inputs.

## Build & Run
```bash
make test_reference
make test_student
```