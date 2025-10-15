# reduction-sum-2048 (single-turn)

Implement a convergent shared-memory reduction that sums **exactly 2048** `float`
values into `out[0]`. The test launches with **1 block** of **1024 threads**,
and each thread loads two elements.

## Contract
- Kernel: `__global__ void reduce_sum_2048(const float* in, float* out)`
- Grid: `<<<1, 1024>>>`
- Input length: exactly 2048 floats
- Only write `out[0]`; do **not** write anywhere else
- Use convergent reduction: `for (stride = blockDim.x/2; stride >= 1; stride >>= 1)`

## Build & Run
```bash
make test_reference   # should pass all tests
make test_student     # fails until you implement student_kernel.cu
```