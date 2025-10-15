# ch14-spmv-coo-single

Implement `spmv_coo_kernel` to compute **y = A*x** where `A` is in **COO** format.
- Use `atomicAdd(&y[row], val * x[col])`.
- Grid-stride loop.
- Don't modify inputs.


## Build & run
```bash
make
./test_reference   # should pass all tests
./test_student     # should fail until you implement the kernel
```
