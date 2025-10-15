# ch14-spmv-csr-thread-per-row-single

Implement `spmv_csr_kernel` (**thread-per-row**) to compute **y = A*x** for CSR matrices:
- One thread sums one row (use a grid-stride loop over rows).
- Overwrite `y[row]` (no atomics).


## Build & run
```bash
make
./test_reference   # should pass all tests
./test_student     # should fail until you implement the kernel
```
