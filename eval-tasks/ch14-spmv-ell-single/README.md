# ch14-spmv-ell-single

Implement `spmv_ell_kernel` for **ELL** SpMV (padding slots have `colIdx = -1`).


## Build & Run
```bash
make
./test_reference   # passes all tests
./test_student     # fails until you implement the kernel
```
