# ch14-spmv-hyb-single

Implement **Hybrid ELL+COO** SpMV via two passes:
1) `spmv_ell_rows_kernel` (thread-per-row) writes `y[row] = sum(ELL row)`.
2) `spmv_coo_accum_kernel` grid-strides overflow COO entries and `atomicAdd`s into `y`.

Use the provided `spmv_hyb(...)` host wrapper to launch in the correct order.


## Build & Run
```bash
make
./test_reference   # passes all tests
./test_student     # fails until you implement the kernels + wrapper
```
