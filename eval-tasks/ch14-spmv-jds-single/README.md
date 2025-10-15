# ch14-spmv-jds-single

Implement **Jagged Diagonal Storage (JDS)** SpMV via permuted diagonal traversal:
1) Rows are sorted by descending non-zero count and permuted accordingly.
2) `spmv_jds_kernel` uses grid-stride over permuted rows, accumulating across jagged diagonals.
3) Results are written back to original row positions via the permutation array.

Use the provided `spmv_jds(...)` host wrapper with appropriate grid/block configuration.


## Build & Run
```bash
make
./test_reference   # passes all tests
./test_student     # fails until you implement the kernel + wrapper
```
