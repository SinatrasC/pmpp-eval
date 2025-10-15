# ch14-coo-to-csr-single

Implement a **stable** COO → CSR conversion on the GPU.

## Spec

Given COO arrays:
- `row[nnz]`, `col[nnz]`, `val[nnz]` (unsorted, may include duplicates)

Produce CSR:
- `rowPtr[m+1]`, `colCSR[nnz]`, `valCSR[nnz]`


### Requirements
- **Stability**: For entries within the same row, preserve their original COO order.
- **No combining** duplicates (each COO triplet becomes one CSR entry).
- **Bounds**: `rowPtr[m] == nnz`.
- **Robustness**: Must handle empty rows and any input ordering.

### Hints
A simple, correct approach:
1. **Device histogram** of row indices → `rowCounts[m]` (with atomics).
2. **Host exclusive scan** on `rowCounts` → `rowPtr[m+1]` (copy back to device).
3. **Stable scatter**: Use a **single-thread device kernel** that walks `i=0..nnz-1`
   and writes to `colCSR[rowPtr[row[i]] + k]` using a per-row cursor (`rowNext`).

The tests check:
- Exact match vs a CPU **stable** COO→CSR conversion.
- **Guard canaries** detect OOB writes; inputs must remain unchanged.
- A secondary **CSR SpMV** on the result is compared with a CPU oracle.

## Build & Run
```bash
make
./test_reference   # reference must pass all
./test_student     # student (initially) will fail until implemented
```
