# ch13-merge-path-fullsort-single

Implement stable GPU merge sort by iteratively doubling run width and merging with a merge-path kernel.

- Input: uint32_t keys
- Output: ascending, stable
- No external libs, fully self-contained

## Build & Run

```
make
./test_reference
./test_student
```

## Acceptance
- Exact match vs `std::stable_sort` for all tests
- Inputs unchanged; guard canaries intact