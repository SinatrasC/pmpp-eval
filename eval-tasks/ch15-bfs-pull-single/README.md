# ch15-bfs-pull-single

Implement **vertex-centric PULL BFS** over a CSR graph using frontier bitmaps.

## Contract

- Device inputs:
  - `d_row_ptr[V+1]`, `d_col_idx[E]` (CSR, **must remain unchanged**)
- Output:
  - `d_level[V]`: BFS levels from `src` (`INF_LVL` for unreachable)
- API:
```c++
extern "C" void bfs_pull_gpu(const int* d_row_ptr,
                             const int* d_col_idx,
                             int V, int E,
                             int src,
                             int* d_level);
```

## Algorithm

- Maintain `in_frontier` and `out_frontier` bitmaps on device.
- For each level `L`: every undiscovered vertex `v` scans neighbors `u`;
  if any `u` is in `in_frontier`, set `level[v]=L+1`, mark `out_frontier[v]=1`.
- Count discoveries to know when to stop; then swap bitmaps.


## Build & Run

```bash
make
./test_reference
./test_student
```
