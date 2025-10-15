# ch15-bfs-push-single

Implement **vertex-centric PUSH BFS** over a CSR graph.

## Contract

- Device inputs:
  - `d_row_ptr[V+1]`, `d_col_idx[E]` (CSR, **must remain unchanged**)
- Output:
  - `d_level[V]`: BFS levels from `src` (`INF_LVL` for unreachable)
- API the test calls:
```c++
extern "C" void bfs_push_gpu(const int* d_row_ptr,
                             const int* d_col_idx,
                             int V, int E,
                             int src,
                             int* d_level);
```

## Algorithm

- While frontier non-empty: for each `u` in frontier, scan neighbors `v`.
- If `level[v]==INF_LVL`, set `level[v]=cur_level+1` (use `atomicCAS`) and enqueue `v`.


## Build & Run

```bash
make
./test_reference
./test_student
```

The reference must pass all tests. The student binary should fail until you implement the kernel.
