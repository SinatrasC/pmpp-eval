# ch15-bfs-edge-centric-single

**Task:** Edge-Centric BFS on GPU

**Chapter:** 15 (Graph Algorithms)

## Problem Description

Implement an **edge-centric BFS** that parallelizes over edges rather than vertices.

### Key Concepts
- **Edge-Centric Parallelism**: One thread per edge per iteration
- **Frontier Crossing**: For each edge `(u,v)`, check if `level[u] == cur_level && level[v] == INF_LVL`
- **Level Synchronization**: All edges at current frontier level are processed in parallel
- **Termination**: Continue until no new vertices are discovered

### Contract
- **Input**: CSR graph on device (`d_row_ptr[V+1]`, `d_col_idx[E]`) - **MUST remain unchanged**
- **Output**: `d_level[V]` = BFS distances from `src`, or `INF_LVL` if unreachable
- **Algorithm**:
  1. Initialize all levels to `INF_LVL`, set `level[src] = 0`
  2. For each iteration: parallel over ALL edges
  3. Check if edge `(u,v)` crosses frontier: `level[u] == cur_level && level[v] == INF_LVL`
  4. If crossing: set `level[v] = cur_level + 1` and signal activity
  5. Continue until no edges cross frontier

### Performance Characteristics
- **Good for**: Graphs with high edge parallelism, regular structure
- **Challenge**: Load balancing (some levels may have few active edges)
- **Memory Access**: Regular access pattern to edge arrays

### Files
- `student_kernel.cu`: TODO skeleton for students
- `reference_solution.cu`: Complete working implementation
- `test_bfs_edgecentric.cu`: Test harness with multiple graph types
- `Makefile`: Build configuration


### Testing
```bash
make test_student    # Build student version
make test_reference  # Build reference version
./test_student       # Run student tests
./test_reference     # Run reference tests (should pass)
```

The test harness validates correctness against CPU BFS oracle and checks for memory safety violations using guard canaries.

### Graph Test Cases
- **Chain graph** (32 vertices): Linear topology, few edges per level
- **Star graph** (128 vertices): All edges activate simultaneously after level 1
- **2D Grid** (8×8): Structured, moderate edge parallelism
- **Erdős–Rényi** (200 vertices, p=0.05): Random, varying edge workload per level

Expected output: `Summary: 4/4 passed`
