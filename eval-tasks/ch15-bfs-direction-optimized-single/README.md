# ch15-bfs-direction-optimized-single

**Task:** Direction-Optimized (hybrid push↔pull) BFS on GPU

**Chapter:** 15 (Graph Algorithms)

## Problem Description

Implement a **direction-optimized BFS** that dynamically switches between push and pull strategies based on frontier size heuristics.

### Key Concepts
- **Push Mode**: For each vertex `u` in frontier, explore neighbors `v` (good for narrow frontiers)
- **Pull Mode**: For each undiscovered vertex `v`, check if any neighbor `u` is in frontier (good for wide frontiers)
- **Dynamic Switching**:
  - Switch to pull when `frontier_size > V/16`
  - Switch back to push when `frontier_size < V/64`

### Contract
- **Input**: CSR graph on device (`d_row_ptr[V+1]`, `d_col_idx[E]`) - **MUST remain unchanged**
- **Output**: `d_level[V]` = BFS distances from `src`, or `INF_LVL` if unreachable
- **Algorithm**:
  1. Start in PUSH mode with frontier = {src}
  2. Apply heuristic switching based on frontier size
  3. Use frontier arrays + bitmap (for pull mode)
  4. Use atomic operations for race-free discovery

### Files
- `student_kernel.cu`: TODO skeleton for students
- `reference_solution.cu`: Complete working implementation
- `test_bfs_diropt.cu`: Test harness with multiple graph types
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
- **Chain graph** (64 vertices): Stays mostly in push mode
- **Star graph** (1024 vertices): Triggers pull mode (frontier ≈ V)
- **2D Grid** (16×16): May switch based on heuristic
- **Erdős–Rényi** (400 vertices, p=0.03): Mixed behavior

Expected output: `Summary: 4/4 passed`
