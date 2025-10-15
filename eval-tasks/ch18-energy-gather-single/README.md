# ch18-energy-gather-single

Implement the **GATHER** potential kernel (Fig. 18.6): one thread per grid cell, looping over the current atom chunk in constant memory, accumulating a private sum, and performing exactly one `+=` write.

## Files
- `student_kernel.cu` — TODO kernel with contract/hints.
- `reference_solution.cu` — Baseline implementation.
- `test_energy_gather.cu` — Deterministic tests (CPU oracle, guard canaries, sequential vs shuffled chunk order).
- `Makefile` — Builds `test_reference` and `test_student`.

## Build & Run
```bash
make
./test_reference
./test_student     # should FAIL until you implement the kernel
```

## Algorithm Description

The gather approach uses one thread per grid cell to accumulate contributions from all atoms:

1. **Thread Assignment**: Each thread processes one (i,j) grid cell in the target z-slice
2. **2D Launch**: Uses 2D thread blocks (e.g., 16x16) for natural grid mapping
3. **Atom Loop**: Loop over all atoms in the current constant memory chunk
4. **Private Accumulation**: Each thread maintains a private energy sum
5. **Single Write**: Write exactly once: `energygrid[idx] += local_sum`

## Key Requirements

* Use `__constant__ float atoms[CHUNK_SIZE*4]` for each uploaded chunk.
* One thread per output cell; NO atomics. Write exactly once: `energygrid[idx] += local_sum`.
* Correct indexing: `idx = grid.x*grid.y*k + grid.x*j + i`, with `k = int(z/gridspacing)`.

## Comparison with Scatter

| Aspect | Scatter | Gather |
|--------|---------|--------|
| Thread Assignment | One per atom | One per grid cell |
| Loop Target | Grid cells | Atoms |
| Memory Access | Atomic writes | Private accumulation |
| Launch Geometry | 1D blocks | 2D blocks |
| Synchronization | Required (atomics) | None needed |

## Implementation Notes

* **Constant Memory**: Efficient broadcast access pattern for atom data
* **No Atomics**: Each thread owns its output cell exclusively
* **Order Independence**: Results must be identical regardless of chunk order
* **Memory Coalescing**: 2D thread layout provides good memory access patterns
* **Accuracy**: Same tolerance as scatter version (~1e-5)