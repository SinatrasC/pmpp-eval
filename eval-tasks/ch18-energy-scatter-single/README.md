# ch18-energy-scatter-single

Implement the **SCATTER** potential kernel (Fig. 18.5): one thread per atom, atomically accumulating into a z-slice of `energygrid`.

## Files
- `student_kernel.cu` — TODO kernel with detailed contract and hints.
- `reference_solution.cu` — Working baseline.
- `test_energy_scatter.cu` — Deterministic tests (CPU oracle, guard canaries, shuffled chunk order).
- `Makefile` — Builds `test_reference` and `test_student`.

## Build & Run
```bash
make
./test_reference
./test_student     # should FAIL until you implement the kernel
```

## Algorithm Description

The scatter approach uses one thread per atom to compute its contribution to all grid points:

1. **Thread Assignment**: Each thread processes one atom from constant memory
2. **Grid Traversal**: Loop over all (i,j) points in the target z-slice
3. **Distance Calculation**: Compute 3D Euclidean distance from atom to grid point
4. **Atomic Accumulation**: Use `atomicAdd` to safely accumulate contributions

## Key Requirements

* Use `__constant__ float atoms[CHUNK_SIZE*4]` (AoS: x,y,z,q).
* One thread per atom; each thread loops over all `(i,j)` in the target z-slice.
* Use `atomicAdd` on `energygrid[...]`.
* Compute `k = int(z/gridspacing)`; `x = i*gridspacing`, `y = j*gridspacing`.
* Respect bounds; avoid division by 0 (clamp denom by ~1e-12).

The harness uploads atoms in **chunks** (size `CHUNK_SIZE`) and launches your kernel once per chunk, accumulating results.

## Implementation Notes

* **Constant Memory**: Atoms uploaded in chunks via `cudaMemcpyToSymbol`
* **Atomicity**: Multiple threads may write to same grid cell - requires `atomicAdd`
* **Order Independence**: Test verifies results are identical regardless of chunk processing order
* **Guard Canaries**: Memory safety validation with sentinel values
* **Accuracy**: Relative tolerance ~1e-5 for floating-point comparison