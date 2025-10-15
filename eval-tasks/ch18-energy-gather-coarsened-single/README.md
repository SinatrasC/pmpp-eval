# ch18-energy-gather-coarsened-single

Implement the **coarsened gather** kernel described in Chapter 18 (Molecular
Simulation). One thread owns `COARSEN_FACTOR` consecutive grid points along X on
fixed `(j,k)` slices and accumulates energy contributions from the atoms staged in
constant memory.

## Requirements
- Use the constant-memory atom buffer `atoms[CHUNK_SIZE*4]` (AoS: x, y, z, q).
- Derive `base_i` and `j` from block and thread indices, compute lattice `k` from
  the provided `z` coordinate.
- Accumulate contributions into a register array of length `COARSEN_FACTOR`.
- Write exactly once per owned grid cell (no atomics, no double counting).
- Clamp denominators with `fmaxf(denom, 1e-12f)`.

## Starter Code
`student_kernel.cu` now contains only bounds checks and placeholder writes.
Consult `reference_solution.cu` for the expected structure and fill in the full
implementation.

## Build & Run
```bash
make
./test_reference
./test_student
```
