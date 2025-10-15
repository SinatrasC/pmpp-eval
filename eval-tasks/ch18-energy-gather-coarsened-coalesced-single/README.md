# ch18-energy-gather-coarsened-coalesced-single

Extend the coarsened gather kernel to use **shared-memory staging** and
**coalesced flushes**. Each block cooperatively accumulates energies in registers
then stages them in a 2-D shared-memory buffer before writing back to global
memory in a strided loop.

## Requirements
- Same atom chunk interface as the coarsened task (`atoms[CHUNK_SIZE*4]`).
- Threads compute `COARSEN_FACTOR` samples along X for their `(j,k)` row.
- Use shared memory to stage per-thread results and perform a coalesced flush.
- Respect bounds on `i`, `j`, and `k`. No atomics.

## Starter Code
`student_kernel.cu` is a minimal scaffold that zeros outputs. Recreate the
register accumulation, shared-memory staging, and coalesced writeback inspired
by `reference_solution.cu`.

## Build & Run
```bash
make
./test_reference
./test_student
```
