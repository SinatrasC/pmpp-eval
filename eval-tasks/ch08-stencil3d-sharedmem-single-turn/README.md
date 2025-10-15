# Stencil 3D (Shared Memory Tiled) — Single Turn

**Goal:** Implement a 3D shared-memory tiled 7-point stencil. Tile size `IN_TILE_DIM=8`, output region per block is `6×6×6`.

## Task Description

Implement a shared-memory optimized 3D 7-point stencil:
- Use `IN_TILE_DIM×IN_TILE_DIM×IN_TILE_DIM` (8×8×8) threads per block
- Load data cooperatively into shared memory with 1-cell halo
- Each block produces `OUT_TILE_DIM×OUT_TILE_DIM×OUT_TILE_DIM` (6×6×6) output values
- Interior threads compute 7-point stencil from shared memory
- Boundary handling: copy-through at domain edges

## Files

- **Edit:** `student_kernel.cu` only
- **Test:** `make test_student` and `./test_student`
- **Reference:** `make test_reference` for validation

## Grading Criteria

- Exact match vs CPU oracle across multiple grid sizes
- Input immutability (no corruption of input arrays)
- Correct shared memory tiling and halo handling
- Proper synchronization (__syncthreads())
- Edge cases including non-multiples of `OUT_TILE_DIM`

## Test Coverage

- Grid sizes: {0,1,2,3,4,6,8,10,16,18,32}
- Comprehensive boundary condition testing
- Deterministic adversarial data patterns
- Out-of-bounds access detection