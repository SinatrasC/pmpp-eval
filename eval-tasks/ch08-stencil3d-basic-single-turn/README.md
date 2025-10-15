# Stencil 3D (Basic) — Single Turn

**Goal:** Implement a 7-point 3D stencil kernel with boundary copy-through.

## Task Description

Implement a basic 3D 7-point stencil computation:
- For interior points (1 ≤ i,j,k ≤ N-2): Apply 7-point stencil with coefficients c0-c6
- For boundary points: Copy input to output (out[idx] = in[idx])
- Handle edge cases (N=0, N=1) safely

## Files

- **Edit:** `student_kernel.cu` only
- **Test:** `make test_student` and `./test_student`
- **Reference:** `make test_reference` for validation

## Grading Criteria

- Exact match vs CPU oracle across multiple grid sizes
- Input immutability (no corruption of input arrays)
- Correct handling of boundary conditions
- Multiple block sizes and adversarial data patterns
- Edge cases (small N values)

## Test Coverage

- Grid sizes: {0,1,2,3,4,7,8,16,17,32}
- Block configurations: (8,8,8) and (4,8,16)
- Deterministic adversarial data patterns