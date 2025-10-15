# Heat-3D Single Turn (7-point stencil)

**Goal:** Implement one explicit time step of the 3D heat equation on an N×N×N grid.

## Task Description

Implement a single explicit time step of the 3D heat equation:

```
r = alpha * dt / (dx * dx)

out(i,j,k) = in(i,j,k) + r * (
    in(i-1,j,k) + in(i+1,j,k) +
    in(i,j-1,k) + in(i,j+1,k) +
    in(i,j,k-1) + in(i,j,k+1) -
    6 * in(i,j,k)
)
```

**Boundary handling:** Copy-through (Dirichlet hold) - for any cell with a neighbor out of bounds, `out = in` (boundaries remain unchanged this step).

## Files

- **Edit:** `student_kernel.cu` only
- **Test:** `make test_student` and `./test_student`
- **Reference:** `make test_reference` for validation

## Grading Criteria

- Exact match vs CPU oracle across multiple grid sizes
- Input immutability (no corruption of input arrays)
- Proper boundary handling (copy-through at domain edges)
- Guard canary validation (no out-of-bounds writes)
- Multi-step chaining correctness

## Test Coverage

- Grid sizes: {1,2,3,8,17,32,48}
- Multi-step sequences: {1,3} steps
- Multiple block configurations
- Adversarial input patterns
- CFL-stable parameters (alpha=0.01, dx=1.0, dt computed for stability)

## Notes

- Data type is `float`
- Comparison epsilon: `1e-4f`
- Kernel must not modify input buffer
- Tests include canary guards to detect out-of-bounds access