# ch21-bezier-dp-parent-child-single

Implement a classic **dynamic parallelism** (DP) parent/child pipeline that tessellates quadratic Bézier curves.

## What you implement (student_kernel.cu)
- `__device__ float curvature_of(const float2 P0, const float2 P1, const float2 P2)`
- `__global__ void computeBezierLines_parent(BezierLine* bLines, int nLines, int maxTess)`
  - For each line `lidx`, compute curvature, choose `nVertices` in `[4, maxTess]`
  - **Device-side allocation**: `bLines[lidx].vertexPos = (float2*)malloc(nVertices*sizeof(float2))`
  - Launch child grid:
    `computeBezierLine_child<<<(nVertices+31)/32, 32>>>(lidx, bLines, nVertices)`
- `__global__ void computeBezierLine_child(int lidx, BezierLine* bLines, int nTess)`


## Build
```
make
```

This produces:
- `test_reference` (links `reference_solution.cu`)
- `test_student`   (links `student_kernel.cu`)

## Run
```
./test_reference
./test_student
```

Both binaries should report all tests **OK**.

## Constraints
- No Thrust, no Python.
- Deterministic math; compare against a CPU oracle (tolerance 1e-6).
- Compile with dynamic parallelism: `-rdc=true -lcudadevrt`

## Files
- `student_kernel.cu` – TODO skeleton
- `reference_solution.cu` – complete DP implementation
- `test_bezier_dp.cu` – deterministic tests + CPU oracle
- `Makefile`