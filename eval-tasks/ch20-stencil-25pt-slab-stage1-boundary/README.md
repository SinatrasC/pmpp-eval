# ch20-stencil-25pt-slab-stage1-boundary

**Goal:** Implement **Stage-1 boundary** update for a 3D axis-aligned **25-point** stencil (radius R=4) on a **slab with halos in z**.
Given device arrays `d_in` and `d_out` sized for `(dimx*dimy*(dimz+8))`, compute **only the first and last 4 owned planes** (z-slices) without needing freshly exchanged halos (Stage-1); do **not** write the interior planes.

**Local z layout (with 4-deep halos):**
```
[0..3]      : left  halo
[4..7]      : left  boundary  (OWNED, Stage-1)
[8..(8+dimz-1-8)] : interior (OWNED, Stage-2)
[dimz+4 .. dimz+7] : right boundary (OWNED, Stage-1)
[dimz+8 .. dimz+11]: right halo
```
For simplicity in this task we define **owned region** indices as `[4 .. 4+dimz-1]`.
Thus Stage-1 owns planes `z ∈ [4..7]` and `z ∈ [4+dimz-4 .. 4+dimz-1]`.

**Stencil (same as Task 1):**
```
out(i,j,k) = w0*center + Σ_{d=1..4} w[d]*(±d along x/y/z)
w0=0.5; w1=0.10; w2=0.05; w3=0.025; w4=0.0125
```

**Contract**
- Do **not** write halos or interior planes.
- For Stage-1 planes, use the full radius-4 neighborhood (which is available locally because halos are present).
- x/y boundaries: copy-through within Stage-1 regions (if i/j too close to faces).

**Files**
- `student_kernel.cu` — TODO skeleton
- `reference_solution.cu` — working Stage-1 implementation
- `test_stage1.cu` — deterministic test harness
- `Makefile`

**Build & run**
```bash
make test_reference
make test_student
```