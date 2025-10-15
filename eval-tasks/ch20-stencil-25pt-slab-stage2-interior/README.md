# ch20-stencil-25pt-slab-stage2-interior

**Goal:** Implement **Stage-2 interior** update for the 3D axis-aligned **25-point** stencil (R=4) on a slab with 4-deep z halos. This stage updates only the **interior owned planes**; boundary planes were handled in Stage-1.

**Owned z range:** `[4 .. 4+dimz-1]`
**Stage-1 planes:** `[4..7]` and `[4+dimz-4 .. 4+dimz-1]`
**Stage-2 interior planes:** `[8 .. 4+dimz-1-4]` (i.e., `k ∈ [8 .. 4+dimz-5]`)

**Stencil (same as Task 1):**
```
out(i,j,k) = w0*center + Σ_{d=1..4} w[d]*(±d along x/y/z)
w0=0.5; w1=0.10; w2=0.05; w3=0.025; w4=0.0125
```

**Contract**
- Arrays sized for `(dimx*dimy*(dimz+8))`.
- **Write only** for interior planes (k in `[8..zEnd-4]`, where `zEnd=4+dimz-1`).
- For x/y faces inside those planes: **copy-through** (i or j within 4 of a face).
- Do not touch halos or Stage-1 planes.

**Files**
- `student_kernel.cu` — TODO skeleton
- `reference_solution.cu` — working implementation
- `test_stage2.cu` — deterministic test (CPU oracle + canaries)
- `Makefile`

**Build & run**
```bash
make test_reference
make test_student
```