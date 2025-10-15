# ch20-stencil-25pt-single-gpu-single

**Goal:** Implement a single-GPU 3D **25-point** stencil with **axis-aligned radius R=4** (halo depth 4 in each axis).
The stencil updates each interior cell `(i,j,k)` using the center value and its ±1..±4 neighbors **only along x, y, z axes** (no diagonals).
Boundary cells (within 4 cells of any face) are **copied through** from the input.

**Stencil definition (25 points):**
```
out(i,j,k) =
  w0 * in(i,j,k)
  + Σ_{d=1..4} w[d] * ( in(i-d,j,k) + in(i+d,j,k)
                      + in(i,j-d,k) + in(i,j+d,k)
                      + in(i,j,k-d) + in(i,j,k+d) )
```
with weights:
```
w0 = 0.5
w1 = 0.10,  w2 = 0.05,  w3 = 0.025,  w4 = 0.0125
```

**Contract:**
- Input/Output are **dense 3D arrays** of size `(dimx*dimy*dimz)` in **row-major** layout with index
  `idx = (k*dimy + j)*dimx + i`.
- Launch a 3D CUDA grid (you choose block/grid).
- Compute only cells with `i,j,k ∈ [R, dim-1-R]`. For others, copy `out = in`.
- No shared memory is required (correctness-focused).

**Files**
- `student_kernel.cu` — TODO skeleton for you to implement
- `reference_solution.cu` — working reference
- `test_single.cu` — deterministic test harness (CPU oracle, canary guards)
- `Makefile` — build targets

**Build & run**
```bash
make test_reference   # runs tests against reference_solution.cu
make test_student     # runs tests against student_kernel.cu
```

**What's graded**
- Numerical equality to CPU oracle (abs/rel tol 1e-5).
- No out-of-bounds writes (guard canaries stay intact).
- Boundary pass-through and interior-only writes respected.