# ch20-mpi-halo-pack-unpack

**Goal:** Implement GPU packing/unpacking of **radius-4 z-halo planes** for a 3D slab-decomposed domain used by the 25-point stencil.

We assume each MPI rank owns a local subdomain whose **owned z range** is `[4 .. 4+dimz-1]` inside a buffer with **4 halo planes** on each side, so the total z extent is `dimz + 8`.

- **Pack:** take the **left boundary** (the first 4 owned planes, k = 4..7) and the **right boundary** (the last 4 owned planes, k = (4+dimz-4)..(4+dimz-1)) and write them into two contiguous send buffers:
  - `left_send[p, j, i]` where `p = 0..3` corresponds to `k = 4 + p`
  - `right_send[p, j, i]` where `p = 0..3` corresponds to `k = (4+dimz-4) + p`
- **Unpack:** take `left_recv` and `right_recv` (same layout) and write into the **left halo** (`k = 0..3`) and **right halo** (`k = dimz+4 .. dimz+7`) respectively.

**Layout**
- Dense row-major: `idx(i,j,k) = (k*dimy + j)*dimx + i`
- Packed planes: `pack_idx(p,j,i) = (p*dimy + j)*dimx + i` for `p ∈ [0..3]`

**Files**
- `student_kernel.cu` — TODOs for `halo_pack_boundaries` and `halo_unpack_to_halos`
- `reference_solution.cu` — correct implementation
- `test_halo.cu` — deterministic tests (CPU oracle + guard canaries)
- `Makefile` — build targets

**Build & run**
```bash
make test_reference
make test_student
```

**What's graded**
- Exact match vs CPU oracle for both pack and unpack.
- No OOB writes (canary guards intact).
- Correct plane ordering & row-major packing.