# ch20-mpi-stencil-pipeline-naive

**Goal:** Orchestrate a **naive single-iteration MPI-like stencil pipeline** across `P` slab partitions
(1D decomposition along `z`) on a single GPU **without real MPI**. You will:
1) compute Stage-1 boundary planes for each slab,
2) pack & "exchange" boundary results between neighbors (device-to-device copies),
3) unpack them into the neighbor halos, and
4) compute Stage-2 interior planes.

This produces the full **t+1** output field for a 3D **25-point (radius=4)** Jacobi-style stencil with
**copy-through boundaries** on all global faces (x, y, and the two global z faces).

We simulate MPI by giving you an entire global input field `d_in_full` and a requested `procs` count.
You must subdivide the global domain into equal-depth z-slabs and run the "pipeline" above per slab,
then gather owned planes back into `d_out_full`.

**Stencil (same as prior tasks)**
```
out(i,j,k) = w0*center + Σ_{d=1..4} w[d]*(±d along x/y/z)
w0=0.5, w1=0.10, w2=0.05, w3=0.025, w4=0.0125
```

**Domain & slabs**
- Global domain: `(dimx, dimy, dimz_total)` (no halos).
- Partition into `procs` equal slabs along z. Each slab `r` has depth `dz = dimz_total / procs`,
  global z range `[z0 .. z0+dz-1]` with `z0 = r*dz`.
- Each slab allocates local arrays with **4-deep halos** on both z sides → local z extent `dz + 8`.
- Local owned range is `k_local ∈ [4 .. 4+dz-1]` (maps to global `[z0 .. z0+dz-1]`).

**Stages**
- **Stage-1 (boundary planes only):** update owned planes within 4 of the slab ends:
  `[4..7]` and `[4+dz-4..4+dz-1]`.
  On **global faces** (where z neighbors would be out of global bounds), do **copy-through**.
  (Kernel receives `z_global_beg` and `dimz_total` to detect global faces.)
- **Pack/Exchange/Unpack:** pack 4 updated boundary planes on each side into contiguous buffers;
  device-to-device copy them into neighbor recv buffers; unpack into **neighbor halos** of **output** slabs.
- **Stage-2 (interior planes only):** update owned **interior** planes
  `k_local ∈ [8 .. (4+dz-1)-4]` with standard stencil; copy-through on x/y faces.
- **Gather:** write owned output planes (no halos) back to `d_out_full` at the correct global z offsets.

**What you implement (student_kernel.cu)**
- The **host function**:
  ```c++
  extern "C" void mpi_stencil_pipeline_naive(const float* d_in_full,
                                             float* d_out_full,
                                             int dimx,int dimy,int dimz_total,
                                             int procs);
  ```

Orchestrate the full 4-step pipeline for `procs` slabs as above. Use the provided kernels for:
Stage-1, Stage-2, pack, and unpack (already implemented in the file).
You must allocate/fill per-slab device buffers, launch kernels with reasonable configs,
simulate neighbor exchange with cudaMemcpy (device->device) between slabs, and gather.

**Provided helpers in both reference & student files**
* `stencil25_stage1_boundary(...)` — Stage-1 boundary update kernel launcher
* `stencil25_stage2_interior(...)` — Stage-2 interior update kernel launcher
* `halo_pack_boundaries(...)` — packs 4 planes/side from `out` into send buffers
* `halo_unpack_to_halos(...)` — unpacks recv buffers into **output halos**

**Tests**
* Deterministic CPU oracle for the full 3D update with copy-through on *all* global faces.
* Multiple sizes and `procs` (e.g., P=2 and P=3), mixed dimensions, guard canaries for OOB.
* We compare your final `d_out_full` against the oracle.

**Build & run**
```bash
make test_reference
make test_student
```

**Grading focus**
* Correct partitioning and ownership mapping.
* Correct stage ordering and plane ranges.
* Correct pack/exchange/unpack into halo **of outputs**.
* Exact match vs oracle & no OOB writes (canaries intact).