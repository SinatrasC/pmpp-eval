# ch20-mpi-stencil-pipeline-cudaaware

**Goal:** Implement a *CUDA-aware MPI–style* halo exchange pipeline for one 25-point (radius=4) Jacobi update across `P`
slab partitions (1D decomposition along `z`) — **but in a single process**. We *simulate* CUDA-aware MPI by providing a
device-to-device "sendrecv" wrapper that must be used for halo traffic (no host staging).

This task is the "CUDA-aware" twin of `ch20-mpi-stencil-pipeline-naive`:
- Stage-1: compute **boundary planes** (4 planes on each owned side per slab).
- Exchange: pack 4 updated planes/side, **device-to-device sendrecv** into neighbors' receive buffers.
- Unpack: write received planes into the **output halos**.
- Stage-2: compute **interior planes**.
- Gather: assemble owned planes into the global output.

**Stencil (same as earlier tasks)**

```
out(i,j,k) = w0*center + Σ_{d=1..4} w[d]*(±d along x/y/z)
w0=0.5, w1=0.10, w2=0.05, w3=0.025, w4=0.0125
```

**Domain & slabs**

- Global domain: `(dimx, dimy, dimz_total)` (no halos).
- Partition into `procs` equal slabs along z, each with local z depth `dz = dimz_total/procs`.
- Each slab allocates local arrays with **4-deep halos** on both z sides → local z extent `dz+8`.
- Owned z range in local coords: `[4 .. 4+dz-1]`.

**Copy-through boundaries:** For any access leaving the global domain (x/y/z faces within radius 4), the output must equal the input.

---

## What you implement

In `student_kernel.cu`:

```cpp
extern "C" void mpi_stencil_pipeline_cudaaware(const float* d_in_full,
                                               float* d_out_full,
                                               int dimx,int dimy,int dimz_total,
                                               int procs);
```

This function must:

1. Partition the domain into `procs` slabs (`dz=dimz_total/procs` – assert divisibility).
2. For each slab `r`:

   * Allocate `d_in[r]`, `d_out[r]` of size `dimx*dimy*(dz+8)`.
   * Allocate send/recv buffers `d_Ls[r], d_Rs[r], d_Lr[r], d_Rr[r]` (each `4*dimx*dimy`).
   * Scatter from `d_in_full` into `d_in[r]` (owned planes at `k_local=4..`) using the provided kernel.
   * Seed `d_out[r] = d_in[r]` (so copy-through faces are already correct where we don't write).
3. **Stage-1 boundary:** call `stencil25_stage1_boundary(d_in[r], d_out[r], ..., z0=r*dz, dimz_total)`.
4. **Pack:** call `halo_pack_boundaries(d_out[r], ..., d_Ls[r], d_Rs[r])`.
5. **CUDA-aware Exchange:** use the provided wrapper
   `mpi_cudaaware_sendrecv_device(sendbuf, sendcount, recvbuf, recvcount)` to exchange with neighbors.

   * Left exchange: send our left boundary → neighbor's right recv; receive neighbor's right boundary into our left recv.
   * Right exchange: symmetric.
   * **Do not copy through host. Use the wrapper (device pointers only).**
6. **Unpack:** `halo_unpack_to_halos(d_out[r], ..., d_Lr[r], d_Rr[r])`.
7. **Stage-2 interior:** `stencil25_stage2_interior(d_in[r], d_out[r], ...)`.
8. **Gather:** write owned output planes back to `d_out_full` via the provided kernel.
9. Free all per-slab allocations.

We give you:

* Stage-1/2 kernels (launchers),
* pack/unpack kernels (launchers),
* scatter/gather kernels,
* and a **CUDA-aware sendrecv wrapper** that performs `cudaMemcpyDeviceToDevice`.

**Tests** validate correctness vs CPU oracle, several sizes & `procs`, and guard-canaries.

### Build & run

```bash
make test_reference
make test_student
```

### Grading focus

* Correct partitioning and z-ownership mapping,
* Correct Stage-1/2 ranges (boundary vs interior),
* Proper use of **CUDA-aware sendrecv wrapper** (no host staging),
* Exact equality to oracle within tolerance; no out-of-bounds writes.