# ch21-quadtree-dp-build-single

**Goal:** Build a point quadtree **with dynamic parallelism** (DP) and return a permutation `perm` that packs points by **leaf order**. The leaf traversal order is **NW, NE, SW, SE** (per node). The permutation must be **stable within each quadrant** (points keep their original relative order inside a leaf).

We run the whole build on the device using child launches. For simplicity and determinism, each node uses a **single-block, single-thread** control path to count, partition, and spawn children.

## Inputs
- Device arrays: `x[n]`, `y[n]` (float)
- Root bounds: `{minx, miny, maxx, maxy}`
- `max_depth`, `min_points_per_node`

## Outputs
- `perm[n]`: permutation of `0..n-1` packing by leaves (DP traversal order)
- `leafOffset[L+1]`, `leafCount[L]` (L determined at runtime) — used for verification
  - We also keep a global `leafCounter` (device int) and a global `permCursor` (device int) to assign leaf IDs and contiguous output ranges.

## Contract (high level)
- `quadtree_build_parent<<<1,1>>>` spawns a root node over all points.
- Node kernel:
  - If `depth==max_depth || count<=min_points_per_node`: **reserve a leaf** and **write** `perm` with a **global** `atomicAdd(permCursor, count)`; fill `leafOffset[leafId]=offset`, `leafCount[leafId]=count`.
  - Else: **partition** the segment by quadrants (order **NW, NE, SW, SE**) using **two-pass** (count → exclusive-scan → scatter) into a **device malloc** local buffer for this node; launch up to 4 children with updated bounds and the appropriate subranges of that local buffer; `cudaDeviceSynchronize()` then `free()` the buffer.


## Determinism
- Quadrant coding:
```
midx=(minx+maxx)/2, midy=(miny+maxy)/2
NW: x < midx && y >= midy
NE: x >= midx && y >= midy
SW: x < midx && y <  midy
SE: x >= midx && y <  midy
```
- Stable order inside quadrant: iterate subsegment in order; use running per-quadrant offsets.

## Build
```
make
```

Produces:
- `test_reference` (reference_solution.cu)
- `test_student` (student_kernel.cu)

## Run
```
./test_reference
./test_student
```

Both should print **OK** for all cases.

## Notes
- No Thrust / Python.
- DP linking flags required: `-rdc=true -lcudadevrt`.
- Tests set `cudaLimitMallocHeapSize` so device `malloc/free` is available.
