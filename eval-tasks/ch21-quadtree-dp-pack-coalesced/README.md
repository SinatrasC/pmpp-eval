# ch21-quadtree-dp-pack-coalesced

Implement a **coalesced pack** kernel used by DP quadtree nodes: given a segment `[segBegin, segCount)` of an index array and a bounding box, compute per-quadrant counts, build exclusive offsets, and **scatter** the indices to `idx_out[segBegin..segBegin+segCount)` grouped contiguously by **NW, NE, SW, SE**, preserving **stable order** within each quadrant.

We constrain the kernel to a **single block** (tests keep segment sizes modest). No Thrust; correctness first.


## Kernel Contract
```
__global__ void pack_quadrants_singleblock(
  const float* x, const float* y,
  const int*   idx_in,   // full array
  int*         idx_out,  // full array
  int segBegin, int segCount,
  float minx, float miny, float maxx, float maxy);
```

- Computes counts[4] for the segment
- Builds offsets[4] (exclusive scan)
- Writes:
  - `idx_out[segBegin + 0 .. segBegin + counts[NW]-1]           = NW`
  - `idx_out[segBegin + counts[NW] .. segBegin+counts[NW]+counts[NE]-1] = NE`
  - `...` similarly for SW, SE
- Stable within each quadrant.

## Build
```
make
```

## Run
```
./test_reference
./test_student
```

Both should pass.

## Notes
- Single block (e.g., 256 threads). Use shared memory counters + cooperative loops.
- Deterministic quadrant mapping:
  - NW: x<mid && y>=midy
  - NE: x>=mid && y>=midy
  - SW: x<mid && y< midy
  - SE: x>=mid && y< midy
