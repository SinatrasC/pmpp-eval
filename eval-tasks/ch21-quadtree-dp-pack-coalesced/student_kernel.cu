// ch21-quadtree-dp-pack-coalesced / student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>

// TODO: Implement quadrant classification helpers
// Each function should determine if point (px, py) belongs to the specified quadrant
// of the bounding box [minx, miny, maxx, maxy].
//
// Hints:
// - Compute midpoint: mx = (minx + maxx) / 2, my = (miny + maxy) / 2
// - NW (Northwest): px < mx AND py >= my
// - NE (Northeast): px >= mx AND py >= my
// - SW (Southwest): px < mx AND py < my
// - SE (Southeast): px >= mx AND py < my

extern "C" __global__
void pack_quadrants_singleblock(const float* __restrict__ x,
                                const float* __restrict__ y,
                                const int*   __restrict__ idx_in,
                                int*         __restrict__ idx_out,
                                int segBegin, int segCount,
                                float minx, float miny, float maxx, float maxy)
{
  // TODO: Implement quadrant packing with shared memory count/scan/scatter:
  //
  // Step 1: Count phase
  // - Each thread classifies its assigned points into one of 4 quadrants
  // - Use your quadrant helper functions (in_NW, in_NE, in_SW, in_SE)
  // - Store per-thread counts for each quadrant in shared memory
  //
  // Step 2: Parallel scan (prefix sum)
  // - Compute exclusive scan of counts for each quadrant to get write offsets
  // - This ensures stable ordering (preserves input order within each quadrant)
  // - Use __syncthreads() between phases
  //
  // Step 3: Scatter phase
  // - Each thread writes its assigned points to idx_out at computed offsets
  // - NW quadrant: starts at offset 0
  // - NE quadrant: starts at offset count_NW
  // - SW quadrant: starts at offset count_NW + count_NE
  // - SE quadrant: starts at offset count_NW + count_NE + count_SW
  //
  // Hints:
  // - Single block kernel: use blockDim.x threads
  // - Shared memory arrays for counts and offsets
  // - Points are indexed via idx_in[segBegin + i]
  // - Point coordinates: x[idx], y[idx]
  (void)x; (void)y; (void)idx_in; (void)idx_out;
  (void)segBegin; (void)segCount;
  (void)minx; (void)miny; (void)maxx; (void)maxy;
}
