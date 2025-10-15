// ch18-energy-scatter-single / student_kernel.cu
//
// Implement Fig. 18.5 (SCATTER): one thread per atom, looping over all (i,j)
// grid points on a fixed z-slice and ATOMICALLY accumulating into energygrid.
//
// CONTRACT
// - Constant memory holds a *chunk* of atoms: __constant__ float atoms[CHUNK_SIZE*4]
//   as (x,y,z,charge) AoS, 4 floats per atom.
// - The test harness uploads atoms chunk-by-chunk via cudaMemcpyToSymbol and then
//   launches the kernel once per chunk, accumulating into the same output slice.
// - Kernel params:
//     energygrid    : pointer to [grid.x * grid.y * grid.z] floats
//     grid          : logical 3D grid dimensions (x,y,z) for indexing
//     gridspacing   : spacing used to compute x = i*gridspacing, y = j*gridspacing
//     z             : world-space z coordinate of the slice (must be multiple of gridspacing)
//     atoms_in_chunk: number of atoms loaded in constant memory for this launch (<= CHUNK_SIZE)
//     start_atom    : global offset of first atom of the chunk (not needed for correct math;
//                     included so your signature matches reference; you may ignore it)
// - Must use atomicAdd on energygrid writes (scatter means threads collide on same cell).
// - Bounds: guard i in [0,grid.x), j in [0,grid.y).
//
// HINTS
// - Compute k = int(z / gridspacing) once per kernel; assume z aligns.
// - Thread id tid covers one atom in [0, atoms_in_chunk).
// - Load (ax,ay,az,q) from constant memory as atoms[4*tid + {0,1,2,3}].
// - For each j row, precompute y and (y-ay), (z-az) and reuse (dy*dy + dz*dz).
// - Use sqrtf for single-precision.
// - Use 1D blocks and grids for atom threads (e.g., blockDim.x = 256).
//
// PERFORMANCE is not graded here—correctness, safety (no OOB), and atomicity are.

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif

__constant__ float atoms[CHUNK_SIZE * 4];

extern "C" __global__
void cenergyScatterKernel(float* __restrict__ energygrid,
                          dim3 grid,
                          float gridspacing,
                          float z,
                          int atoms_in_chunk,
                          int /*start_atom_unused*/) {
    // TODO: Implement scatter kernel (one thread per atom)
    // 1. Get thread ID and bounds check
    // 2. Load atom data from constant memory
    // 3. Compute z-slice index k
    // 4. Loop over all (i,j) grid points in the slice
    // 5. For each grid point, compute distance and contribution
    // 6. Use atomicAdd to accumulate into energygrid
}