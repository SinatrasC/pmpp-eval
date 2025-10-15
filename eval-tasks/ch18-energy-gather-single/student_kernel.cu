// ch18-energy-gather-single / student_kernel.cu
//
// Implement Fig. 18.6 (GATHER): one thread per grid point (on a fixed z-slice).
// Each thread loops over all atoms in the current constant-memory chunk and
// accumulates a private sum, then writes exactly once to energygrid (+=).
//
// CONTRACT
// - Constant memory holds a *chunk* of atoms: __constant__ float atoms[CHUNK_SIZE*4].
// - Kernel params:
//     energygrid    : pointer to [grid.x * grid.y * grid.z] floats
//     grid          : logical 3D grid dimensions (x,y,z) for indexing
//     gridspacing   : spacing for x=i*h, y=j*h
//     z             : world-space z of the slice
//     atoms_in_chunk: number of atoms currently loaded (<= CHUNK_SIZE)
//     start_atom    : global offset of first atom in the chunk (not needed here)
// - NO atomics: each thread owns its output cell and does `energygrid[idx] += local_sum`.
// - 2D launch is expected (e.g., block=(16,16)).
//
// HINTS
// - Compute (dy*dy + dz*dz) per row and reuse where reasonable.
// - Use sqrtf and a small denominator clamp to avoid division by zero.

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif

__constant__ float atoms[CHUNK_SIZE * 4];

extern "C" __global__
void cenergyGatherKernel(float* __restrict__ energygrid,
                         dim3 grid,
                         float gridspacing,
                         float z,
                         int atoms_in_chunk,
                         int /*start_atom_unused*/) {
    // TODO: Implement gather kernel (one thread per grid cell)
    // 1. Get 2D thread indices (i,j) for grid position
    // 2. Bounds check against grid dimensions (grid.x, grid.y)
    // 3. Compute z-slice index and validate:
    //    int k = int(z / gridspacing);
    //    if (k < 0 || k >= (int)grid.z) return;
    // 4. Compute world-space coordinates (x = i*gridspacing, y = j*gridspacing)
    // 5. Loop over all atoms in the current chunk
    // 6. For each atom, compute distance and contribution
    // 7. Compute 3D linearized index into energygrid:
    //    size_t idx = (size_t)grid.x * grid.y * k + (size_t)grid.x * j + (size_t)i;
    // 8. Accumulate private sum, then write once: energygrid[idx] += sum
}