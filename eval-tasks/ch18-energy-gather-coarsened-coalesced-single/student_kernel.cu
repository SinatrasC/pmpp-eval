// ch18-energy-gather-coarsened-coalesced-single / student_kernel.cu
//
// Implement a coarsened gather kernel that stages results in shared memory and
// flushes them with coalesced writes.  This starter file intentionally leaves the
// body empty so the model must supply the full algorithm.

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
#ifndef COARSEN_FACTOR
#define COARSEN_FACTOR 8
#endif

__constant__ float atoms[CHUNK_SIZE * 4];

extern "C" __global__
void cenergyCoarsenCoalescedKernel(float* __restrict__ energygrid,
                                   dim3 grid,
                                   float gridspacing,
                                   float z,
                                   int atoms_in_chunk,
                                   int /*start_atom_unused*/) {
    int base_i = blockIdx.x * (blockDim.x * COARSEN_FACTOR) + threadIdx.x * COARSEN_FACTOR;
    int j      = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < 0 || j >= (int)grid.y) {
        return;
    }

    int k = int(z / gridspacing);
    if (k < 0 || k >= (int)grid.z) {
        return;
    }

    // TODO: Implement coarsened energy accumulation with shared memory and coalesced writes:
    //
    // IMPORTANT: This kernel is called multiple times (once per atom chunk).
    // You MUST use += to accumulate results, NOT = to overwrite!
    //
    // Step 1: Declare shared memory buffer
    // - Size: blockDim.x * blockDim.y * COARSEN_FACTOR floats
    // - Layout: [threadIdx.y][threadIdx.x * COARSEN_FACTOR + c] for coalesced access
    //
    // Step 2: Accumulate energy for COARSEN_FACTOR grid points
    // - For each c in [0, COARSEN_FACTOR), compute i = base_i + c
    // - For each grid point (i, j, k), loop over atoms_in_chunk atoms (NOT CHUNK_SIZE!)
    // - Compute distance and energy contribution (charge / distance)
    // - Store accumulated energy in shared memory (NOT directly to global memory yet)
    //
    // Step 3: Synchronize threads
    // - __syncthreads() to ensure all threads finished computation
    //
    // Step 4: Coalesced write to global memory with ACCUMULATION
    // - Reorganize shared memory data to enable coalesced writes
    // - Each thread writes COARSEN_FACTOR consecutive values from shared memory
    // - USE += NOT =: energygrid[idx] += value  (kernel called multiple times!)
    // - This ensures warps write contiguous memory locations (coalesced access)
    //
    // Hints:
    // - Shared memory avoids uncoalesced scattered writes
    // - Atoms layout: atoms[a*4+0], atoms[a*4+1], atoms[a*4+2] (x,y,z), atoms[a*4+3] (charge)
    // - Grid coordinates: x = i * gridspacing, y = j * gridspacing, z given
    // - Global index: idx = k * grid.x * grid.y + j * grid.x + i
}
