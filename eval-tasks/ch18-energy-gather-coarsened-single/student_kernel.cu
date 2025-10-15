// ch18-energy-gather-coarsened-single / student_kernel.cu
//
// Implement a coarsened gather kernel (Fig. 18.8).  The reference solution
// shows the full algorithm; here we deliberately leave the core computation as
// a TODO so the student model must fill it in.

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 256
#endif
#ifndef COARSEN_FACTOR
#define COARSEN_FACTOR 8
#endif

__constant__ float atoms[CHUNK_SIZE * 4];

extern "C" __global__
void cenergyCoarsenKernel(float* __restrict__ energygrid,
                          dim3 grid,
                          float gridspacing,
                          float z,
                          int /*atoms_in_chunk*/,
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

    // TODO: Implement thread-coarsened energy accumulation:
    // - Each thread processes COARSEN_FACTOR grid points (base_i to base_i + COARSEN_FACTOR - 1)
    // - For each grid point (i, j, k), accumulate energy contributions from all atoms in constant memory
    // - Atoms are stored in constant memory as [x, y, z, charge] in structure-of-arrays layout
    // - Compute distance from each atom to grid point: dx, dy, dz
    // - Energy contribution: charge / sqrt(dx^2 + dy^2 + dz^2)
    // - Add accumulated energy to energygrid[idx] (no atomics needed - each thread owns its points)
    // - Use proper bounds checking for i in [0, grid.x)
    //
    // Hints:
    // - Grid point coordinates: x = i * gridspacing, y = j * gridspacing, z already given
    // - Atom coordinates: atoms[a*4+0], atoms[a*4+1], atoms[a*4+2], atoms[a*4+3] (charge)
    // - 3D array index: idx = k * grid.x * grid.y + j * grid.x + i
}
