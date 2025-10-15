// ch18-energy-gather-single / reference_solution.cu
// Working implementation of Fig. 18.6 gather kernel (one thread per grid cell).

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= (int)grid.x || j >= (int)grid.y) return;

    int k = int(z / gridspacing);
    if (k < 0 || k >= (int)grid.z) return;

    float x  = gridspacing * (float)i;
    float y  = gridspacing * (float)j;

    float energy = 0.0f;
    for (int a = 0; a < atoms_in_chunk; ++a) {
        float ax = atoms[4 * a + 0];
        float ay = atoms[4 * a + 1];
        float az = atoms[4 * a + 2];
        float q  = atoms[4 * a + 3];

        float dx = x - ax;
        float dy = y - ay;
        float dz = z - az;
        float denom = sqrtf(dx*dx + dy*dy + dz*dz);
        energy += q / fmaxf(denom, 1e-12f);
    }

    size_t idx = (size_t)grid.x * grid.y * k + (size_t)grid.x * j + (size_t)i;
    energygrid[idx] += energy;
}