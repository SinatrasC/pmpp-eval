// ch18-energy-scatter-single / reference_solution.cu
// Working implementation of Fig. 18.5 scatter kernel (one thread per atom).

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= atoms_in_chunk) return;

    float ax = atoms[4 * tid + 0];
    float ay = atoms[4 * tid + 1];
    float az = atoms[4 * tid + 2];
    float q  = atoms[4 * tid + 3];

    int k = int(z / gridspacing);
    if (k < 0 || k >= (int)grid.z) return;

    for (int j = 0; j < (int)grid.y; ++j) {
        float y  = gridspacing * (float)j;
        float dy = y - ay;
        float dz = z - az;
        float dyz2 = dy * dy + dz * dz;

        for (int i = 0; i < (int)grid.x; ++i) {
            float x  = gridspacing * (float)i;
            float dx = x - ax;

            float denom   = sqrtf(dx * dx + dyz2);
            float contrib = q / fmaxf(denom, 1e-12f);

            size_t idx = (size_t)grid.x * grid.y * k + (size_t)grid.x * j + (size_t)i;
            atomicAdd(&energygrid[idx], contrib);
        }
    }
}