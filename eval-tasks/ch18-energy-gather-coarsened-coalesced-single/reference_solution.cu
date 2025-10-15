// ch18-energy-gather-coarsened-coalesced-single / reference_solution.cu

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
    // Shared memory buffer for coalesced writes
    __shared__ float sbuf[2][128 * COARSEN_FACTOR];  // [blockDim.y][blockDim.x * COARSEN_FACTOR]

    int base_i = blockIdx.x * (blockDim.x * COARSEN_FACTOR) + threadIdx.x * COARSEN_FACTOR;
    int j      = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < 0 || j >= (int)grid.y) return;

    int k = int(z / gridspacing);
    if (k < 0 || k >= (int)grid.z) return;

    float energies[COARSEN_FACTOR];
    #pragma unroll
    for (int c = 0; c < COARSEN_FACTOR; ++c) energies[c] = 0.0f;

    for (int a = 0; a < atoms_in_chunk; ++a) {
        float ax = atoms[4*a + 0];
        float ay = atoms[4*a + 1];
        float az = atoms[4*a + 2];
        float q  = atoms[4*a + 3];

        float y  = gridspacing * (float)j;
        float dy = y - ay;
        float dz = z - az;
        float dyz2 = dy*dy + dz*dz;

        #pragma unroll
        for (int c = 0; c < COARSEN_FACTOR; ++c) {
            int i = base_i + c;
            if (i >= 0 && i < (int)grid.x) {
                float x  = gridspacing * (float)i;
                float dx = x - ax;
                float denom = sqrtf(dx*dx + dyz2);
                energies[c] += q / fmaxf(denom, 1e-12f);
            }
        }
    }

    // Write to shared memory buffer
    #pragma unroll
    for (int c = 0; c < COARSEN_FACTOR; ++c) {
        int i = base_i + c;
        if (i >= 0 && i < (int)grid.x) {
            sbuf[threadIdx.y][threadIdx.x * COARSEN_FACTOR + c] = energies[c];
        } else {
            sbuf[threadIdx.y][threadIdx.x * COARSEN_FACTOR + c] = 0.0f;
        }
    }

    __syncthreads();

    // Coalesced flush to global memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int elements_per_block = blockDim.x * blockDim.y * COARSEN_FACTOR;

    for (int idx = tid; idx < elements_per_block; idx += total_threads) {
        int local_y = idx / (blockDim.x * COARSEN_FACTOR);
        int local_x = idx % (blockDim.x * COARSEN_FACTOR);

        int global_j = blockIdx.y * blockDim.y + local_y;
        int global_i = blockIdx.x * (blockDim.x * COARSEN_FACTOR) + local_x;

        if (global_i >= 0 && global_i < (int)grid.x && global_j >= 0 && global_j < (int)grid.y) {
            size_t global_idx = (size_t)grid.x * grid.y * k + (size_t)grid.x * global_j + (size_t)global_i;
            energygrid[global_idx] += sbuf[local_y][local_x];
        }
    }
}