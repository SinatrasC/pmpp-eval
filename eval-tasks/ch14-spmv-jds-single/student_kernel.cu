// ch14-spmv-jds-single / student_kernel.cu
#include <cuda_runtime.h>

extern "C" __global__
void spmv_jds_kernel(const int* __restrict__ colJds,
                     const float* __restrict__ valJds,
                     const int* __restrict__ permute,
                     const int* __restrict__ jdPtr,
                     const float* __restrict__ x,
                     float* __restrict__ y,
                     int m, int maxJ)
{
    // TODO: Traverse the jagged diagonals, accumulate per permuted row, and
    // write results back to the original ordering via permute[]
    if (colJds && valJds && permute && jdPtr && x && y) {
        (void)m;
        (void)maxJ;
    }
}

extern "C" void spmv_jds(const int* colJds, const float* valJds,
                         const int* permute, const int* jdPtr,
                         const float* x, float* y, int m, int maxJ)
{
    (void)colJds; (void)valJds; (void)permute; (void)jdPtr;
    (void)x; (void)y; (void)m; (void)maxJ;

    // TODO: Configure launch dimensions and invoke spmv_jds_kernel.
}
