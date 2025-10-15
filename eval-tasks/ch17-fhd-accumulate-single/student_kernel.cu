// ch17-fhd-accumulate-single/student_kernel.cu
#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__
void fhd_accumulate_kernel(const float* __restrict__ rPhi,
                           const float* __restrict__ iPhi,
                           const float* __restrict__ rD,
                           const float* __restrict__ iD,
                           const float* __restrict__ kx,
                           const float* __restrict__ ky,
                           const float* __restrict__ kz,
                           const float* __restrict__ x,
                           const float* __restrict__ y,
                           const float* __restrict__ z,
                           int M, int N,
                           float* __restrict__ rFhD,
                           float* __restrict__ iFhD)
{
    // TODO:
    // - 1 thread per n (global id = blockIdx.x*blockDim.x + threadIdx.x)
    // - Guard: if (n >= N) return;
    // - Load x[n], y[n], z[n] into registers
    // - Accumulate over m=0..M-1:
    //      rmu = rPhi[m]*rD[m] + iPhi[m]*iD[m]
    //      imu = rPhi[m]*iD[m] - iPhi[m]*rD[m]
    //      ang = 2*pi*(kx[m]*xn + ky[m]*yn + kz[m]*zn)
    //      c = cosf(ang); s = sinf(ang)
    //      r_acc += rmu*c - imu*s
    //      i_acc += imu*c + rmu*s
    // - Finally: rFhD[n] += r_acc; iFhD[n] += i_acc;

    // TODO: Implement the FHD accumulation kernel here
}