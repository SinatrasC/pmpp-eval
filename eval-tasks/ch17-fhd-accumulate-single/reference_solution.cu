// ch17-fhd-accumulate-single/reference_solution.cu
#include <cuda_runtime.h>
#include <cmath>

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
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float xn = x[n], yn = y[n], zn = z[n];
    float r_acc = 0.f, i_acc = 0.f;
    const float TWO_PI = 6.2831853071795864769f;

    for (int m = 0; m < M; ++m) {
        float rmu = rPhi[m] * rD[m] + iPhi[m] * iD[m];
        float imu = rPhi[m] * iD[m] - iPhi[m] * rD[m];
        float ang = TWO_PI * (kx[m]*xn + ky[m]*yn + kz[m]*zn);
        float c = cosf(ang);
        float s = sinf(ang);
        r_acc += rmu * c - imu * s;
        i_acc += imu * c + rmu * s;
    }

    rFhD[n] += r_acc;
    iFhD[n] += i_acc;
}