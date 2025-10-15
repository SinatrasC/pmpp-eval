// ch17-fhd-fission-two-kernels-single/reference_solution.cu
#include <cuda_runtime.h>
#include <cmath>

extern "C" __global__
void compute_mu_kernel(const float* __restrict__ rPhi,
                       const float* __restrict__ iPhi,
                       const float* __restrict__ rD,
                       const float* __restrict__ iD,
                       int M,
                       float* __restrict__ rMu,
                       float* __restrict__ iMu)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float rphi = rPhi[m], iphi = iPhi[m];
    float rd = rD[m], id = iD[m];
    rMu[m] = rphi*rd + iphi*id;
    iMu[m] = rphi*id - iphi*rd;
}

extern "C" __global__
void fhd_accumulate_mu_kernel(const float* __restrict__ rMu,
                              const float* __restrict__ iMu,
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

    const float TWO_PI = 6.2831853071795864769f;
    float xn = x[n], yn = y[n], zn = z[n];
    float r_acc = 0.f, i_acc = 0.f;

    for (int m = 0; m < M; ++m) {
        float ang = TWO_PI * (kx[m]*xn + ky[m]*yn + kz[m]*zn);
        float c = cosf(ang);
        float s = sinf(ang);
        float rmu = rMu[m], imu = iMu[m];
        r_acc += rmu * c - imu * s;
        i_acc += imu * c + rmu * s;
    }
    rFhD[n] += r_acc;
    iFhD[n] += i_acc;
}