// ch17-fhd-fission-two-kernels-single/student_kernel.cu
#include <cuda_runtime.h>

// TODO (A): compute_mu_kernel
// Each thread handles one m (if m < M):
//   rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m]
//   iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m]
extern "C" __global__
void compute_mu_kernel(const float* __restrict__ rPhi,
                       const float* __restrict__ iPhi,
                       const float* __restrict__ rD,
                       const float* __restrict__ iD,
                       int M,
                       float* __restrict__ rMu,
                       float* __restrict__ iMu)
{
    // TODO: Implement complex multiplication for each m
    // Hint: int m = blockIdx.x * blockDim.x + threadIdx.x;
}

// TODO (B): fhd_accumulate_mu_kernel
// One thread per n; loop over m using precomputed rMu/iMu.
// Accumulate into rFhD[n], iFhD[n] using Fourier transform formula.
//
// IMPORTANT FORMULAS:
// 1. Angle calculation must include TWO_PI (2π):
//    const float TWO_PI = 6.2831853071795864769f;
//    float ang = TWO_PI * (kx[m]*xn + ky[m]*yn + kz[m]*zn);
//
// 2. Use ACCUMULATION (+=), not assignment (=), because this kernel may be
//    called multiple times:
//    rFhD[n] += r_acc;  // NOT rFhD[n] = r_acc;
//    iFhD[n] += i_acc;  // NOT iFhD[n] = i_acc;
//
// Algorithm:
//   int n = blockIdx.x * blockDim.x + threadIdx.x;
//   if (n < N) {
//       float r_acc = 0.0f, i_acc = 0.0f;
//       for (int m = 0; m < M; ++m) {
//           float ang = TWO_PI * (kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]);
//           float c = cosf(ang), s = sinf(ang);
//           r_acc += rMu[m] * c - iMu[m] * s;
//           i_acc += iMu[m] * c + rMu[m] * s;
//       }
//       rFhD[n] += r_acc;
//       iFhD[n] += i_acc;
//   }
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
    // TODO: Implement accumulation using precomputed rMu, iMu (see formula above)
}