// reference_solution.cu
#include <cuda_runtime.h>
#include <cstdio>

#ifndef IN_TILE_DIM
#define IN_TILE_DIM 8
#endif
#define OUT_TILE_DIM (IN_TILE_DIM-2)

__global__ void stencil3d_shared_student(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6)
{
    if (N <= 0) return;

    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Global coordinates of the tile INCLUDING halo:
    // Block's output tile origin in global indices
    int baseK = blockIdx.x * OUT_TILE_DIM - 1;
    int baseJ = blockIdx.y * OUT_TILE_DIM - 1;
    int baseI = blockIdx.z * OUT_TILE_DIM - 1;

    // Local thread coordinates for loads
    int lk = threadIdx.x;
    int lj = threadIdx.y;
    int li = threadIdx.z;

    int gk = baseK + lk;
    int gj = baseJ + lj;
    int gi = baseI + li;

    auto in_idx = [N](int I,int J,int K){ return (I*N + J)*N + K; };

    // Guarded load (zero if OOB)
    float val = 0.0f;
    if (gi>=0 && gi<N && gj>=0 && gj<N && gk>=0 && gk<N){
        val = in[in_idx(gi,gj,gk)];
    }
    tile[li][lj][lk] = val;
    __syncthreads();

    // Compute only the interior threads of the tile
    if (lk>0 && lk<IN_TILE_DIM-1 &&
        lj>0 && lj<IN_TILE_DIM-1 &&
        li>0 && li<IN_TILE_DIM-1)
    {
        int K = baseK + lk;
        int J = baseJ + lj;
        int I = baseI + li;

        // If the global target is outside the domain: ignore
        if (I<0 || I>=N || J<0 || J>=N || K<0 || K>=N) return;

        // Copy-through boundaries at the domain edge
        bool interior = (I>0 && I<N-1) && (J>0 && J<N-1) && (K>0 && K<N-1);
        if (!interior){
            out[in_idx(I,J,K)] = (I>=0 && I<N && J>=0 && J<N && K>=0 && K<N) ?
                                  tile[li][lj][lk] : 0.0f;
            return;
        }

        float ctr = tile[li  ][lj  ][lk  ];
        float xm  = tile[li  ][lj  ][lk-1];
        float xp  = tile[li  ][lj  ][lk+1];
        float ym  = tile[li  ][lj-1][lk  ];
        float yp  = tile[li  ][lj+1][lk  ];
        float zm  = tile[li-1][lj  ][lk  ];
        float zp  = tile[li+1][lj  ][lk  ];

        out[in_idx(I,J,K)] = c0*ctr + c1*xm + c2*xp + c3*ym + c4*yp + c5*zm + c6*zp;
    }
}