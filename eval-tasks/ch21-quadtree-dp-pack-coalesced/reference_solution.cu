// ch21-quadtree-dp-pack-coalesced / reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __host__ inline bool in_NW(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px < mx) && (py >= my);
}
__device__ __host__ inline bool in_NE(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px >= mx) && (py >= my);
}
__device__ __host__ inline bool in_SW(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px < mx) && (py <  my);
}
__device__ __host__ inline bool in_SE(float px,float py,float minx,float miny,float maxx,float maxy){
  float mx=0.5f*(minx+maxx), my=0.5f*(miny+maxy);
  return (px >= mx) && (py <  my);
}

extern "C" __global__
void pack_quadrants_singleblock(const float* __restrict__ x,
                                const float* __restrict__ y,
                                const int*   __restrict__ idx_in,
                                int*         __restrict__ idx_out,
                                int segBegin, int segCount,
                                float minx, float miny, float maxx, float maxy)
{
  extern __shared__ int sh[];
  int* counts  = sh;        // 4
  int* offsets = sh + 4;    // 4
  int* cursors = sh + 8;    // 4

  if (threadIdx.x < 4){ counts[threadIdx.x]=0; }
  __syncthreads();

  // Count
  for (int t=threadIdx.x; t<segCount; t+=blockDim.x){
    int id = idx_in[segBegin + t];
    float px = x[id], py = y[id];
    int q = in_NW(px,py,minx,miny,maxx,maxy) ? 0 :
            in_NE(px,py,minx,miny,maxx,maxy) ? 1 :
            in_SW(px,py,minx,miny,maxx,maxy) ? 2 : 3;
    atomicAdd(&counts[q], 1);
  }
  __syncthreads();

  if (threadIdx.x==0){
    offsets[0]=0;
    offsets[1]=counts[0];
    offsets[2]=counts[0]+counts[1];
    offsets[3]=counts[0]+counts[1]+counts[2];
    cursors[0]=cursors[1]=cursors[2]=cursors[3]=0;
  }
  __syncthreads();

  // Stable scatter: process in sequential order to preserve stability
  for (int t=0; t<segCount; t++){
    if (threadIdx.x == 0) {
      int id = idx_in[segBegin + t];
      float px = x[id], py = y[id];
      int q = in_NW(px,py,minx,miny,maxx,maxy) ? 0 :
              in_NE(px,py,minx,miny,maxx,maxy) ? 1 :
              in_SW(px,py,minx,miny,maxx,maxy) ? 2 : 3;
      int pos_in_q = cursors[q]++;
      int out = segBegin + offsets[q] + pos_in_q;
      idx_out[out] = id;
    }
  }
}