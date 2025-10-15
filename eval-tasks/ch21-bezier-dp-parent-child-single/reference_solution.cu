// ch21-bezier-dp-parent-child-single / reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <vector_types.h>
#include <device_functions.h>

struct BezierLine {
  float2 CP[3];
  float2* vertexPos;
  int     nVertices;
};

__device__ float curvature_of(const float2 P0, const float2 P1, const float2 P2) {
  float vx = P2.x - P0.x, vy = P2.y - P0.y;
  float wx = P1.x - P0.x, wy = P1.y - P0.y;
  float area2 = fabsf(vx*wy - vy*wx);
  float base  = sqrtf(vx*vx + vy*vy);
  if (base < 1e-8f) return 0.0f;
  return area2 / base; // proportional to distance of P1 from line P0-P2
}

__global__ void computeBezierLine_child(int lidx, BezierLine* bLines, int nTess) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nTess) return;

  float u = (nTess>1) ? (float)idx / (float)(nTess-1) : 0.f;
  float omu = 1.f - u;
  float B0 = omu * omu;
  float B1 = 2.f * u * omu;
  float B2 = u * u;

  const float2 P0 = bLines[lidx].CP[0];
  const float2 P1 = bLines[lidx].CP[1];
  const float2 P2 = bLines[lidx].CP[2];

  float2 pos;
  pos.x = B0*P0.x + B1*P1.x + B2*P2.x;
  pos.y = B0*P0.y + B1*P1.y + B2*P2.y;

  bLines[lidx].vertexPos[idx] = pos;
}

__global__ void computeBezierLines_parent(BezierLine* bLines, int nLines, int maxTess) {
  int lidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (lidx >= nLines) return;

  float2 P0 = bLines[lidx].CP[0];
  float2 P1 = bLines[lidx].CP[1];
  float2 P2 = bLines[lidx].CP[2];

  float curv = curvature_of(P0, P1, P2);
  int nVerts = (int)(curv * 16.0f) + 4;
  if (nVerts < 4) nVerts = 4;
  if (nVerts > maxTess) nVerts = maxTess;

  bLines[lidx].nVertices = nVerts;

  float2* buf = (float2*)malloc((size_t)nVerts * sizeof(float2));
  bLines[lidx].vertexPos = buf;
  if (!buf) { bLines[lidx].nVertices = 0; return; }

  dim3 block(32);
  dim3 grid((nVerts + block.x - 1) / block.x);
  computeBezierLine_child<<<grid, block>>>(lidx, bLines, nVerts);
  // No device-side sync here.
  // Host test already calls cudaDeviceSynchronize() after the parent launch,
  // and with DP semantics a parent grid doesn't complete until its children do.
}