// ch21-bezier-dp-parent-child-single / student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>

// Use CUDA builtin float2
#include <vector_types.h>

struct BezierLine {
  float2 CP[3];        // P0, P1, P2
  float2* vertexPos;   // device buffer (allocated in parent via device malloc)
  int     nVertices;   // chosen per line in parent
};

// --- You implement these -----------------------------------------------------

// Geometric curvature proxy: distance from P1 to line P0-P2 (normalized by |P2-P0|)
// Return non-negative curvature (0 for degenerate segment).
__device__ float curvature_of(const float2 P0, const float2 P1, const float2 P2) {
  // TODO: implement robust point-to-segment distance proxy.
  // Hints:
  //   v = P2 - P0
  //   w = P1 - P0
  //   area2 = |v.x*w.y - v.y*w.x|     (2x triangle area)
  //   base  = sqrt(v.x*v.x + v.y*v.y)
  //   curvature ~ area2 / max(base, 1e-8)
  return 0.0f; // TODO
}

// Child kernel: compute tessellated positions for one line lidx.
__global__ void computeBezierLine_child(int lidx, BezierLine* bLines, int nTess) {
  // TODO:
  //  - idx = blockIdx.x*blockDim.x + threadIdx.x
  //  - if idx >= nTess: return
  //  - u = idx / (nTess-1)   (float)
  //  - B0=(1-u)^2, B1=2u(1-u), B2=u^2
  //  - position = B0*P0 + B1*P1 + B2*P2
  //  - write to bLines[lidx].vertexPos[idx]
}

// Parent kernel: choose tessellation density, allocate vertex buffers, launch child.
__global__ void computeBezierLines_parent(BezierLine* bLines, int nLines, int maxTess) {
  // TODO:
  //  - lidx = blockIdx.x*blockDim.x + threadIdx.x; if (lidx>=nLines) return
  //  - compute curvature_of(...)
  //  - nVerts = clamp( int(curv*16.f)+4, 4, maxTess );
  //  - bLines[lidx].nVertices = nVerts;
  //  - bLines[lidx].vertexPos = (float2*)malloc(nVerts * sizeof(float2));
  //      * if malloc returns nullptr, set nVertices=0 and return.
  //  - launch child: <<< (nVerts+31)/32, 32 >>> computeBezierLine_child(lidx, bLines, nVerts);
}