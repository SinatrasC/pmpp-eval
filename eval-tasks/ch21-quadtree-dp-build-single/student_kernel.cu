// ch21-quadtree-dp-build-single / student_kernel.cu
#include <cuda_runtime.h>
#include <stdint.h>

// ----------------------------- Data types -----------------------------
struct Bounds { float minx, miny, maxx, maxy; };
__device__ __host__ inline Bounds make_bounds(float a,float b,float c,float d){ return Bounds{a,b,c,d}; }

struct QuadWork {
  const float* x;
  const float* y;
  const int*   idx;         // indices of points for this segment
  int          begin;       // segment begin (relative to idx)
  int          count;       // segment length
  Bounds       b;
  int          depth;
  int          max_depth;
  int          min_points;
  // outputs/globals
  int*         perm;        // output permutation (length n)
  int*         leafOffset;  // length >= n
  int*         leafCount;   // length >= n
  int*         leafCounter; // single int in device memory
  int*         permCursor;  // single int in device memory
};

// Prototypes
__global__ void quadtree_build_parent(const float* x, const float* y, int n,
                                      Bounds root, int max_depth, int min_points,
                                      int* perm, int* leafOffset, int* leafCount,
                                      int* leafCounter, int* permCursor,
                                      const int* idx_root);

__global__ void quadtree_node(QuadWork w);

// ----------------------------- Helpers -----------------------------
__device__ __host__ inline bool in_NW(float px, float py, const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px < mx) && (py >= my);
}
__device__ __host__ inline bool in_NE(float px, float py, const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px >= mx) && (py >= my);
}
__device__ __host__ inline bool in_SW(float px, float py, const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px < mx) && (py < my);
}
__device__ __host__ inline bool in_SE(float px, float py, const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px >= mx) && (py < my);
}

__device__ __host__ inline Bounds child_bounds_NW(const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return make_bounds(b.minx, my, mx, b.maxy);
}
__device__ __host__ inline Bounds child_bounds_NE(const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return make_bounds(mx, my, b.maxx, b.maxy);
}
__device__ __host__ inline Bounds child_bounds_SW(const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return make_bounds(b.minx, b.miny, mx, my);
}
__device__ __host__ inline Bounds child_bounds_SE(const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return make_bounds(mx, b.miny, b.maxx, my);
}

// ----------------------------- Kernels -----------------------------
__global__ void quadtree_build_parent(const float* x, const float* y, int n,
                                      Bounds root, int max_depth, int min_points,
                                      int* perm, int* leafOffset, int* leafCount,
                                      int* leafCounter, int* permCursor,
                                      const int* idx_root)
{
  // TODO: Construct a QuadWork item for the root node and invoke child kernels
  // to recursively partition the point set.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    (void)x; (void)y; (void)n;
    (void)root; (void)max_depth; (void)min_points;
    (void)perm; (void)leafOffset; (void)leafCount;
    (void)leafCounter; (void)permCursor; (void)idx_root;
  }
}

__global__ void quadtree_node(QuadWork w)
{
  // TODO: Recursively process the current node, partitioning points into child
  // quadrants and recording leaf metadata.
  if (threadIdx.x == 0) {
    (void)w;
  }
}
