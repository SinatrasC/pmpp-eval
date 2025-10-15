// ch21-quadtree-dp-build-single / reference_solution.cu
#include <cuda_runtime.h>
#include <stdint.h>

struct Bounds { float minx, miny, maxx, maxy; };
__device__ __host__ inline Bounds make_bounds(float a,float b,float c,float d){ return Bounds{a,b,c,d}; }

struct QuadWork {
  const float* x;
  const float* y;
  const int*   idx;
  int          begin;
  int          count;
  Bounds       b;
  int          depth;
  int          max_depth;
  int          min_points;
  int*         perm;
  int*         leafOffset;
  int*         leafCount;
  int*         leafCounter;
  int*         permCursor;
};

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
  return (px < mx) && (py <  my);
}
__device__ __host__ inline bool in_SE(float px, float py, const Bounds& b){
  float mx=0.5f*(b.minx+b.maxx), my=0.5f*(b.miny+b.maxy);
  return (px >= mx) && (py <  my);
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

__global__ void quadtree_node(struct QuadWork w);

__global__ void quadtree_build_parent(const float* x, const float* y, int n,
                                      Bounds root, int max_depth, int min_points,
                                      int* perm, int* leafOffset, int* leafCount,
                                      int* leafCounter, int* permCursor,
                                      const int* idx_root)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  QuadWork w;
  w.x = x; w.y = y; w.idx = idx_root;
  w.begin = 0; w.count = n;
  w.b = root;
  w.depth = 0; w.max_depth = max_depth; w.min_points = min_points;
  w.perm = perm; w.leafOffset = leafOffset; w.leafCount = leafCount;
  w.leafCounter = leafCounter; w.permCursor = permCursor;

  quadtree_node<<<1,1>>>(w);
}

__global__ void quadtree_node(struct QuadWork w)
{
  if (threadIdx.x != 0) return;

  const int begin = w.begin, count = w.count;
  const Bounds b = w.b;

  if (w.depth >= w.max_depth || count <= w.min_points) {
    int leafId   = atomicAdd(w.leafCounter, 1);
    int outBegin = atomicAdd(w.permCursor, count);
    w.leafOffset[leafId] = outBegin;
    w.leafCount[leafId]  = count;
    for (int i=0;i<count;i++){
      int pidx = w.idx[begin + i];
      w.perm[outBegin + i] = pidx;
    }
    return;
  }

  int cNW=0,cNE=0,cSW=0,cSE=0;
  for (int i=0;i<count;i++){
    int id = w.idx[begin+i];
    float px = w.x[id], py = w.y[id];
    if      (in_NW(px,py,b)) cNW++;
    else if (in_NE(px,py,b)) cNE++;
    else if (in_SW(px,py,b)) cSW++;
    else                     cSE++;
  }
  int sNW=0, sNE=cNW, sSW=cNW+cNE, sSE=cNW+cNE+cSW;

  int* localIdx = (int*)malloc(sizeof(int)*count);
  if (!localIdx){
    int leafId   = atomicAdd(w.leafCounter, 1);
    int outBegin = atomicAdd(w.permCursor, count);
    w.leafOffset[leafId] = outBegin;
    w.leafCount[leafId]  = count;
    for (int i=0;i<count;i++){
      w.perm[outBegin + i] = w.idx[begin + i];
    }
    return;
  }
  int pNW=0,pNE=0,pSW=0,pSE=0;
  for (int i=0;i<count;i++){
    int id = w.idx[begin+i];
    float px = w.x[id], py = w.y[id];
    if      (in_NW(px,py,b)) localIdx[sNW + (pNW++)] = id;
    else if (in_NE(px,py,b)) localIdx[sNE + (pNE++)] = id;
    else if (in_SW(px,py,b)) localIdx[sSW + (pSW++)] = id;
    else                     localIdx[sSE + (pSE++)] = id;
  }

  int off=0;
  if (cNW>0){
    QuadWork c=w;
    c.idx=localIdx; c.begin=0; c.count=cNW; c.b=make_bounds(b.minx, 0.5f*(b.miny+b.maxy), 0.5f*(b.minx+b.maxx), b.maxy);
    c.depth=w.depth+1;
    quadtree_node<<<1,1>>>(c);
  }
  off += cNW;
  if (cNE>0){
    QuadWork c=w;
    c.idx=localIdx; c.begin=off; c.count=cNE; c.b=make_bounds(0.5f*(b.minx+b.maxx), 0.5f*(b.miny+b.maxy), b.maxx, b.maxy);
    c.depth=w.depth+1;
    quadtree_node<<<1,1>>>(c);
  }
  off += cNE;
  if (cSW>0){
    QuadWork c=w;
    c.idx=localIdx; c.begin=off; c.count=cSW; c.b=make_bounds(b.minx, b.miny, 0.5f*(b.minx+b.maxx), 0.5f*(b.miny+b.maxy));
    c.depth=w.depth+1;
    quadtree_node<<<1,1>>>(c);
  }
  off += cSW;
  if (cSE>0){
    QuadWork c=w;
    c.idx=localIdx; c.begin=off; c.count=cSE; c.b=make_bounds(0.5f*(b.minx+b.maxx), b.miny, b.maxx, 0.5f*(b.miny+b.maxy));
    c.depth=w.depth+1;
    quadtree_node<<<1,1>>>(c);
  }

  // Note: Device-side sync not supported in this compiler environment
  // free(localIdx); // Remove to avoid use-after-free race with child kernels
}