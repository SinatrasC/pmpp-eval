#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

// Both student and reference must provide this:
extern "C" void launch_tiled_matmul(const float* M_h, const float* N_h, float* P_h,
                                    int m, int n, int o);

// --------------------- CPU ORACLE ---------------------
static void cpu_matmul(const float* M, const float* N, float* P, int m, int n, int o){
  for(int i=0;i<m;i++){
    for(int j=0;j<o;j++){
      float s=0.f;
      for(int k=0;k<n;k++) s += M[i*n + k]*N[k*o + j];
      P[i*o + j] = s;
    }
  }
}

// ---------------------- HELPERS -----------------------
static void fill_pattern(std::vector<float>& v, int stride){
  // Adversarial, deterministic patterns
  for(size_t i=0;i<v.size();++i){
    int x = static_cast<int>(i);
    switch(stride % 4){
      case 0: v[i] = float((x%13)-6) * 0.1f; break;
      case 1: v[i] = float(((x*7)%17)-8) * 0.05f; break;
      case 2: v[i] = float(((x*9973)%101)-50) * 0.02f; break;
      default:v[i] = float(((x*29)%31)-15) * 0.03f; break;
    }
  }
}

static bool allclose(const std::vector<float>& a, const std::vector<float>& b, float eps=1e-4f){
  if(a.size()!=b.size()) return false;
  for(size_t i=0;i<a.size();++i){
    float da = a[i], db = b[i];
    float diff = std::fabs(da-db);
    if(diff > eps*(1.f + std::fabs(da))) return false;
  }
  return true;
}

static bool equal_arrays(const std::vector<float>& a, const std::vector<float>& b){
  if(a.size()!=b.size()) return false;
  for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]) return false;
  return true;
}

// ---------------------- TESTS -------------------------
static bool run_case(int m,int n,int o,int pat_id){
  std::string name = "m="+std::to_string(m)+" n="+std::to_string(n)+" o="+std::to_string(o);
  std::cout << "Test " << name << " ... ";

  // host data
  std::vector<float> M(m*n), N(n*o), P_gpu(m*o, 0.f), P_cpu(m*o, 0.f);
  fill_pattern(M, pat_id);
  fill_pattern(N, pat_id+1);

  // keep pristine copies for immutability checks
  auto M_copy = M;
  auto N_copy = N;

  // compute CPU reference
  if(m>0 && n>0 && o>0) cpu_matmul(M.data(), N.data(), P_cpu.data(), m, n, o);

  // run student/ref implementation
  launch_tiled_matmul(M.data(), N.data(), P_gpu.data(), m, n, o);

  // Verify inputs unchanged
  if(!equal_arrays(M, M_copy)){ std::cout << "FAIL (M modified)\n"; return false; }
  if(!equal_arrays(N, N_copy)){ std::cout << "FAIL (N modified)\n"; return false; }

  // Compare results
  if(!allclose(P_gpu, P_cpu)){
    std::cout << "FAIL (mismatch)\n";
    return false;
  }
  std::cout << "OK\n";
  return true;
}

int main(){
  // Enhanced coverage: more edge cases and adversarial TILE alignment
  const int sizes[][3] = {
    {0,0,0},
    {1,1,1},
    {7,5,9},
    {15,15,15},    // TILE-1 (should expose bounds issues)
    {16,16,16},    // Perfect TILE alignment
    {17,16,33},
    {31,31,31},    // 2*TILE-1
    {32,32,32},    // 2*TILE
    {33,17,64},
    {48,48,48},    // 3*TILE
    {64,32,48},
    {96,64,37},
    {127,128,129}, // Large near power-of-2
    {128,64,96},
    {123,111,117},
    {255,256,257}  // Very large edge case
  };

  bool all_ok = true;
  const int num_tests = sizeof(sizes) / sizeof(sizes[0]);
  for(int t=0;t<num_tests;++t){
    int m = sizes[t][0], n = sizes[t][1], o = sizes[t][2];
    all_ok &= run_case(m,n,o, t);
  }

  std::cout << (all_ok ? "All tests passed.\n" : "Some tests FAILED.\n");
  return all_ok ? 0 : 1;
}