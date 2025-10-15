// ch14-spmv-coo-single / test_spmv_coo.cu
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" __global__
void spmv_coo_kernel(const int*, const int*, const float*, const float*, float*, int);

static void ck(cudaError_t e, const char* m){
    if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

static inline bool feq(float a, float b, float eps=1e-6f){
    float d = std::fabs(a-b);
    if (d <= eps) return true;
    float ma = std::max(1.f, std::max(std::fabs(a), std::fabs(b)));
    return d <= eps*ma;
}

static bool vec_close(const std::vector<float>& a, const std::vector<float>& b, float eps=1e-6f){
    if (a.size()!=b.size()) return false;
    for (size_t i=0;i<a.size();++i) if (!feq(a[i],b[i],eps)) return false;
    return true;
}

// CPU oracle for COO (duplicates allowed)
static void spmv_coo_cpu(const std::vector<int>& row,
                         const std::vector<int>& col,
                         const std::vector<float>& val,
                         const std::vector<float>& x,
                         std::vector<float>& y,
                         int m)
{
    y.assign(m, 0.f);
    for (size_t k=0;k<val.size();++k){
        y[row[k]] += val[k] * x[col[k]];
    }
}

// Generate adversarial COO matrices
struct Coo { int m,n,nnz; std::vector<int> row, col; std::vector<float> val; };
static Coo gen_coo(int m, int n, int nnz, int pat){
    std::mt19937 rng(42 + pat*9973 + m*3 + n*5 + nnz*7);
    std::uniform_int_distribution<int> rdist(0, m? m-1:0);
    std::uniform_int_distribution<int> cdist(0, n? n-1:0);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    Coo A{m,n,nnz};
    A.row.resize(nnz);
    A.col.resize(nnz);
    A.val.resize(nnz);

    switch (pat % 6) {
    case 0: // random, with duplicates possible
        for (int k=0;k<nnz;k++){ A.row[k]=rdist(rng); A.col[k]=cdist(rng); A.val[k]=fdist(rng); }
        break;
    case 1: // identity-like (square only)
        for (int k=0;k<nnz;k++){ int i=k%(std::min(m,n)); A.row[k]=i; A.col[k]=i; A.val[k]=1.f; }
        break;
    case 2: // hot row (row 0)
        for (int k=0;k<nnz;k++){ A.row[k]=0; A.col[k]=cdist(rng); A.val[k]=fdist(rng); }
        break;
    case 3: // banded around diagonal (if square), else random
        if (m==n){
            for (int k=0;k<nnz;k++){
                int i = rdist(rng);
                int j = std::min(std::max(0, i + ((k%3)-1)), n-1);
                A.row[k]=i; A.col[k]=j; A.val[k]=fdist(rng);
            }
        } else {
            for (int k=0;k<nnz;k++){ A.row[k]=rdist(rng); A.col[k]=cdist(rng); A.val[k]=fdist(rng); }
        }
        break;
    case 4: // duplicates (same (i,j) repeats)
        for (int k=0;k<nnz;k++){
            int i = rdist(rng)%std::max(1,m/2+1);
            int j = cdist(rng)%std::max(1,n/2+1);
            A.row[k]=i; A.col[k]=j; A.val[k]=1.0f;
        }
        break;
    default: // sparse corners
        for (int k=0;k<nnz;k++){
            int i = (k%2==0)? 0 : (m? m-1:0);
            int j = (k%3==0)? 0 : (n? n-1:0);
            A.row[k]=i; A.col[k]=j; A.val[k]=fdist(rng);
        }
        break;
    }
    // keep indices in range if m or n == 0
    if (m==0 || n==0) {
        for (int k=0;k<nnz;k++){ A.row[k]=0; A.col[k]=0; A.val[k]=0.f; }
    }
    return A;
}

int main(){
    printf("ch14-spmv-coo-single tests\n");

    struct Case { int m,n,nnz; const char* name; };
    const Case cases[] = {
        {0,0,0,    "empty"},
        {1,1,0,    "1x1 zero nnz"},
        {1,1,3,    "1x1 duplicates"},
        {4,4,6,    "4x4 small"},
        {64,64,64, "64x64 nnz=64"},
        {77,41,200,"rectangular"},
        {256,256,2048,"dense-ish"},
        {513,257,1024,"prime dims"},
    };

    const dim3 blocks[] = { dim3(128), dim3(256), dim3(512) };

    const size_t GUARD_I = 1024;
    const int    SENT_I  = 0x7f7f7f7f;
    const size_t GUARD_F = 1024;
    const float  SENT_F  = 1337.0f;

    int total=0, pass=0;

    for (const auto& cs : cases){
      for (int pat=0; pat<6; ++pat){
        for (auto bdim : blocks){
          ++total;

          // Generate matrix + vector x
          Coo A = gen_coo(cs.m, cs.n, cs.nnz, pat);
          std::vector<float> x(cs.n, 0.f);
          for (int j=0;j<cs.n;j++) x[j] = 0.01f*(float)(j+1);

          // CPU oracle
          std::vector<float> y_ref(cs.m, 0.f);
          spmv_coo_cpu(A.row, A.col, A.val, x, y_ref, cs.m);

          // Guarded device buffers
          int *d_row_all=nullptr, *d_col_all=nullptr;
          float *d_val_all=nullptr, *d_x_all=nullptr, *d_y_all=nullptr;

          ck(cudaMalloc(&d_row_all, (A.nnz + 2*GUARD_I)*sizeof(int)), "malloc row");
          ck(cudaMalloc(&d_col_all, (A.nnz + 2*GUARD_I)*sizeof(int)), "malloc col");
          ck(cudaMalloc(&d_val_all, (A.nnz + 2*GUARD_F)*sizeof(float)), "malloc val");
          ck(cudaMalloc(&d_x_all,   (cs.n  + 2*GUARD_F)*sizeof(float)), "malloc x");
          ck(cudaMalloc(&d_y_all,   (cs.m  + 2*GUARD_F)*sizeof(float)), "malloc y");

          std::vector<int>   h_row_guard(A.nnz + 2*GUARD_I, SENT_I);
          std::vector<int>   h_col_guard(A.nnz + 2*GUARD_I, SENT_I);
          std::vector<float> h_val_guard(A.nnz + 2*GUARD_F, SENT_F);
          std::vector<float> h_x_guard(  cs.n  + 2*GUARD_F, SENT_F);
          std::vector<float> h_y_guard(  cs.m  + 2*GUARD_F, SENT_F);

          if (A.nnz>0){
              std::copy(A.row.begin(), A.row.end(), h_row_guard.begin()+GUARD_I);
              std::copy(A.col.begin(), A.col.end(), h_col_guard.begin()+GUARD_I);
              std::copy(A.val.begin(), A.val.end(), h_val_guard.begin()+GUARD_F);
          }
          if (cs.n>0) std::copy(x.begin(), x.end(), h_x_guard.begin()+GUARD_F);

          // Zero interior of y (atomic accumulation target)
          if (cs.m>0) std::fill(h_y_guard.begin()+GUARD_F, h_y_guard.begin()+GUARD_F+cs.m, 0.f);

          ck(cudaMemcpy(d_row_all, h_row_guard.data(), h_row_guard.size()*sizeof(int),   cudaMemcpyHostToDevice), "H2D row");
          ck(cudaMemcpy(d_col_all, h_col_guard.data(), h_col_guard.size()*sizeof(int),   cudaMemcpyHostToDevice), "H2D col");
          ck(cudaMemcpy(d_val_all, h_val_guard.data(), h_val_guard.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D val");
          ck(cudaMemcpy(d_x_all,   h_x_guard.data(),   h_x_guard.size()*sizeof(float),   cudaMemcpyHostToDevice), "H2D x");
          ck(cudaMemcpy(d_y_all,   h_y_guard.data(),   h_y_guard.size()*sizeof(float),   cudaMemcpyHostToDevice), "H2D y");

          int*   d_row = d_row_all + GUARD_I;
          int*   d_col = d_col_all + GUARD_I;
          float* d_val = d_val_all + GUARD_F;
          float* d_x   = d_x_all   + GUARD_F;
          float* d_y   = d_y_all   + GUARD_F;

          // Launch
          auto cdiv = [](int a,int b){ return (a + b - 1)/b; };
          int grid_x = std::max(1, cdiv(A.nnz, (int)bdim.x));
          grid_x = std::min(grid_x, 65535);
          spmv_coo_kernel<<<dim3(grid_x), bdim>>>(d_row, d_col, d_val, d_x, d_y, A.nnz);
          ck(cudaGetLastError(), "launch");
          ck(cudaDeviceSynchronize(), "sync");

          // Download
          ck(cudaMemcpy(h_y_guard.data(), d_y_all, h_y_guard.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H y");
          ck(cudaMemcpy(h_row_guard.data(), d_row_all, h_row_guard.size()*sizeof(int), cudaMemcpyDeviceToHost), "D2H row");
          ck(cudaMemcpy(h_col_guard.data(), d_col_all, h_col_guard.size()*sizeof(int), cudaMemcpyDeviceToHost), "D2H col");
          ck(cudaMemcpy(h_val_guard.data(), d_val_all, h_val_guard.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H val");
          ck(cudaMemcpy(h_x_guard.data(),   d_x_all,   h_x_guard.size()*sizeof(float),   cudaMemcpyDeviceToHost), "D2H x");

          // Validate
          bool ok = true;

          // y equals oracle
          std::vector<float> y_got(cs.m, 0.f);
          if (cs.m>0) std::copy(h_y_guard.begin()+GUARD_F, h_y_guard.begin()+GUARD_F+cs.m, y_got.begin());
          ok = ok && vec_close(y_got, y_ref, 5e-5f);  // Higher tolerance for atomic operations

          // Guards unchanged
          auto guard_ok_f = [&](const std::vector<float>& g){
              for (size_t i=0;i<GUARD_F;i++) if (g[i]!=SENT_F || g[g.size()-1-i]!=SENT_F) return false;
              return true;
          };
          auto guard_ok_i = [&](const std::vector<int>& g){
              for (size_t i=0;i<GUARD_I;i++) if (g[i]!=SENT_I || g[g.size()-1-i]!=SENT_I) return false;
              return true;
          };
          ok = ok && guard_ok_f(h_y_guard);
          ok = ok && guard_ok_f(h_x_guard) && guard_ok_f(h_val_guard);
          ok = ok && guard_ok_i(h_row_guard) && guard_ok_i(h_col_guard);

          // Inputs unchanged (interior)
          if (A.nnz>0){
              std::vector<int> row_in(A.nnz), col_in(A.nnz);
              std::vector<float> val_in(A.nnz);
              std::copy(h_row_guard.begin()+GUARD_I, h_row_guard.begin()+GUARD_I+A.nnz, row_in.begin());
              std::copy(h_col_guard.begin()+GUARD_I, h_col_guard.begin()+GUARD_I+A.nnz, col_in.begin());
              std::copy(h_val_guard.begin()+GUARD_F, h_val_guard.begin()+GUARD_F+A.nnz, val_in.begin());
              ok = ok && std::equal(row_in.begin(), row_in.end(), A.row.begin());
              ok = ok && std::equal(col_in.begin(), col_in.end(), A.col.begin());
              ok = ok && vec_close(val_in, A.val);
          }
          if (cs.n>0){
              std::vector<float> x_in(cs.n);
              std::copy(h_x_guard.begin()+GUARD_F, h_x_guard.begin()+GUARD_F+cs.n, x_in.begin());
              ok = ok && vec_close(x_in, x);
          }

          std::printf("COO  %-14s pat=%d block=%3u grid=%4d -> %s\n",
                      cs.name, pat, bdim.x, grid_x, ok?"OK":"FAIL");
          if (ok) ++pass;

          cudaFree(d_row_all); cudaFree(d_col_all);
          cudaFree(d_val_all); cudaFree(d_x_all); cudaFree(d_y_all);
        }
      }
    }

    std::printf("Summary: %d / %d passed\n", pass, total);
    return (pass==total)?0:1;
}