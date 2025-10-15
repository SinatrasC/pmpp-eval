// ch14-spmv-csr-thread-per-row-single / test_spmv_csr.cu
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cmath>

extern "C" __global__
void spmv_csr_kernel(const int*, const int*, const float*, const float*, float*, int);

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

// Build CSR from a pattern (duplicates allowed -> they remain, CSR sum still works)
struct Csr { int m,n,nnz; std::vector<int> rowPtr, colIdx; std::vector<float> val; };

static Csr gen_csr(int m, int n, int nnz, int pat){
    std::mt19937 rng(1337 + pat*1313 + m*11 + n*17 + nnz*23);
    std::uniform_int_distribution<int> rdist(0, (m? m-1:0));
    std::uniform_int_distribution<int> cdist(0, (n? n-1:0));
    std::uniform_real_distribution<float> fdist(-1.0f,1.0f);

    // First: per-row counts
    std::vector<int> rowCount(m, 0);

    std::vector<int> tmp_row(nnz);
    std::vector<int> tmp_col(nnz);
    std::vector<float> tmp_val(nnz);

    switch (pat % 6) {
    case 0: // random
        for (int k=0;k<nnz;k++){
            int r = rdist(rng), c = cdist(rng);
            tmp_row[k]=r; tmp_col[k]=c; tmp_val[k]=fdist(rng);
            if (m>0) rowCount[r]++;
        }
        break;
    case 1: // diagonal-ish (square)
        for (int k=0;k<nnz;k++){
            int i = k % std::max(1,std::min(m,n));
            tmp_row[k]=i; tmp_col[k]=i; tmp_val[k]=1.f;
            if (m>0) rowCount[i]++;
        }
        break;
    case 2: // hot row at end
        for (int k=0;k<nnz;k++){
            int r = (m? m-1:0), c = cdist(rng);
            tmp_row[k]=r; tmp_col[k]=c; tmp_val[k]=fdist(rng);
            if (m>0) rowCount[r]++;
        }
        break;
    case 3: // empty rows sprinkled
        for (int k=0;k<nnz;k++){
            int r = (rdist(rng)%2==0)? rdist(rng) : std::min(rdist(rng), std::max(0,m-2));
            tmp_row[k]=r; tmp_col[k]=cdist(rng); tmp_val[k]=fdist(rng);
            if (m>0) rowCount[r]++;
        }
        break;
    case 4: // duplicates in same row/col region
        for (int k=0;k<nnz;k++){
            int r = rdist(rng)%std::max(1,m/2+1);
            int c = cdist(rng)%std::max(1,n/2+1);
            tmp_row[k]=r; tmp_col[k]=c; tmp_val[k]=1.f;
            if (m>0) rowCount[r]++;
        }
        break;
    default: // band around diagonal (if square), else random
        if (m==n){
            for (int k=0;k<nnz;k++){
                int i = rdist(rng);
                int j = std::min(std::max(0, i + ((k%3)-1)), n-1);
                tmp_row[k]=i; tmp_col[k]=j; tmp_val[k]=fdist(rng);
                if (m>0) rowCount[i]++;
            }
        } else {
            for (int k=0;k<nnz;k++){
                int r=rdist(rng), c=cdist(rng);
                tmp_row[k]=r; tmp_col[k]=c; tmp_val[k]=fdist(rng);
                if (m>0) rowCount[r]++;
            }
        }
        break;
    }

    Csr A{m,n,nnz};
    A.rowPtr.assign(m+1, 0);
    for (int i=0;i<m;i++) A.rowPtr[i+1] = A.rowPtr[i] + rowCount[i];
    A.colIdx.assign(nnz, 0);
    A.val.assign(nnz, 0.f);

    // write cursor per row
    std::vector<int> wr(m, 0);
    for (int i=0;i<m;i++) wr[i] = A.rowPtr[i];

    for (int k=0;k<nnz;k++){
        int r = (m==0? 0 : tmp_row[k]);
        int pos = wr[std::min(std::max(0,r), std::max(0,m-1))]++;
        if (nnz>0 && pos < (int)A.colIdx.size()){
            A.colIdx[pos] = (n==0? 0 : tmp_col[k]);
            A.val[pos]    = tmp_val[k];
        }
    }
    return A;
}

// CPU oracle for CSR
static void spmv_csr_cpu(const Csr& A, const std::vector<float>& x, std::vector<float>& y){
    y.assign(A.m, 0.f);
    for (int i=0;i<A.m;i++){
        float sum=0.f;
        for (int j=A.rowPtr[i]; j<A.rowPtr[i+1]; ++j){
            sum += A.val[j] * x[A.colIdx[j]];
        }
        y[i]=sum;
    }
}

int main(){
    printf("ch14-spmv-csr-thread-per-row-single tests\n");

    struct Case { int m,n,nnz; const char* name; };
    const Case cases[] = {
        {0,0,0,      "empty"},
        {1,1,0,      "1x1 zero"},
        {1,1,3,      "1x1 duplicates"},
        {4,4,6,      "4x4 small"},
        {64,64,128,  "64x64"},
        {77,41,200,  "rectangular"},
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

          Csr A = gen_csr(cs.m, cs.n, cs.nnz, pat);
          std::vector<float> x(cs.n, 0.f);
          for (int j=0;j<cs.n;j++) x[j] = 0.01f*(float)(j+1);

          std::vector<float> y_ref; spmv_csr_cpu(A, x, y_ref);

          // Guarded buffers
          int *d_row_all=nullptr, *d_col_all=nullptr;
          float *d_val_all=nullptr, *d_x_all=nullptr, *d_y_all=nullptr;

          ck(cudaMalloc(&d_row_all, (A.m+1 + 2*GUARD_I)*sizeof(int)), "malloc rowPtr");
          ck(cudaMalloc(&d_col_all, (A.nnz  + 2*GUARD_I)*sizeof(int)), "malloc colIdx");
          ck(cudaMalloc(&d_val_all, (A.nnz  + 2*GUARD_F)*sizeof(float)), "malloc vals");
          ck(cudaMalloc(&d_x_all,   (A.n    + 2*GUARD_F)*sizeof(float)), "malloc x");
          ck(cudaMalloc(&d_y_all,   (A.m    + 2*GUARD_F)*sizeof(float)), "malloc y");

          std::vector<int>   h_row_guard(A.m+1 + 2*GUARD_I, SENT_I);
          std::vector<int>   h_col_guard(A.nnz  + 2*GUARD_I, SENT_I);
          std::vector<float> h_val_guard(A.nnz  + 2*GUARD_F, SENT_F);
          std::vector<float> h_x_guard(  A.n    + 2*GUARD_F, SENT_F);
          std::vector<float> h_y_guard(  A.m    + 2*GUARD_F, SENT_F);

          if (A.m+1>0) std::copy(A.rowPtr.begin(), A.rowPtr.end(), h_row_guard.begin()+GUARD_I);
          if (A.nnz>0){
              std::copy(A.colIdx.begin(), A.colIdx.end(), h_col_guard.begin()+GUARD_I);
              std::copy(A.val.begin(),    A.val.end(),    h_val_guard.begin()+GUARD_F);
          }
          if (A.n>0) std::copy(x.begin(), x.end(), h_x_guard.begin()+GUARD_F);

          // y interior set to sentinel to ensure kernel overwrites (not accumulates)
          if (A.m>0) std::fill(h_y_guard.begin()+GUARD_F, h_y_guard.begin()+GUARD_F+A.m, -999.0f);

          ck(cudaMemcpy(d_row_all, h_row_guard.data(), h_row_guard.size()*sizeof(int),   cudaMemcpyHostToDevice), "H2D rowPtr");
          ck(cudaMemcpy(d_col_all, h_col_guard.data(), h_col_guard.size()*sizeof(int),   cudaMemcpyHostToDevice), "H2D colIdx");
          ck(cudaMemcpy(d_val_all, h_val_guard.data(), h_val_guard.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D vals");
          ck(cudaMemcpy(d_x_all,   h_x_guard.data(),   h_x_guard.size()*sizeof(float),   cudaMemcpyHostToDevice), "H2D x");
          ck(cudaMemcpy(d_y_all,   h_y_guard.data(),   h_y_guard.size()*sizeof(float),   cudaMemcpyHostToDevice), "H2D y");

          int*   d_row = d_row_all + GUARD_I;
          int*   d_col = d_col_all + GUARD_I;
          float* d_val = d_val_all + GUARD_F;
          float* d_x   = d_x_all   + GUARD_F;
          float* d_y   = d_y_all   + GUARD_F;

          auto cdiv = [](int a,int b){ return (a + b - 1)/b; };
          int grid_x = std::max(1, cdiv(A.m, (int)bdim.x));
          grid_x = std::min(grid_x, 65535);

          spmv_csr_kernel<<<dim3(grid_x), bdim>>>(d_row, d_col, d_val, d_x, d_y, A.m);
          ck(cudaGetLastError(), "launch");
          ck(cudaDeviceSynchronize(), "sync");

          // Download
          ck(cudaMemcpy(h_y_guard.data(), d_y_all, h_y_guard.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H y");
          ck(cudaMemcpy(h_row_guard.data(), d_row_all, h_row_guard.size()*sizeof(int), cudaMemcpyDeviceToHost), "D2H rowPtr");
          ck(cudaMemcpy(h_col_guard.data(), d_col_all, h_col_guard.size()*sizeof(int), cudaMemcpyDeviceToHost), "D2H colIdx");
          ck(cudaMemcpy(h_val_guard.data(), d_val_all, h_val_guard.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H vals");
          ck(cudaMemcpy(h_x_guard.data(),   d_x_all,   h_x_guard.size()*sizeof(float),   cudaMemcpyDeviceToHost), "D2H x");

          bool ok = true;

          // y equals oracle
          std::vector<float> y_got(A.m, 0.f);
          if (A.m>0) std::copy(h_y_guard.begin()+GUARD_F, h_y_guard.begin()+GUARD_F+A.m, y_got.begin());
          ok = ok && vec_close(y_got, y_ref, 1e-6f);

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
          if (A.m+1>0){
              std::vector<int> row_in(A.m+1);
              std::copy(h_row_guard.begin()+GUARD_I, h_row_guard.begin()+GUARD_I+A.m+1, row_in.begin());
              ok = ok && std::equal(row_in.begin(), row_in.end(), A.rowPtr.begin());
          }
          if (A.nnz>0){
              std::vector<int> col_in(A.nnz); std::vector<float> val_in(A.nnz);
              std::copy(h_col_guard.begin()+GUARD_I, h_col_guard.begin()+GUARD_I+A.nnz, col_in.begin());
              std::copy(h_val_guard.begin()+GUARD_F, h_val_guard.begin()+GUARD_F+A.nnz, val_in.begin());
              ok = ok && std::equal(col_in.begin(), col_in.end(), A.colIdx.begin());
              ok = ok && vec_close(val_in, A.val);
          }
          if (A.n>0){
              std::vector<float> x_in(A.n);
              std::copy(h_x_guard.begin()+GUARD_F, h_x_guard.begin()+GUARD_F+A.n, x_in.begin());
              ok = ok && vec_close(x_in, x);
          }

          std::printf("CSR  %-14s pat=%d block=%3u grid=%4d -> %s\n",
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