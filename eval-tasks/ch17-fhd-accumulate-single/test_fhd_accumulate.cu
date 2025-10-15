// ch17-fhd-accumulate-single/test_fhd_accumulate.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

extern "C" __global__
void fhd_accumulate_kernel(const float*, const float*, const float*, const float*,
                           const float*, const float*, const float*,
                           const float*, const float*, const float*,
                           int M, int N, float*, float*);

static void ck(cudaError_t e,const char* m){ if(e){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

static void cpu_fhd_accumulate(const std::vector<float>& rPhi,
                               const std::vector<float>& iPhi,
                               const std::vector<float>& rD,
                               const std::vector<float>& iD,
                               const std::vector<float>& kx,
                               const std::vector<float>& ky,
                               const std::vector<float>& kz,
                               const std::vector<float>& x,
                               const std::vector<float>& y,
                               const std::vector<float>& z,
                               int M, int N,
                               std::vector<float>& rFhD,
                               std::vector<float>& iFhD)
{
    const float TWO_PI = 6.2831853071795864769f;
    for (int n = 0; n < N; ++n) {
        float rn = rFhD[n], in = iFhD[n];
        float xn = x[n], yn = y[n], zn = z[n];
        for (int m = 0; m < M; ++m) {
            float rmu = rPhi[m]*rD[m] + iPhi[m]*iD[m];
            float imu = rPhi[m]*iD[m] - iPhi[m]*rD[m];
            float ang = TWO_PI * (kx[m]*xn + ky[m]*yn + kz[m]*zn);
            float c = std::cos(ang), s = std::sin(ang);
            rn += rmu*c - imu*s;
            in += imu*c + rmu*s;
        }
        rFhD[n] = rn; iFhD[n] = in;
    }
}

static bool almost_equal(const std::vector<float>& a,
                         const std::vector<float>& b,
                         float eps=1e-4f)
{
    if (a.size()!=b.size()) return false;
    for (size_t i=0;i<a.size();++i){
        float da = std::fabs(a[i]-b[i]);
        float ma = std::max(1.f, std::max(std::fabs(a[i]), std::fabs(b[i])));
        if (!(da <= eps || da <= eps*ma)) return false;
    }
    return true;
}

int main(){
    printf("ch17-fhd-accumulate-single tests\n");
    struct Case { int M,N; const char* name; };
    const Case cases[] = {
        {0,0,"M=0 N=0"}, {0,5,"M=0 N=5"}, {7,0,"M=7 N=0"},
        {3,4,"M=3 N=4"}, {17,8,"M=17 N=8"}, {64,63,"M=64 N=63"},
        {128,129,"M=128 N=129"}
    };
    const dim3 blocks[] = { dim3(128), dim3(256) };
    const size_t GUARD = 1024;
    const float SENT = 1337.f;

    int total=0, pass=0;

    for (auto cs : cases) {
        for (auto bdim : blocks) {
            ++total;
            const int M=cs.M, N=cs.N;
            std::vector<float> rPhi(M), iPhi(M), rD(M), iD(M), kx(M), ky(M), kz(M);
            std::vector<float> x(N), y(N), z(N);
            for (int m=0;m<M;++m){
                rPhi[m]=0.1f+0.01f*m; iPhi[m]=-0.2f+0.02f*m;
                rD[m]=0.15f-0.003f*m; iD[m]=0.05f+0.004f*m;
                kx[m]=0.001f*(1+m%7); ky[m]=0.002f*(1+m%5); kz[m]=0.0015f*(1+m%3);
            }
            for (int n=0;n<N;++n){
                x[n]=0.01f*n; y[n]=0.02f*n; z[n]=0.015f*n;
            }

            // CPU oracle
            std::vector<float> rRef(N,0.f), iRef(N,0.f);
            cpu_fhd_accumulate(rPhi,iPhi,rD,iD,kx,ky,kz,x,y,z,M,N,rRef,iRef);

            // Guarded device buffers
            auto alloc_guard = [&](size_t elems){
                float* p=nullptr; ck(cudaMalloc(&p, (elems+2*GUARD)*sizeof(float)), "malloc");
                return p;
            };

            // Upload helpers
            auto h2d_guard = [&](float* d, const std::vector<float>& h){
                std::vector<float> g(h.size()+2*GUARD, SENT);
                if (!h.empty()) std::copy(h.begin(),h.end(),g.begin()+GUARD);
                ck(cudaMemcpy(d, g.data(), g.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D");
            };

            float *d_rPhi=alloc_guard(M), *d_iPhi=alloc_guard(M), *d_rD=alloc_guard(M), *d_iD=alloc_guard(M);
            float *d_kx=alloc_guard(M), *d_ky=alloc_guard(M), *d_kz=alloc_guard(M);
            float *d_x=alloc_guard(N), *d_y=alloc_guard(N), *d_z=alloc_guard(N);
            float *d_rF=alloc_guard(N), *d_iF=alloc_guard(N);

            h2d_guard(d_rPhi, rPhi); h2d_guard(d_iPhi, iPhi);
            h2d_guard(d_rD, rD);     h2d_guard(d_iD, iD);
            h2d_guard(d_kx, kx);     h2d_guard(d_ky, ky); h2d_guard(d_kz, kz);
            h2d_guard(d_x, x);       h2d_guard(d_y, y);   h2d_guard(d_z, z);

            std::vector<float> zeroN(N,0.f);
            h2d_guard(d_rF, zeroN);  h2d_guard(d_iF, zeroN);

            float *rPhi_i = d_rPhi+GUARD, *iPhi_i=d_iPhi+GUARD, *rD_i=d_rD+GUARD, *iD_i=d_iD+GUARD;
            float *kx_i=d_kx+GUARD, *ky_i=d_ky+GUARD, *kz_i=d_kz+GUARD;
            float *x_i=d_x+GUARD, *y_i=d_y+GUARD, *z_i=d_z+GUARD;
            float *rF_i=d_rF+GUARD, *iF_i=d_iF+GUARD;

            auto cdiv = [](int a,int b){ return (a+b-1)/b; };

            if (N > 0) {
                dim3 grid( cdiv(N, (int)bdim.x) );
                fhd_accumulate_kernel<<<grid, bdim>>>(rPhi_i,iPhi_i,rD_i,iD_i,
                                                      kx_i,ky_i,kz_i, x_i,y_i,z_i,
                                                      M,N, rF_i,iF_i);
                ck(cudaGetLastError(),"launch");
                ck(cudaDeviceSynchronize(),"sync");
            }

            // Download outputs & guards
            std::vector<float> rOutG(N+2*GUARD), iOutG(N+2*GUARD);
            ck(cudaMemcpy(rOutG.data(), d_rF, rOutG.size()*sizeof(float), cudaMemcpyDeviceToHost),"D2H r");
            ck(cudaMemcpy(iOutG.data(), d_iF, iOutG.size()*sizeof(float), cudaMemcpyDeviceToHost),"D2H i");

            // Validate
            auto guards_ok = [&](const std::vector<float>& g){
                for (size_t t=0;t<GUARD;++t)
                    if (g[t]!=1337.f || g[g.size()-1-t]!=1337.f) return false;
                return true;
            };
            bool ok = true;
            std::vector<float> rGot(N), iGot(N);
            if (N>0) {
                std::copy(rOutG.begin()+GUARD, rOutG.begin()+GUARD+N, rGot.begin());
                std::copy(iOutG.begin()+GUARD, iOutG.begin()+GUARD+N, iGot.begin());
            }
            ok = ok && almost_equal(rGot, rRef) && almost_equal(iGot, iRef);
            ok = ok && guards_ok(rOutG) && guards_ok(iOutG);

            printf("Case %-12s block=%3u -> %s\n", cs.name, bdim.x, ok?"OK":"FAIL");
            if (ok) ++pass;

            cudaFree(d_rPhi); cudaFree(d_iPhi); cudaFree(d_rD); cudaFree(d_iD);
            cudaFree(d_kx); cudaFree(d_ky); cudaFree(d_kz);
            cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
            cudaFree(d_rF); cudaFree(d_iF);
        }
    }

    printf("Summary: %d/%d passed\n", pass, total);
    return (pass==total)?0:1;
}