// ch17-fhd-fission-two-kernels-single/test_fhd_fission.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

extern "C" __global__
void compute_mu_kernel(const float*, const float*, const float*, const float*, int, float*, float*);
extern "C" __global__
void fhd_accumulate_mu_kernel(const float*, const float*, const float*, const float*, const float*,
                              const float*, const float*, const float*, int, int, float*, float*);

static void ck(cudaError_t e,const char* m){ if(e){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

static void cpu_fused(const std::vector<float>& rPhi,
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
                      std::vector<float>& rOut,
                      std::vector<float>& iOut)
{
    const float TWO_PI = 6.2831853071795864769f;
    for (int n=0;n<N;++n){
        float rn=0.f, in=0.f;
        float xn=x[n], yn=y[n], zn=z[n];
        for (int m=0;m<M;++m){
            float rmu = rPhi[m]*rD[m] + iPhi[m]*iD[m];
            float imu = rPhi[m]*iD[m] - iPhi[m]*rD[m];
            float ang = TWO_PI*(kx[m]*xn + ky[m]*yn + kz[m]*zn);
            float c=std::cos(ang), s=std::sin(ang);
            rn += rmu*c - imu*s;
            in += imu*c + rmu*s;
        }
        rOut[n]=rn; iOut[n]=in;
    }
}

static bool eq(const std::vector<float>& a,const std::vector<float>& b,float eps=1e-4f){
    if (a.size()!=b.size()) return false;
    for (size_t i=0;i<a.size();++i){
        float d=std::fabs(a[i]-b[i]), m=std::max(1.f,std::max(std::fabs(a[i]),std::fabs(b[i])));
        if (!(d<=eps || d<=eps*m)) return false;
    }
    return true;
}

int main(){
    printf("ch17-fhd-fission-two-kernels-single tests\n");
    struct Case{int M,N; const char* name;};
    const Case cases[]={{0,0,"0,0"},{0,5,"0,5"},{7,0,"7,0"},{3,4,"3,4"},{17,8,"17,8"},{64,63,"64,63"},{128,129,"128,129"}};
    const dim3 blkM(256), blkN(256);

    int total=0, pass=0;

    for (auto cs : cases){
        ++total;
        int M=cs.M,N=cs.N;

        std::vector<float> rPhi(M), iPhi(M), rD(M), iD(M), kx(M), ky(M), kz(M);
        std::vector<float> x(N), y(N), z(N);
        for (int m=0;m<M;++m){
            rPhi[m]=0.1f+0.01f*m; iPhi[m]=-0.2f+0.02f*m;
            rD[m]=0.15f-0.003f*m; iD[m]=0.05f+0.004f*m;
            kx[m]=0.001f*(1+m%7); ky[m]=0.002f*(1+m%5); kz[m]=0.0015f*(1+m%3);
        }
        for (int n=0;n<N;++n){ x[n]=0.01f*n; y[n]=0.02f*n; z[n]=0.015f*n; }

        std::vector<float> rRef(N,0.f), iRef(N,0.f);
        cpu_fused(rPhi,iPhi,rD,iD,kx,ky,kz,x,y,z,M,N,rRef,iRef);

        // Device
        auto cdiv=[](int a,int b){return (a+b-1)/b;};
        float *d_rPhi,*d_iPhi,*d_rD,*d_iD,*d_kx,*d_ky,*d_kz,*d_x,*d_y,*d_z,*d_rMu,*d_iMu,*d_rF,*d_iF;
        ck(cudaMalloc(&d_rPhi,M*sizeof(float)),"m"); ck(cudaMalloc(&d_iPhi,M*sizeof(float)),"m");
        ck(cudaMalloc(&d_rD,M*sizeof(float)),"m");   ck(cudaMalloc(&d_iD,M*sizeof(float)),"m");
        ck(cudaMalloc(&d_kx,M*sizeof(float)),"m");   ck(cudaMalloc(&d_ky,M*sizeof(float)),"m");
        ck(cudaMalloc(&d_kz,M*sizeof(float)),"m");
        ck(cudaMalloc(&d_x,N*sizeof(float)),"n");    ck(cudaMalloc(&d_y,N*sizeof(float)),"n"); ck(cudaMalloc(&d_z,N*sizeof(float)),"n");
        ck(cudaMalloc(&d_rMu,M*sizeof(float)),"m");  ck(cudaMalloc(&d_iMu,M*sizeof(float)),"m");
        ck(cudaMalloc(&d_rF,N*sizeof(float)),"n");   ck(cudaMalloc(&d_iF,N*sizeof(float)),"n");

        if(M){ ck(cudaMemcpy(d_rPhi,rPhi.data(),M*sizeof(float),cudaMemcpyHostToDevice),"H2D"); ck(cudaMemcpy(d_iPhi,iPhi.data(),M*sizeof(float),cudaMemcpyHostToDevice),"H2D");
               ck(cudaMemcpy(d_rD,rD.data(),M*sizeof(float),cudaMemcpyHostToDevice),"H2D");   ck(cudaMemcpy(d_iD,iD.data(),M*sizeof(float),cudaMemcpyHostToDevice),"H2D");
               ck(cudaMemcpy(d_kx,kx.data(),M*sizeof(float),cudaMemcpyHostToDevice),"H2D");   ck(cudaMemcpy(d_ky,ky.data(),M*sizeof(float),cudaMemcpyHostToDevice),"H2D");
               ck(cudaMemcpy(d_kz,kz.data(),M*sizeof(float),cudaMemcpyHostToDevice),"H2D"); }
        if(N){ ck(cudaMemcpy(d_x,x.data(),N*sizeof(float),cudaMemcpyHostToDevice),"H2D"); ck(cudaMemcpy(d_y,y.data(),N*sizeof(float),cudaMemcpyHostToDevice),"H2D");
               ck(cudaMemcpy(d_z,z.data(),N*sizeof(float),cudaMemcpyHostToDevice),"H2D"); }

        ck(cudaMemset(d_rF,0, N*sizeof(float)),"ms"); ck(cudaMemset(d_iF,0, N*sizeof(float)),"ms");

        // Launch fission path
        dim3 gridM(cdiv(M,(int)blkM.x)), gridN(cdiv(N,(int)blkN.x));
        if (M) compute_mu_kernel<<<gridM, blkM>>>(d_rPhi,d_iPhi,d_rD,d_iD,M,d_rMu,d_iMu);
        ck(cudaGetLastError(),"mu"); ck(cudaDeviceSynchronize(),"mu sync");
        if (N) fhd_accumulate_mu_kernel<<<gridN, blkN>>>(d_rMu,d_iMu,d_kx,d_ky,d_kz,d_x,d_y,d_z,M,N,d_rF,d_iF);
        ck(cudaGetLastError(),"acc"); ck(cudaDeviceSynchronize(),"acc sync");

        std::vector<float> rGot(N), iGot(N);
        if (N){ ck(cudaMemcpy(rGot.data(), d_rF, N*sizeof(float), cudaMemcpyDeviceToHost),"D2H");
                ck(cudaMemcpy(iGot.data(), d_iF, N*sizeof(float), cudaMemcpyDeviceToHost),"D2H"); }

        bool ok = eq(rGot,rRef) && eq(iGot,iRef);
        printf("Case M=%-4d N=%-4d -> %s\n", M, N, ok?"OK":"FAIL");
        if (ok) ++pass;

        cudaFree(d_rPhi); cudaFree(d_iPhi); cudaFree(d_rD); cudaFree(d_iD);
        cudaFree(d_kx); cudaFree(d_ky); cudaFree(d_kz);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_rMu); cudaFree(d_iMu); cudaFree(d_rF); cudaFree(d_iF);
    }
    printf("Summary: %d/%d passed\n", pass, total);
    return (pass==total)?0:1;
}