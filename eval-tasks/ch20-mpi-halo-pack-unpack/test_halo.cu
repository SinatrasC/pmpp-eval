#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cassert>

extern "C" void halo_pack_boundaries(const float* d_grid,
                                     int dimx,int dimy,int dimz,
                                     float* d_left_send,
                                     float* d_right_send);
extern "C" void halo_unpack_to_halos(float* d_grid,
                                     int dimx,int dimy,int dimz,
                                     const float* d_left_recv,
                                     const float* d_right_recv);

static inline size_t idx3(int i,int j,int k,int dx,int dy){ return (size_t(k)*dy + j)*dx + i; }
static inline size_t pack_idx(int p,int j,int i,int dx,int dy){ return (size_t(p)*dy + j)*dx + i; }

static void ck(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(2);} }

static void fill_grid(std::vector<float>& g){
    for(size_t t=0;t<g.size();++t){
        g[t] = 0.01f*float(t%173) - 0.37f*float((t*13)%97);
    }
}
static void fill_buf(std::vector<float>& b, int seed){
    for(size_t t=0;t<b.size();++t) b[t] = 0.5f*seed + 0.001f*float((t*29)%101);
}

static bool equalv(const std::vector<float>& a, const std::vector<float>& b){
    if(a.size()!=b.size()) return false;
    for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]) return false;
    return true;
}

int main(){
    printf("ch20-mpi-halo-pack-unpack tests\n");
    struct C{int x,y,z;}; // z is OWNED depth
    const C cases[] = {{8,8,8}, {17,13,9}, {32,16,10}, {48,48,40}};

    const size_t GUARD=4096;
    const float SENT=1337.0f;

    int total=0, pass=0;

    for(auto cs : cases){
        ++total;
        int dx=cs.x, dy=cs.y, dz=cs.z;
        int totz = dz + 8;
        size_t Ngrid = size_t(dx)*dy*totz;
        size_t Npack = size_t(4)*dy*dx;

        // host buffers with guards
        std::vector<float> h_grid(Ngrid), h_grid_after(Ngrid);
        std::vector<float> h_left(Npack), h_right(Npack);
        std::vector<float> h_left_ref(Npack), h_right_ref(Npack);

        fill_grid(h_grid);

        // CPU oracle for PACK
        for(int p=0;p<4;++p){
            int kL = 4 + p;
            int kR = (4+dz-1) - 3 + p;
            for(int j=0;j<dy;++j)
            for(int i=0;i<dx;++i){
                size_t dst = pack_idx(p,j,i,dx,dy);
                h_left_ref [dst] = h_grid[idx3(i,j,kL,dx,dy)];
                h_right_ref[dst] = h_grid[idx3(i,j,kR,dx,dy)];
            }
        }

        // Guards
        std::vector<float> h_grid_guard(Ngrid+2*GUARD, SENT);
        std::copy(h_grid.begin(),h_grid.end(),h_grid_guard.begin()+GUARD);
        std::vector<float> h_left_guard (Npack+2*GUARD, SENT);
        std::vector<float> h_right_guard(Npack+2*GUARD, SENT);

        float *d_grid_all=nullptr, *d_left_all=nullptr, *d_right_all=nullptr;
        ck(cudaMalloc(&d_grid_all,  (Ngrid+2*GUARD)*sizeof(float)),"malloc grid");
        ck(cudaMalloc(&d_left_all,  (Npack+2*GUARD)*sizeof(float)),"malloc left");
        ck(cudaMalloc(&d_right_all, (Npack+2*GUARD)*sizeof(float)),"malloc right");
        ck(cudaMemcpy(d_grid_all, h_grid_guard.data(), (Ngrid+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice),"H2D grid");
        ck(cudaMemcpy(d_left_all, h_left_guard.data(), (Npack+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice),"H2D left");
        ck(cudaMemcpy(d_right_all,h_right_guard.data(),(Npack+2*GUARD)*sizeof(float), cudaMemcpyHostToDevice),"H2D right");

        float* d_grid = d_grid_all + GUARD;
        float* d_left = d_left_all + GUARD;
        float* d_right= d_right_all + GUARD;

        // PACK
        halo_pack_boundaries(d_grid, dx,dy,dz, d_left, d_right);

        // Download and check
        ck(cudaMemcpy(h_left_guard.data(),  d_left_all,  (Npack+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost),"D2H left");
        ck(cudaMemcpy(h_right_guard.data(), d_right_all, (Npack+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost),"D2H right");
        std::copy(h_left_guard.begin()+GUARD,  h_left_guard.begin()+GUARD+Npack,  h_left.begin());
        std::copy(h_right_guard.begin()+GUARD, h_right_guard.begin()+GUARD+Npack, h_right.begin());

        auto guard_ok=[&](const std::vector<float>& g){
            for(size_t t=0;t<GUARD;t++){
                if(g[t]!=SENT || g[g.size()-1-t]!=SENT) return false;
            } return true;
        };
        bool ok_pack = equalv(h_left, h_left_ref) && equalv(h_right, h_right_ref)
                       && guard_ok(h_left_guard) && guard_ok(h_right_guard);

        // UNPACK test: write known buffers into halos
        std::vector<float> recvL(Npack), recvR(Npack);
        fill_buf(recvL, 7); fill_buf(recvR, 11);
        std::vector<float> recvL_guard(Npack+2*GUARD, SENT), recvR_guard(Npack+2*GUARD, SENT);
        std::copy(recvL.begin(),recvL.end(),recvL_guard.begin()+GUARD);
        std::copy(recvR.begin(),recvR.end(),recvR_guard.begin()+GUARD);

        float *d_recvL_all=nullptr,*d_recvR_all=nullptr;
        ck(cudaMalloc(&d_recvL_all,(Npack+2*GUARD)*sizeof(float)),"malloc rL");
        ck(cudaMalloc(&d_recvR_all,(Npack+2*GUARD)*sizeof(float)),"malloc rR");
        ck(cudaMemcpy(d_recvL_all,recvL_guard.data(),(Npack+2*GUARD)*sizeof(float),cudaMemcpyHostToDevice),"H2D rL");
        ck(cudaMemcpy(d_recvR_all,recvR_guard.data(),(Npack+2*GUARD)*sizeof(float),cudaMemcpyHostToDevice),"H2D rR");
        float* d_recvL = d_recvL_all + GUARD;
        float* d_recvR = d_recvR_all + GUARD;

        // run UNPACK
        halo_unpack_to_halos(d_grid, dx,dy,dz, d_recvL, d_recvR);

        // download grid back
        ck(cudaMemcpy(h_grid_guard.data(), d_grid_all,(Ngrid+2*GUARD)*sizeof(float), cudaMemcpyDeviceToHost),"D2H grid");
        std::copy(h_grid_guard.begin()+GUARD, h_grid_guard.begin()+GUARD+Ngrid, h_grid_after.begin());

        // CPU oracle for UNPACK
        std::vector<float> h_grid_ref = h_grid; // start from original
        for(int p=0;p<4;++p){
            int kL = 0 + p;
            int kR = dz + 4 + p;
            for(int j=0;j<dy;++j)
            for(int i=0;i<dx;++i){
                size_t s = pack_idx(p,j,i,dx,dy);
                h_grid_ref[idx3(i,j,kL,dx,dy)] = recvL[s];
                h_grid_ref[idx3(i,j,kR,dx,dy)] = recvR[s];
            }
        }

        bool ok_unpack = equalv(h_grid_after, h_grid_ref) && guard_ok(h_grid_guard);

        bool ok = ok_pack && ok_unpack;
        printf("Case %3dx%3dx%3d -> %s\n", dx,dy,dz, ok?"OK":"FAIL");
        if(ok) ++pass;

        cudaFree(d_recvL_all); cudaFree(d_recvR_all);
        cudaFree(d_grid_all); cudaFree(d_left_all); cudaFree(d_right_all);
    }

    printf("Summary: %d/%d passed\n", pass,total);
    return (pass==total)?0:1;
}