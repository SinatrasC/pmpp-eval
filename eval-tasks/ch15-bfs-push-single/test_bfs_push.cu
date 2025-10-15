// ch15-bfs-push-single / test_bfs_push.cu
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cassert>
#include <random>

extern "C" void bfs_push_gpu(const int* d_row_ptr,
                             const int* d_col_idx,
                             int V, int E,
                             int src,
                             int* d_level);

static void ck(cudaError_t e, const char* m){
    if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s: %s\n", m, cudaGetErrorString(e)); std::exit(2); }
}

struct CSR {
    int V=0; int E=0;
    std::vector<int> row_ptr; // V+1
    std::vector<int> col_idx; // E
};

static CSR make_chain(int V){
    CSR g; g.V=V; g.row_ptr.resize(V+1,0);
    std::vector<std::vector<int>> adj(V);
    for(int i=0;i<V-1;i++){ adj[i].push_back(i+1); adj[i+1].push_back(i); }
    int E=0; for(int i=0;i<V;i++){ E+= (int)adj[i].size(); }
    g.E=E; g.col_idx.resize(E);
    int off=0; for(int i=0;i<V;i++){ g.row_ptr[i]=off; for(int v:adj[i]) g.col_idx[off++]=v; }
    g.row_ptr[V]=off; return g;
}

static CSR make_star(int V){
    CSR g; g.V=V; g.row_ptr.resize(V+1,0);
    std::vector<std::vector<int>> adj(V);
    if(V>0){ for(int i=1;i<V;i++){ adj[0].push_back(i); adj[i].push_back(0);} }
    int E=0; for(int i=0;i<V;i++) E+=(int)adj[i].size();
    g.E=E; g.col_idx.resize(E);
    int off=0; for(int i=0;i<V;i++){ g.row_ptr[i]=off; for(int v:adj[i]) g.col_idx[off++]=v; }
    g.row_ptr[V]=off; return g;
}

static CSR make_two_components(int a, int b){
    CSR g; g.V=a+b; g.row_ptr.resize(g.V+1,0);
    std::vector<std::vector<int>> adj(g.V);
    for(int i=0;i<a-1;i++){ adj[i].push_back(i+1); adj[i+1].push_back(i); }
    for(int i=0;i<b-1;i++){ int u=a+i, v=a+i+1; adj[u].push_back(v); adj[v].push_back(u); }
    int E=0; for(int i=0;i<g.V;i++) E+=(int)adj[i].size();
    g.E=E; g.col_idx.resize(E);
    int off=0; for(int i=0;i<g.V;i++){ g.row_ptr[i]=off; for(int v:adj[i]) g.col_idx[off++]=v; }
    g.row_ptr[g.V]=off; return g;
}

static CSR make_grid2d(int W,int H){
    int V=W*H; auto id=[&](int x,int y){return y*W+x;};
    CSR g; g.V=V; g.row_ptr.resize(V+1,0);
    std::vector<std::vector<int>> adj(V);
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
        int u=id(x,y);
        if(x+1<W){ adj[u].push_back(id(x+1,y)); adj[id(x+1,y)].push_back(u); }
        if(y+1<H){ adj[u].push_back(id(x,y+1)); adj[id(x,y+1)].push_back(u); }
    }
    int E=0; for(int i=0;i<V;i++) E+=(int)adj[i].size();
    g.E=E; g.col_idx.resize(E);
    int off=0; for(int i=0;i<V;i++){ g.row_ptr[i]=off; for(int v:adj[i]) g.col_idx[off++]=v; }
    g.row_ptr[V]=off; return g;
}

static CSR make_erdos(int V, float p, unsigned seed=123){
    CSR g; g.V=V; g.row_ptr.resize(V+1,0);
    std::vector<std::vector<int>> adj(V);
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(0,1);
    for(int i=0;i<V;i++) for(int j=i+1;j<V;j++) if(U(rng)<p){ adj[i].push_back(j); adj[j].push_back(i); }
    int E=0; for(int i=0;i<V;i++) E+=(int)adj[i].size();
    g.E=E; g.col_idx.resize(E);
    int off=0; for(int i=0;i<V;i++){ g.row_ptr[i]=off; for(int v:adj[i]) g.col_idx[off++]=v; }
    g.row_ptr[V]=off; return g;
}

static std::vector<int> cpu_bfs_levels(const CSR& g, int src){
    const int INF = 0x3f3f3f3f;
    std::vector<int> lvl(g.V, INF);
    if(g.V==0) return lvl;
    std::queue<int>q; lvl[src]=0; q.push(src);
    while(!q.empty()){
        int u=q.front(); q.pop();
        for(int e=g.row_ptr[u]; e<g.row_ptr[u+1]; ++e){
            int v=g.col_idx[e];
            if(lvl[v]==INF){ lvl[v]=lvl[u]+1; q.push(v); }
        }
    }
    return lvl;
}

int main(){
    printf("ch15-bfs-push-single tests\n");

    std::vector<CSR> gs;
    gs.push_back(make_chain(1));
    gs.push_back(make_chain(10));
    gs.push_back(make_star(101));
    gs.push_back(make_two_components(10,15));
    gs.push_back(make_grid2d(8,8));   // 64
    gs.push_back(make_grid2d(16,16)); // 256
    gs.push_back(make_erdos(200, 0.03f, 777));

    const int SENT = 0xDEADBEEF;
    int total=0, passed=0;

    for(size_t gi=0; gi<gs.size(); ++gi){
        const CSR& g = gs[gi];
        int V=g.V, E=g.E, src=0;
        ++total;

        // CPU oracle
        auto ref = cpu_bfs_levels(g, src);

        // Guarded device buffers
        size_t GU = 1024;
        int *d_row_all=nullptr, *d_col_all=nullptr, *d_lvl_all=nullptr;
        ck(cudaMalloc(&d_row_all, (V+1+2*GU)*sizeof(int)), "malloc row");
        ck(cudaMalloc(&d_col_all, (E  +2*GU)*sizeof(int)), "malloc col");
        ck(cudaMalloc(&d_lvl_all, (V  +2*GU)*sizeof(int)), "malloc lvl");

        std::vector<int> h_row_guard(V+1+2*GU, SENT);
        std::vector<int> h_col_guard(E  +2*GU, SENT);
        std::vector<int> h_lvl_guard(V  +2*GU, SENT);

        if(V>0){ std::copy(g.row_ptr.begin(), g.row_ptr.end(), h_row_guard.begin()+GU); }
        if(E>0){ std::copy(g.col_idx.begin(), g.col_idx.end(), h_col_guard.begin()+GU); }

        ck(cudaMemcpy(d_row_all, h_row_guard.data(), (V+1+2*GU)*sizeof(int), cudaMemcpyHostToDevice), "H2D row");
        ck(cudaMemcpy(d_col_all, h_col_guard.data(), (E  +2*GU)*sizeof(int), cudaMemcpyHostToDevice), "H2D col");
        ck(cudaMemcpy(d_lvl_all, h_lvl_guard.data(), (V  +2*GU)*sizeof(int), cudaMemcpyHostToDevice), "H2D lvl");

        int* d_row = d_row_all + GU;
        int* d_col = d_col_all + GU;
        int* d_lvl = d_lvl_all + GU;

        // RUN
        bfs_push_gpu(d_row, d_col, V, E, src, d_lvl);
        ck(cudaDeviceSynchronize(), "sync");

        // Download
        ck(cudaMemcpy(h_row_guard.data(), d_row_all, (V+1+2*GU)*sizeof(int), cudaMemcpyDeviceToHost), "D2H row");
        ck(cudaMemcpy(h_col_guard.data(), d_col_all, (E  +2*GU)*sizeof(int), cudaMemcpyDeviceToHost), "D2H col");
        ck(cudaMemcpy(h_lvl_guard.data(), d_lvl_all, (V  +2*GU)*sizeof(int), cudaMemcpyDeviceToHost), "D2H lvl");

        // Extract level
        std::vector<int> got(V, SENT);
        if(V>0) std::copy(h_lvl_guard.begin()+GU, h_lvl_guard.begin()+GU+V, got.begin());

        // Checks
        auto guards_ok = [&](const std::vector<int>& gbuf){
            for(size_t t=0;t<GU;t++){ if(gbuf[t]!=SENT || gbuf[gbuf.size()-1-t]!=SENT) return false; }
            return true;
        };

        bool ok = true;
        // CSR input immutability
        if(V>0){
            std::vector<int> row_in(V+1);
            std::copy(h_row_guard.begin()+GU, h_row_guard.begin()+GU+V+1, row_in.begin());
            ok = ok && (row_in == g.row_ptr);
        }
        if(E>0){
            std::vector<int> col_in(E);
            std::copy(h_col_guard.begin()+GU, h_col_guard.begin()+GU+E, col_in.begin());
            ok = ok && (col_in == g.col_idx);
        }
        ok = ok && guards_ok(h_row_guard) && guards_ok(h_col_guard) && guards_ok(h_lvl_guard);
        ok = ok && (got == ref);

        printf("Graph %zu (V=%d,E=%d) -> %s\n", gi, V, E, ok?"OK":"FAIL");
        if(ok) ++passed;

        cudaFree(d_row_all); cudaFree(d_col_all); cudaFree(d_lvl_all);
    }

    printf("Summary: %d/%d passed\n", passed, total);
    return (passed==total)?0:1;
}