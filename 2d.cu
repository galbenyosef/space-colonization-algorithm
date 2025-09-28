
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "opengl32.lib")
#endif

#include <SDL3/SDL.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>


static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static const float PI_F = 3.14159265358979323846f;

static const int   NUM_ATTRACTORS = 1500;
static const float STEP = 4.0f;
static const float KILL_DIST = 4.0f;
static const float INFLUENCE_DIST = 70.0f;
static const float WORLD_RADIUS = 300.0f;
static const int   MAX_BRANCHES = 5000;
static const int   MAX_SEGMENTS = 5000;
static const float DEPTH_FADE_RATE = 0.004f;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(_err));                \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static inline void hsb2rgb(float h, float s, float br, float& r, float& g, float& b) {
    h = std::fmod(h, 360.0f);
    if (h < 0) h += 360.0f;
    s = s / 100.0f;
    br = br / 100.0f;

    float c = br * s;
    float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = br - c;
    float rp, gp, bp;

    if (h < 60) { rp = c; gp = x; bp = 0; }
    else if (h < 120) { rp = x; gp = c; bp = 0; }
    else if (h < 180) { rp = 0; gp = c; bp = x; }
    else if (h < 240) { rp = 0; gp = x; bp = c; }
    else if (h < 300) { rp = x; gp = 0; bp = c; }
    else { rp = c; gp = 0; bp = x; }

    r = rp + m; g = gp + m; b = bp + m;
}

static inline float hashRange(int i, float lo, float hi) {
    float x = std::sin((float)i * 12.9898f) * 43758.5453f;
    float f = x - std::floor(x);
    return lo + f * (hi - lo);
}

struct Segment {
    float ax, ay;
    float bx, by;
    int   depth;
    float hue, sat, bri, glow;
};

struct BranchPoint {
    float x, y;
    int   depth;
};

__global__ void pullKernel(
    const float* __restrict__ ax, const float* __restrict__ ay,
    int nAttr, int* __restrict__ alive,
    const float* __restrict__ bx, const float* __restrict__ by,
    int nBranch,
    float* __restrict__ bdx, float* __restrict__ bdy,
    int* __restrict__ bcount,
    float killD2, float infD2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nAttr) return;

    float axx = ax[i], ayy = ay[i];

    int   nearest = -1;
    float bestD2 = infD2;

    for (int b = 0; b < nBranch; ++b) {
        float dx = axx - bx[b];
        float dy = ayy - by[b];
       float d2 = dx * dx + dy * dy;

        if (d2 < killD2) {
            alive[i] = 0;
            return;
        }
        if (d2 < bestD2) {
            bestD2 = d2;
            nearest = b;
        }
    }

    alive[i] = 1;

    if (nearest >= 0) {
        float dx = axx - bx[nearest];
        float dy = ayy - by[nearest];
        float m = sqrtf(dx * dx + dy * dy);
        if (m > 0.0f) {
            float inv = 1.0f / m;
            atomicAdd(&bdx[nearest], dx * inv);
            atomicAdd(&bdy[nearest], dy * inv);
            atomicAdd(&bcount[nearest], 1);
        }
    }
}

static std::vector<float> g_ax, g_ay;
static std::vector<BranchPoint> g_branches;
static std::vector<bool> g_hasKid;
static std::vector<Segment> g_segments;

static float HUE_MIN = 0, HUE_MAX = 0, SAT_BASE = 0, BRI_BASE = 0;

static std::mt19937 g_rng(std::random_device{}());

static float* d_ax = nullptr, * d_ay = nullptr;
static int* d_alive = nullptr;
static float* d_bx = nullptr, * d_by = nullptr;
static float* d_bdx = nullptr, * d_bdy = nullptr;
static int* d_bcount = nullptr;

static bool g_done = false;

static float uniform(float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g_rng);
}

static void randUnitVec(float& x, float& y) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    x = nd(g_rng); y = nd(g_rng);
    float m = std::sqrt(x * x + y * y);
    if (m < 1e-6f) { x = 1; y = 0; return; }
    x /= m; y /= m;
}

static void randPointInCirle(float radius, float& x, float& y) {
    float dx, dy;
    randUnitVec(dx, dy);
    float u = uniform(0.0f, 1.0f);
    float rr = radius * std::sqrt(u);
    x = dx * rr; y = dy * rr;
}

static bool insideWorld(float x, float y) {
    return (x * x + y * y) <= (WORLD_RADIUS * WORLD_RADIUS);
}

static void pickColours() {
    static const float ranges[8][2] = {
        {180, 215}, {200, 235}, {240, 275}, {280, 315},
        {330, 360}, {20, 55},   {70, 110},  {120, 160}
    };
    int idx = std::uniform_int_distribution<int>(0, 7)(g_rng);
    HUE_MIN = ranges[idx][0];
    HUE_MAX = ranges[idx][1];
    SAT_BASE = uniform(78.0f, 100.0f);
    BRI_BASE = uniform(85.0f, 100.0f);
}

static Segment makeSegment(float ax_, float ay_, float bx_, float by_, int depth) {
    Segment s;
    s.ax = ax_; s.ay = ay_;
    s.bx = bx_; s.by = by_;
    s.depth = depth;
    s.hue = uniform(HUE_MIN, HUE_MAX);
    s.sat = clampf(SAT_BASE + uniform(-10.0f, 10.0f), 0.0f, 100.0f);
    s.bri = clampf(BRI_BASE + uniform(-8.0f, 8.0f), 0.0f, 100.0f);
    s.glow = uniform(0.9f, 1.25f);
    return s;
}

static void initSystem() {
    g_ax.clear(); g_ay.clear();
    g_branches.clear();
    g_hasKid.clear();
    g_segments.clear();
    g_done = false;

    pickColours();

    g_ax.resize(NUM_ATTRACTORS);
    g_ay.resize(NUM_ATTRACTORS);
    for (int i = 0; i < NUM_ATTRACTORS; ++i) {
        randPointInCirle(WORLD_RADIUS, g_ax[i], g_ay[i]);
    }

    g_branches.push_back({ 0.0f, 0.0f, 0 });
    g_hasKid.push_back(false);

    Segment seed;
    seed.ax = seed.ay = 0.0f;
    seed.bx = seed.by = 0.0f;
    seed.depth = 0;
    seed.hue = (HUE_MIN + HUE_MAX) * 0.5f;
    seed.sat = SAT_BASE;
    seed.bri = BRI_BASE;
    seed.glow = 1.0f;
    g_segments.push_back(seed);
}

static void allocGpu() {
    CUDA_CHECK(cudaMalloc(&d_ax, sizeof(float) * NUM_ATTRACTORS));
    CUDA_CHECK(cudaMalloc(&d_ay, sizeof(float) * NUM_ATTRACTORS));
    CUDA_CHECK(cudaMalloc(&d_alive, sizeof(int) * NUM_ATTRACTORS));
    CUDA_CHECK(cudaMalloc(&d_bx, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_by, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bdx, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bdy, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bcount, sizeof(int) * MAX_BRANCHES));
}

static void freeGpu() {
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_alive);
    cudaFree(d_bx); cudaFree(d_by);
    cudaFree(d_bdx); cudaFree(d_bdy); cudaFree(d_bcount);
}

static void growOneStep() {
    if (g_done) return;
    if (g_ax.empty() || (int)g_branches.size() >= MAX_BRANCHES) {
        g_done = true;
        return;
    }

    int nAttr = (int)g_ax.size();
    int nBranch = (int)g_branches.size();

    CUDA_CHECK(cudaMemcpy(d_ax, g_ax.data(), sizeof(float) * nAttr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ay, g_ay.data(), sizeof(float) * nAttr, cudaMemcpyHostToDevice));

    {
        std::vector<float> bx(nBranch), by(nBranch);
        for (int i = 0; i < nBranch; ++i) {
            bx[i] = g_branches[i].x; by[i] = g_branches[i].y;
        }
        CUDA_CHECK(cudaMemcpy(d_bx, bx.data(), sizeof(float) * nBranch, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_by, by.data(), sizeof(float) * nBranch, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(d_bdx, 0, sizeof(float) * nBranch));
    CUDA_CHECK(cudaMemset(d_bdy, 0, sizeof(float) * nBranch));
    CUDA_CHECK(cudaMemset(d_bcount, 0, sizeof(int) * nBranch));

    const int threads = 256;
    const int blocks = (nAttr + threads - 1) / threads;

    pullKernel << <blocks, threads >> > (
        d_ax, d_ay, nAttr, d_alive,
        d_bx, d_by, nBranch,
        d_bdx, d_bdy, d_bcount,
        KILL_DIST * KILL_DIST, INFLUENCE_DIST * INFLUENCE_DIST);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int>   alive(nAttr);
    std::vector<float> bdx(nBranch), bdy(nBranch);
    std::vector<int>   bcount(nBranch);

    CUDA_CHECK(cudaMemcpy(alive.data(), d_alive, sizeof(int) * nAttr, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bdx.data(), d_bdx, sizeof(float) * nBranch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bdy.data(), d_bdy, sizeof(float) * nBranch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bcount.data(), d_bcount, sizeof(int) * nBranch, cudaMemcpyDeviceToHost));

    {
        std::vector<float> nax, nay;
        nax.reserve(nAttr); nay.reserve(nAttr);
        for (int i = 0; i < nAttr; ++i) {
            if (alive[i]) {
                nax.push_back(g_ax[i]);
                nay.push_back(g_ay[i]);
            }
        }
        g_ax = std::move(nax);
        g_ay = std::move(nay);
    }

    int branchTotal = (int)g_branches.size();
    std::vector<BranchPoint> freshBranches;
    std::vector<Segment> freshSegments;

    for (int b = 0; b < nBranch; ++b) {
        if (bcount[b] <= 0) continue;
        if (branchTotal + (int)freshBranches.size() >= MAX_BRANCHES) break;

        float inv = 1.0f / (float)bcount[b];
        float dx = bdx[b] * inv, dy = bdy[b] * inv;
        float m = std::sqrt(dx * dx + dy * dy);
        if (m == 0.0f) continue;

        dx = (dx / m) * STEP;
        dy = (dy / m) * STEP;

        const BranchPoint& parent = g_branches[b];
        float nx = parent.x + dx, ny = parent.y + dy;
        if (!insideWorld(nx, ny)) continue;

        g_hasKid[b] = true;

        int newDepth = parent.depth + 1;
        freshBranches.push_back({ nx, ny, newDepth });
        freshSegments.push_back(makeSegment(parent.x, parent.y, nx, ny, newDepth));
    }

    if (!freshBranches.empty()) {
        g_branches.insert(g_branches.end(), freshBranches.begin(), freshBranches.end());
        g_segments.insert(g_segments.end(), freshSegments.begin(), freshSegments.end());
        g_hasKid.resize(g_branches.size(), false);
    }

    if ((int)g_segments.size() > MAX_SEGMENTS) {
        g_segments.erase(g_segments.begin(), g_segments.begin() + (g_segments.size() - MAX_SEGMENTS));
    }

    if (g_ax.empty() || (int)g_branches.size() >= MAX_BRANCHES) {
        g_done = true;
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("growth loop working\n");
    return 0;
}