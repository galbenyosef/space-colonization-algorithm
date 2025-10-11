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
#include <algorithm>

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static const float PI_F = 3.14159265358979323846f;

static const int   NUM_ATTRACTORS = 2200;
static const float STEP = 6.0f;
static const float KILL_DIST = 6.0f;
static const float INFLUENCE_DIST = 90.0f;
static const float WORLD_RADIUS = 330.0f;
static const int   MAX_BRANCHES = 7000;
static const int   MAX_SEGMENTS = 7000;
static const float DEPTH_FADE_RATE = 0.00035f;
static const bool  RAINBOW_MODE       = true;
static const float RAINBOW_DEPTH_SPIN = 0.9f;  
static const float RAINBOW_SAT        = 95.0f;  
static const float RAINBOW_BRI        = 100.0f; 

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
    float ax, ay, az;
    float bx, by, bz;
    int   depth;
    float hue, sat, bri, glow;
};

__global__ void pullKernel(
    const float* __restrict__ ax, const float* __restrict__ ay, const float* __restrict__ az,
    int nAttr, int* __restrict__ alive,
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    int nBranch,
    float* __restrict__ bdx, float* __restrict__ bdy, float* __restrict__ bdz,
    int* __restrict__ bcount,
    float killD2, float infD2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nAttr) return;

    float axx = ax[i], ayy = ay[i], azz = az[i];

    int   nearest = -1;
    float bestD2 = infD2;

    for (int b = 0; b < nBranch; ++b) {
        float dx = axx - bx[b];
        float dy = ayy - by[b];
        float dz = azz - bz[b];
       float d2 = dx * dx + dy * dy + dz * dz;

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
        float dz = azz - bz[nearest];
        float m = sqrtf(dx * dx + dy * dy + dz * dz);
        if (m > 0.0f) {
            float inv = 1.0f / m;
            atomicAdd(&bdx[nearest], dx * inv);
            atomicAdd(&bdy[nearest], dy * inv);
            atomicAdd(&bdz[nearest], dz * inv);
            atomicAdd(&bcount[nearest], 1);
        }
    }
}

struct BranchPoint {
    float x, y, z;
    int   depth;
};

static std::vector<float> g_ax, g_ay, g_az;
static std::vector<BranchPoint> g_branches;
static std::vector<bool> g_hasKid;
static std::vector<Segment> g_segments;

static float HUE_MIN = 0, HUE_MAX = 0, SAT_BASE = 0, BRI_BASE = 0;

static std::mt19937 g_rng(std::random_device{}());

static float* d_ax = nullptr, * d_ay = nullptr, * d_az = nullptr;
static int* d_alive = nullptr;
static float* d_bx = nullptr, * d_by = nullptr, * d_bz = nullptr;
static float* d_bdx = nullptr, * d_bdy = nullptr, * d_bdz = nullptr;
static int* d_bcount = nullptr;

static bool g_done = false;

static float uniform(float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g_rng);
}

static void randUnitVec(float& x, float& y, float& z) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    x = nd(g_rng); y = nd(g_rng); z = nd(g_rng);
    float m = std::sqrt(x * x + y * y + z * z);
    if (m < 1e-6f) { x = 1; y = 0; z = 0; return; }
    x /= m; y /= m; z /= m;
}

static void randPointInSphere(float radius, float& x, float& y, float& z) {
    float dx, dy, dz;
    randUnitVec(dx, dy, dz);
    float u = uniform(0.0f, 1.0f);
    float rr = radius * std::cbrt(u);
    x = dx * rr; y = dy * rr; z = dz * rr;
}

static bool insideWorld(float x, float y, float z) {
    return (x * x + y * y + z * z) <= (WORLD_RADIUS * WORLD_RADIUS);
}

static void pickColours() {
    if (RAINBOW_MODE) {
        HUE_MIN = 0.0f;
        HUE_MAX = 360.0f;
        SAT_BASE = 0.0f;
        BRI_BASE = RAINBOW_BRI;
        return;
    }

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


static float rainbow_hue(float x, float y, float z, int depth) {
    (void)y; (void)depth;
    float angleDeg = std::atan2(z, x) * 180.0f / PI_F;
    float hue = angleDeg + 180.0f;
    hue = std::fmod(hue, 360.0f);
    if (hue < 0.0f) hue += 360.0f;
    return hue;
}

static Segment makeSegment(float ax_, float ay_, float az_,
    float bx_, float by_, float bz_, int depth) {
    Segment s;
    s.ax = ax_; s.ay = ay_; s.az = az_;
    s.bx = bx_; s.by = by_; s.bz = bz_;
    s.depth = depth;
    s.hue = rainbow_hue(bx_, by_, bz_, depth);
    s.sat = clampf(RAINBOW_SAT + uniform(-5.0f, 5.0f), 0.0f, 100.0f);
    s.bri = clampf(RAINBOW_BRI + uniform(-5.0f, 0.0f), 0.0f, 100.0f);
    s.glow = uniform(0.9f, 1.25f);
    return s;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("rainbow 3d - wip\n");
    return 0;
}