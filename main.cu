
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

// TODO: figure out good values
static const int   NUM_ATTRACTORS = 1500;
static const float STEP = 4.0f;
static const float KILL_DIST = 4.0f;
static const float INFLUENCE_DIST = 70.0f;
static const float WORLD_RADIUS = 300.0f;
static const int   MAX_BRANCHES = 5000;
static const int   MAX_SEGMENTS = 5000;

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

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("space colonization algo - scaffold\n");
    return 0;
}