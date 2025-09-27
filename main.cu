
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

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::printf("space colonization algo - scaffold\n");
    return 0;
}