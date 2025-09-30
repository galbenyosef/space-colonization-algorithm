#include <cstdio>
#include <cuda_runtime.h>

__global__ void testKernel(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = i * 2;
}

int main() {
    int n = 256;
    int* d_data;
    cudaMalloc(&d_data, sizeof(int) * n);
    testKernel<<<1, 256>>>(d_data, n);
    cudaDeviceSynchronize();
    int result[256];
    cudaMemcpy(result, d_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) std::printf("%d ", result[i]);
    std::printf("\n");
    cudaFree(d_data);
    return 0;
}