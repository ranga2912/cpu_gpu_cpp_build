#ifndef CUDA_ADD_H
#define CUDA_ADD_H

#include <cuda_runtime.h>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// GPU kernel (simplified for 1 thread)
__global__ void addGPU(int* a, int* b, int* result) {
    *result = *a + *b;
}

// Function to perform addition on GPU
int addOnGPU(int a, int b) {
    int *d_a, *d_b, *d_result;
    int result = 0;  // Initialize to avoid undefined behavior

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice));

    addGPU<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

#endif  // CUDA_ADD_H
