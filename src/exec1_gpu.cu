#include <iostream>
#include <cuda_runtime.h>

// GPU kernel to perform addition
__global__ void addGPU(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {  // Only one thread needs to do this simple operation
        *result = *a + *b;
    }
}

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

int main() {
    // Declare and initialize the variables
    int variable1 = 5;
    int variable2 = 10;
    
    // GPU calculation ONLY
    int *d_a, *d_b, *d_result;
    int result;
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_a, &variable1, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &variable2, sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch GPU kernel
    addGPU<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back from GPU
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    // Print the GPU result
    std::cout << "=== GPU Addition Demo ===" << std::endl;
    std::cout << "GPU calculation: " << variable1 << " + " << variable2 << " = " << result << std::endl;
    std::cout << "GPU acceleration working!" << std::endl;
    
    return 0;
}
