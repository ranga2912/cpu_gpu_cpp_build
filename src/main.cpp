#include <iostream>

// Conditionally include CUDA header and define GPU function
#ifdef USE_CUDA
#include "cuda_add.h"
#endif

int main() {
    int variable1 = 5;
    int variable2 = 10;
    int sum;

#ifdef USE_CUDA
    // Use GPU if enabled
    sum = addOnGPU(variable1, variable2);
    std::cout << "=== GPU Addition Demo ===" << std::endl;
    std::cout << "GPU calculation: " << variable1 << " + " << variable2 << " = " << sum << std::endl;
    std::cout << "GPU acceleration working!" << std::endl;
#else
    // Fallback to CPU
    sum = variable1 + variable2;
    std::cout << "The sum of " << variable1 << " and " << variable2 << " is " << sum << "." << std::endl;
#endif

    return 0;
}
