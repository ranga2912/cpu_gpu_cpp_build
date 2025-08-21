# Smart C++ and CUDA Build System

This guide shows you how to use **intelligent batch scripts** that automatically detect source files and handle both CPU and GPU compilation with robust compatibility handling.

## **🚀 Quick Start**

### **CPU Compilation (Automatic)**
```cmd
# Auto-detect and build all C++ files in src/
.\build_cpu_generic.bat

# Build specific file (if you have the old PowerShell version)
.\build_cpu_generic.ps1 hello.cpp
```

### **GPU Compilation (Automatic)**
```cmd
# Auto-detect and build all CUDA files in src/
.\build_gpu_generic.bat

# Build specific file (if you have the old PowerShell version)
.\build_gpu_generic.ps1 hello_gpu.cu
```

---

## **🧠 Smart Features**

### **Auto-Detection**
- **Searches `src/` directory** for `.cpp` or `.cu` files
- **Automatic compilation** of all found files
- **No interactive selection** - builds everything automatically
- **Automatic path resolution** for `src/` files

### **User-Friendly**
- **No hardcoded filenames** - works with any project
- **Automatic compiler detection** - finds best available compiler
- **Clear error messages** with suggestions
- **Cross-compiler support** (MSVC, Clang++, G++)

---

## **📁 Project Structure Support**

The scripts work with this project layout:

```
Project Root/
├── src/
│   ├── exec1.cpp                 # ✅ Auto-detected and compiled
│   ├── exec2.cpp                 # ✅ Auto-detected and compiled
│   └── exec1_gpu.cu             # ✅ Auto-detected and compiled
├── build/                        # Created automatically for CPU builds
└── build_gpu/                    # Created automatically for GPU builds
```

---

## **💡 Example Programs**

### **1. Simple Addition (CPU) - `exec1.cpp`**
```cpp
#include <iostream>

int main() {
    int variable1 = 5;
    int variable2 = 10;
    int sum = variable1 + variable2;
    std::cout << "The sum of " << variable1 << " and " << variable2 << " is " << sum << "." << std::endl;
    return 0;
}
```

**Build:** `.\build_cpu_generic.bat`  
**Run:** `.\build\exec1.exe`

### **2. GPU-Accelerated Addition - `exec1_gpu.cu`**
```cuda
#include <iostream>
#include <cuda_runtime.h>

__global__ void addGPU(int* a, int* b, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        *result = *a + *b;
    }
}

int main() {
    int variable1 = 5;
    int variable2 = 10;
    
    // GPU calculation
    int *d_a, *d_b, *d_result;
    int h_result;
    
    // Allocate GPU memory
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(d_a, &variable1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &variable2, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch GPU kernel
    addGPU<<<1, 1>>>(d_a, d_b, d_result);
    cudaDeviceSynchronize();
    
    // Copy result back from GPU
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print the results
    std::cout << "=== GPU Addition Demo ===" << std::endl;
    std::cout << "GPU calculation: " << variable1 << " + " << variable2 << " = " << h_result << std::endl;
    std::cout << "GPU acceleration working!" << std::endl;
    
    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    return 0;
}
```

**Build:** `.\build_gpu_generic.bat`  
**Run:** `.\build_gpu\exec1_gpu.exe`

---

## **🔧 Build Script Behavior**

### **CPU Script (`build_cpu_generic.bat`)**

**Automatic compilation:**
```cmd
C:\> .\build_cpu_generic.bat
Checking for C++ files in src/ directory...
Found .cpp files in src/:
exec1.cpp
exec2.cpp

Checking for compilers...
Using GCC (g++)
Building all .cpp files...
Building exec1.cpp...
Success: exec1.exe
Building exec2.cpp...
Success: exec2.exe

Build complete! Check the build/ directory for executables.
```

**Compiler Priority:**
1. **GCC (g++)** - First choice, most compatible
2. **MSVC (cl)** - Second choice, Windows native
3. **Clang (clang++)** - Third choice, modern compiler

### **GPU Script (`build_gpu_generic.bat`)**

**Automatic compilation with compatibility handling:**
```cmd
C:\> .\build_gpu_generic.bat
Checking for CUDA files in src/ directory...
Found .cu files in src/:
exec1_gpu.cu

Setting up MSVC environment...
Found MSVC at: C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64
Checking for CUDA compiler...
Warning: CUDA 12.1 detected in PATH - incompatible with VS2022
Using CUDA 12.6
Building all .cu files...
Building exec1_gpu.cu...
Success: exec1_gpu.exe

GPU build complete! Check the build_gpu/ directory for executables.
```

**CUDA Version Priority:**
1. **CUDA 12.4** - Most compatible with VS2022
2. **PATH nvcc** - If not CUDA 12.1 (avoid compatibility issues)
3. **CUDA 12.6, 12.0, 11.8** - Stable versions
4. **CUDA 12.1** - Last resort with compatibility flag

---

## **📊 Comparison Table**

| Feature | CPU Script | GPU Script |
|---------|------------|------------|
| **File Types** | `.cpp` | `.cu` |
| **Auto-Detection** | ✅ src/ directory | ✅ src/ directory |
| **Compilation Mode** | ✅ All files automatically | ✅ All files automatically |
| **Compilers** | MSVC, Clang++, G++ | NVCC with smart version selection |
| **Output Directory** | `build/` | `build_gpu/` |
| **Requirements** | Any C++ compiler | CUDA Toolkit + MSVC |
| **Compatibility** | ✅ Universal | ✅ Smart VS2022 compatibility handling |

---

## **⚙️ Advanced Features**

### **Automatic Compiler Detection**
Both scripts automatically find the best available compiler:
- **CPU**: Tests GCC → MSVC → Clang in order
- **GPU**: Tests CUDA versions for VS2022 compatibility

### **MSVC Environment Setup**
GPU script automatically sets up MSVC environment for CUDA compilation:
- Adds MSVC to PATH
- Handles VS2022 compatibility issues
- Warns about problematic CUDA versions

### **Compatibility Handling**
GPU script intelligently handles CUDA version compatibility:
- **CUDA 12.4+**: Full VS2022 support
- **CUDA 12.1**: Uses compatibility flag with warnings
- **Automatic fallback**: Finds best available version

---

## **🛠️ Requirements**

### **CPU Compilation**
- **Windows 10/11**
- **One of:** Visual Studio (MSVC), Clang++, or MinGW-w64 (G++)

### **GPU Compilation**
- **Windows 10/11**
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.4+** (recommended) or 12.6, 12.0, 11.8
- **Visual Studio 2022** (required for nvcc compiler host)

---

## **🎯 Benefits**

✅ **No more hardcoded filenames**  
✅ **Works with any project structure**  
✅ **Automatic compilation of all files**  
✅ **Intelligent compiler detection**  
✅ **One script per compilation type**  
✅ **Comprehensive compatibility handling**  
✅ **Cross-compiler support**  
✅ **Automatic MSVC environment setup**  

**Bottom Line:** These smart batch scripts automatically compile all source files while handling compiler compatibility issues intelligently!
