@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   SIMPLE CPU/GPU BUILD SYSTEM
echo ========================================

REM Check for src directory
if not exist "src" (
    echo ERROR: No src directory found!
    pause & exit /b 1
)

REM Create build directories
if not exist "build" mkdir build
if not exist "build_gpu" mkdir build_gpu

REM Check what files we have
set "HAS_CPP=0"
set "HAS_CU=0"
if exist "src\*.cpp" set "HAS_CPP=1"
if exist "src\*.cu" set "HAS_CU=1"

if %HAS_CPP% equ 0 if %HAS_CU% equ 0 (
    echo ERROR: No .cpp or .cu files found in src/
    pause & exit /b 1
)

echo Found in src/:
if %HAS_CPP% equ 1 (
    echo   C++ files:
    for %%f in (src\*.cpp) do echo     - %%~nxf
)
if %HAS_CU% equ 1 (
    echo   CUDA files:
    for %%f in (src\*.cu) do echo     - %%~nxf
)
echo.

REM ========================================
REM COMPILE .CPP FILES (CPU)
REM ========================================
if %HAS_CPP% equ 1 (
    echo [CPU] Compiling .cpp files...
    
    REM Find CPU compiler
    set "CPU_COMPILER="
    where g++ >nul 2>&1
    if !errorlevel! equ 0 (
        set "CPU_COMPILER=g++"
        echo Using: g++
        goto :compile_cpp
    )
    
    where cl >nul 2>&1
    if !errorlevel! equ 0 (
        set "CPU_COMPILER=cl"
        echo Using: cl (MSVC)
        goto :compile_cpp
    )
    
    echo ERROR: No C++ compiler found
    goto :skip_cpu
    
    :compile_cpp
    
    REM Compile each .cpp file
    for %%f in (src\*.cpp) do (
        echo   Building %%~nxf...
        if "!CPU_COMPILER!"=="g++" (
            g++ -std=c++17 -O3 -o "build\%%~nf.exe" "%%f"
        )
        if "!CPU_COMPILER!"=="cl" (
            cl /EHsc /O2 /Fe:"build\%%~nf.exe" "%%f"
        )
        
        if !errorlevel! equ 0 (
            echo   SUCCESS: build\%%~nf.exe
        ) else (
            echo   FAILED: %%~nxf
        )
    )
    echo.
)

:skip_cpu

REM ========================================
REM COMPILE .CU FILES (GPU)
REM ========================================
if %HAS_CU% equ 1 (
    echo [GPU] Compiling .cu files...
    
    REM Find CUDA compiler - prioritize CUDA 12.6 over PATH
    set "NVCC_PATH="
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" (
        set "NVCC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
        echo Using: CUDA 12.6 (compatible with MSVC)
    ) else (
        where nvcc >nul 2>&1
        if !errorlevel! neq 0 (
            echo ERROR: nvcc not found
            goto :skip_gpu
        )
        set "NVCC_PATH=nvcc"
        echo Using: nvcc from PATH (may have compatibility issues)
    )
    
    REM Setup MSVC for CUDA (simple version)
    set "MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
    if exist "!MSVC_PATH!\cl.exe" (
        set "PATH=!MSVC_PATH!;!PATH!"
        echo MSVC added to PATH
    )
    
    REM Compile each .cu file
    for %%f in (src\*.cu) do (
        echo   Building %%~nxf...
        "!NVCC_PATH!" -o "build_gpu\%%~nf.exe" "%%f"
        if !errorlevel! equ 0 (
            echo   SUCCESS: build_gpu\%%~nf.exe
        ) else (
            echo   FAILED: %%~nxf
        )
    )
    echo.
)

:skip_gpu

REM ========================================
REM SUMMARY
REM ========================================
echo ========================================
echo   BUILD COMPLETE
echo ========================================

if %HAS_CPP% equ 1 (
    echo CPU executables in build/:
    if exist "build\*.exe" (
        for %%f in (build\*.exe) do echo   - %%~nxf
    ) else (
        echo   - None (compilation failed)
    )
)

if %HAS_CU% equ 1 (
    echo GPU executables in build_gpu/:
    if exist "build_gpu\*.exe" (
        for %%f in (build_gpu\*.exe) do echo   - %%~nxf
    ) else (
        echo   - None (compilation failed)
    )
)

echo.
echo Run executables:
if exist "build\*.exe" echo   CPU: .\build\[filename].exe
if exist "build_gpu\*.exe" echo   GPU: .\build_gpu\[filename].exe
echo.
pause