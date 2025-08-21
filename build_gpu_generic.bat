@echo off
setlocal enabledelayedexpansion
echo Checking for CUDA files in src/ directory...

REM Check for src directory and .cu files
if not exist "src" (
    echo Error: No src directory found! Run from project root.
    pause & exit /b 1
)

if not exist "src\*.cu" (
    echo Error: No .cu files found in src/ directory!
    pause & exit /b 1
)

echo Found .cu files in src/:
dir /b src\*.cu
echo.

REM Create build_gpu directory
if not exist "build_gpu" mkdir build_gpu

REM ============================================================================
REM MSVC ENVIRONMENT SETUP
REM ============================================================================
REM CUDA compilation on Windows requires Microsoft Visual C++ compiler (cl.exe)
REM This section adds MSVC to PATH so nvcc can find the required host compiler
echo Setting up MSVC environment...
set "MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
if exist "%MSVC_PATH%\cl.exe" (
    echo Found MSVC at: %MSVC_PATH%
    set "PATH=%MSVC_PATH%;%PATH%"
)

REM ============================================================================
REM CUDA COMPILER DETECTION & COMPATIBILITY HANDLING
REM ============================================================================
REM This section implements a smart CUDA version selection strategy:
REM 1. Prioritize CUDA 12.4+ (fully compatible with VS2022)
REM 2. Avoid CUDA 12.1 in PATH (known VS2022 compatibility issues)
REM 3. Fall back to other stable versions (12.6, 12.0, 11.8)
REM 4. Last resort: use CUDA 12.1 with compatibility flag
echo Checking for CUDA compiler...
set "NVCC_PATH="

REM PRIORITY 1: CUDA 12.4 (Most Compatible)
REM CUDA 12.4+ has full VS2022 support, no compatibility flags needed
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" (
    set "NVCC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
    set "CUDA_VERSION=12.4"
    goto :compile
)

REM PRIORITY 2: Check PATH for nvcc (Smart Compatibility Filtering)
REM This avoids CUDA 12.1 which has VS2022 compatibility issues
REM Error: "unsupported Microsoft Visual Studio version! Only versions between 2017-2022"
nvcc --version >nul 2>&1
if !errorlevel! equ 0 (
    REM Check if PATH version is the problematic CUDA 12.1
    nvcc --version | findstr "12.1" >nul
    if !errorlevel! neq 0 (
        REM PATH version is NOT 12.1, safe to use
        set "NVCC_PATH=nvcc"
        set "CUDA_VERSION=PATH"
        goto :compile
    ) else (
        REM PATH version IS 12.1 - warn about compatibility issues
        echo Warning: CUDA 12.1 detected in PATH - incompatible with VS2022
        echo This version requires -allow-unsupported-compiler flag and may have issues
    )
)

REM PRIORITY 3: Check Other Stable CUDA Versions (Hard-coded Paths)
REM These versions are known to work well with VS2022
REM Order: 12.6 (newer), 12.0 (stable), 11.8 (older but reliable)
for %%v in (12.6 12.0 11.8) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe" (
        set "NVCC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe"
        set "CUDA_VERSION=%%v"
        goto :compile
    )
)

REM PRIORITY 4: Last Resort - CUDA 12.1 with Compatibility Flag
REM Only use this if no other versions are available
REM The -allow-unsupported-compiler flag bypasses version checks but may cause issues
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe" (
    echo Warning: Using CUDA 12.1 which requires -allow-unsupported-compiler flag
    echo This may cause compilation failures or runtime issues with VS2022
    echo Consider upgrading to CUDA 12.4+ for better compatibility
    set "NVCC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe"
    set "CUDA_VERSION=12.1"
    goto :compile
)

REM ============================================================================
REM ERROR HANDLING - NO CUDA FOUND
REM ============================================================================
REM If we reach here, no CUDA installation was found
echo Error: No CUDA compiler found!
echo Please install CUDA Toolkit or add nvcc to your PATH.
echo Recommended: CUDA 12.4+ for full VS2022 compatibility
pause & exit /b 1

REM ============================================================================
REM COMPILATION SECTION
REM ============================================================================
:compile
echo Using CUDA !CUDA_VERSION!
echo Building all .cu files...

REM Build each .cu file with appropriate flags
for %%f in (src\*.cu) do (
    echo Building %%~nf.cu...
    
    REM Special handling for CUDA 12.1 compatibility issues
    if "!CUDA_VERSION!"=="12.1" (
        echo Using -allow-unsupported-compiler flag for CUDA 12.1
        REM This flag bypasses VS2022 version checks but may cause issues
        "!NVCC_PATH!" -allow-unsupported-compiler -o "build_gpu\%%~nf.exe" "%%f"
    ) else (
        REM Standard compilation for compatible CUDA versions
        "!NVCC_PATH!" -o "build_gpu\%%~nf.exe" "%%f"
    )
    
    REM Check compilation success
    if !errorlevel! equ 0 (
        echo Success: %%~nf.exe
    ) else (
        echo Failed: %%~nf.cu
    )
)

echo.
echo GPU build complete! Check the build_gpu/ directory for executables.
pause