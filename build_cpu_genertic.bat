@echo off
echo Checking for C++ files in src/ directory...

REM Check for src directory and .cpp files
if not exist "src" (
    echo Error: No src directory found! Run from project root.
    pause & exit /b 1
)

if not exist "src\*.cpp" (
    echo Error: No .cpp files found in src/ directory!
    pause & exit /b 1
)

echo Found .cpp files in src/:
dir /b src\*.cpp
echo.

REM Create build directory
if not exist "build" mkdir build

REM Find and use available compiler
echo Checking for compilers...
set COMPILER=
set COMPILE_CMD=

where g++ >nul 2>&1 && set COMPILER=GCC (g++) && set COMPILE_CMD=g++ -o "build\%%~nf.exe" "%%f"
if not defined COMPILER where cl >nul 2>&1 && set COMPILER=MSVC (cl) && set COMPILE_CMD=cl /Fe:"build\%%~nf.exe" "%%f"
if not defined COMPILER where clang++ >nul 2>&1 && set COMPILER=Clang (clang++) && set COMPILE_CMD=clang++ -o "build\%%~nf.exe" "%%f"

if not defined COMPILER (
    echo Error: No compiler found! Install GCC, MSVC, or Clang.
    pause & exit /b 1
)

echo Using %COMPILER%
echo Building all .cpp files...

REM Build each .cpp file
for %%f in (src\*.cpp) do (
    echo Building %%~nf.cpp...
    %COMPILE_CMD%
    if %errorlevel% equ 0 (echo Success: %%~nf.exe) else (echo Failed: %%~nf.cpp)
)

echo.
echo Build complete! Check the build/ directory for executables.
pause