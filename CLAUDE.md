# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OSQP (Operator Splitting Quadratic Program) is a numerical optimization solver for quadratic programming problems with modular algebra backends supporting CPU (builtin), Intel MKL, and CUDA GPU acceleration.

## Build Commands

**Configure with specific algebra backend:**
```bash
# Using presets (recommended)
cmake --preset cuda              # CUDA backend
cmake --preset mkl               # Intel MKL backend  
cmake --preset builtin-double    # Builtin CPU double precision
cmake --preset builtin-float     # Builtin CPU single precision

# Manual configuration
cmake -B build -DOSQP_ALGEBRA_BACKEND=cuda
cmake -B build -DOSQP_ALGEBRA_BACKEND=mkl
cmake -B build -DOSQP_ALGEBRA_BACKEND=builtin
```

**Build:**
```bash
cmake --build build
```

**Run tests:**
```bash
ctest --test-dir build           # All tests
./build/out/osqp_tester          # Main test executable
```

**Debug builds:**
```bash
cmake --preset cuda-debug       # Debug + unit tests
cmake --preset builtin-double-debug  # Debug + ASAN
```

## Architecture

**Modular Algebra System:**
- `algebra/builtin/` - CPU implementation with built-in linear algebra
- `algebra/mkl/` - Intel MKL optimized CPU backend  
- `algebra/cuda/` - CUDA GPU acceleration with CSR sparse format
- `algebra/_common/` - Shared algebra interfaces and utilities

**CUDA Backend Specifics:**
- Uses CSR (Compressed Sparse Row) matrix format for GPU efficiency
- Leverages cuBLAS, cuSPARSE, and CUDA runtime
- Indirect linear system solver via PCG (Preconditioned Conjugate Gradient)
- Requires compute capability 5.2+ (float) or 6.0+ (double)

**Key Files:**
- `include/public/osqp.h` - Main solver API
- `algebra/cuda/algebra_types.h` - CUDA-specific type definitions
- `algebra/builtin/algebra_impl.h` - CPU type definitions
- `algebra/_common/kkt.h` - KKT system handling

**Test Data Generation:**
Tests use Python (NumPy/SciPy) to generate problem data, then C++ Catch2 framework for execution. Backend selection is handled at CMake configure time.

**Current Branch:** `b/cudss_interface` suggests active work on CUDSS (CUDA direct sparse solver) integration.