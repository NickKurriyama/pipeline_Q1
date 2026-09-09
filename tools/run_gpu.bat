@echo off
REM run_gpu.bat -- environment wrapper for gsplat (JIT CUDA build) on this Windows box.
REM Usage:  tools\run_gpu.bat pipeline3d\train_synthetic.py --device cuda --backend gsplat
REM Sets up: MSVC (cl.exe) env, target GPU arch, and a header workaround for the
REM Windows-SDK `small` macro that breaks CUDA 11.8's cub headers.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set TORCH_CUDA_ARCH_LIST=8.6
set NVCC_PREPEND_FLAGS=-include %~dp0win_macro_fix.h -allow-unsupported-compiler
set PYTHONIOENCODING=utf-8
python %*
