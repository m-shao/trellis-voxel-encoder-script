@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDA_PATH=d:\cuda124"
set "CUDA_HOME=d:\cuda124"
set "PATH=d:\cuda124\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "MAX_JOBS=16"
