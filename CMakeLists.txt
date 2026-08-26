cmake_minimum_required(VERSION 3.24)
project(CUDAStuff CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(CUDAStuff main.cu)

set_target_properties(CUDAStuff PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
