cmake_minimum_required(VERSION 3.24)
project(CUDAStuff CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(CUDAStuff
        main.cu
        kernels/matrix_multiply.cu
        kernels/naivenetworkcontraction.cu
        kernels/order3contraction.cu
        kernels/order4contraction.cu
        kernels/tiltedcontraction.cu
        kernels/tilted_matrixmult.cu)

target_include_directories(CUDAStuff PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(CUDAStuff PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
