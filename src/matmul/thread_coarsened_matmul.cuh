#ifndef THREAD_COARSENED_MATMUL_CUH
#define THREAD_COARSENED_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>

/*
    ==================================================
        THREAD COARSENED GEMM KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix
*/

__global__ void threadCoarsenedMatmulKernel(

) {

}

void threadCoarsenedMatmulGPU(
    const int blockSize
) {
    dim3 dimBlock(
        blockSize, 
        blockSize, 
        1
    );
    dim3 dimGrid(
        (M + dimBlock.x - 1)/dimBlock.x,
        (N + dimBlock.y - 1 )/dimBlock.y,
        1
    );

    threadCoarsenedMatmulKernel<<<dimGrid, dimBlock>>>(

    );

    err = cudaGetLastError();

    if(err != cudaSuccess)
        std::cout << "Thread Coarsened Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';

    err = cudaDeviceSynchronize();

    if(err != cudaSuccess)
        std::cout << "Thread Coarsened Matmul Kernel Exectution Error: " << cudaGetErrorString(err) << '\n'
}

#endif