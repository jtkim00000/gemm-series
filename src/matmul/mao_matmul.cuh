#ifndef MAO_MATMUL_CUH
#define MAO_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>

/*
    ==================================================
        MEMORY ACCESS OPTIMIZED GEMM KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix
*/

__global__ void maoMatmulKernel(
    
) {

}

void maoMamultGPU(

) {

    dim3 dimBlock(
        blockSize,
        blockSize,
        1
    );

    dim3 dimGrid(
        (M + dimBlock.x - 1)/dimBlock.x,
        (N + dimBlock.y - 1)/dimBlock.y,
        1
    );

    maoMatmulKernel<<<dimGrid, dimBlock>>>(

    );

    cudaError_t err{cudaGetLastError()};

    if(err != cudaSuccess)
        std::cout << "MAO Kernel Launch Error: " << cudaGetErrorString(err) << '\n';
    
    err = cudaDeviceSynchronize();

    if(err != cudaSuccess)
        std::cout << "MAO Kernel Execution Error: " << cudaGetErrorString(err) << '\n'
}



#endif