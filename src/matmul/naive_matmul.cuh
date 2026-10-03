#ifndef NAIVE_MATMUL_CUH
#define NAIVE_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>

/*
    ==================================================
        NAIVE MATRIX MULTIPLICATION KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix

    The code below is for a naive general matrix multiplication kernel.
    This is the most basic implementation of matrix multiplication in CUDA.
    Each thread in the grid is assigned to compute one element of the output
    matrix C. 
*/

__global__ void naiveMatmulKernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M,
    int N, 
    int K
) {

    int row{blockIdx.y * blockDim.y + threadIdx.y};
    int col{blockIdx.x * blockDim.x + threadIdx.x};

    if((row >= M) || (col >= N)) 
        return;

    float sum{};

    for(int i{}; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    
    C[row * N + col] = sum;
}

template <int BLOCK_SIZE>
void naiveMatmulGPU(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K
) {
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(
        (N + dimBlock.x - 1) / dimBlock.x, 
        (M + dimBlock.y - 1) / dimBlock.y,
        1
    );

    naiveMatmulKernel<<<dimGrid, dimBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err{cudaGetLastError()};

    if(err != cudaSuccess)
        std::cout << "Naive Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';

}

#endif