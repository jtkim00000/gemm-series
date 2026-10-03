#ifndef TILED_MATMUL_CUH
#define TILED_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

/*
    ==================================================
        TILED MATRIX MULTIPLICATION KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix

    The code below is for a tiled matrix multiplcation kernel.
    The primary downside of the naive kernel is that it is 
    constantly reading the same elements of the A and B
    matricies from global memory. This makes the kernel "memory
    bound" since, according to Programming Massively Parallel
    Processors (Fourth Edition), the peak global memory bandwidth
    of an A100 is 1555 GB/S. We are doing 2 FLOPs for each 8 bytes
    of data read. Thus we are doing 0.25 FLOP/B, and 388.75 GFLOPS/s.
    However, the peak throughput of the A100 is 19.5 TFLOPS (19,500 GFLOPS).

    To achieve a higher throuhgput we must reduce the number of 
    global memory reads for each FLOP. To do this, we load elements
    of matrices A and B into shared memory, which has block scope.
    The threads can then read the elements from shared memory which
    has much lower access latency than global memory. However, the 
    primary drawback of shared memory is that it isn't large. Thus,
    we break up our matrix multiplication into smaller "tiles" which
    we then use to compute the output in parts.

    The code for this is not too different from the naive GEMM, however,
    you must sync the threads after loading elements into shared memory
    and after computing the dot products before loading the next tile 
    into shared memory. This thread syncing means we can't have an
    early return statement if the threads are accessing elements 
    outside the bounds of our output matrix. 

    Additionally, you have to intialize and load elements into shared 
    memory. Once again in this stage you must check boundary conditions
    so you are not putting elements into shared memory that are outside
    the bounds of matrices A and B. We set these locations to 0.0f so
    that the dot product loop does not include them in the calculation.

    I decided to use template functions for the TILE_WIDTH parameter
    which explains why all of these kernels and kernel launcher
    functions are contained in header files. The optimal TILE_WIDTH
    may depend on what GPU you are running this on. One can also 
    consider preprocessor macros and runtime paramters to achieve
    a similar result. I decided to go with template functions
    such that multiple sizes can be instantiated in the same
    run.

*/

template <int TILE_WIDTH>
__global__ void tiledMatmulKernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K
) {
    int tx{static_cast<int>(threadIdx.x)};
    int ty{static_cast<int>(threadIdx.y)};
    int bx{static_cast<int>(blockIdx.x)};
    int by{static_cast<int>(blockIdx.y)};

    int row{TILE_WIDTH * by + ty};
    int col{TILE_WIDTH * bx + tx};

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    float sum{};

    for(int tile_idx{}; tile_idx < ((K + TILE_WIDTH - 1) / TILE_WIDTH); ++tile_idx) {

        if((row < M) && ((tile_idx * TILE_WIDTH + tx) < K))
            Ads[ty][tx] = A[row * K + tile_idx * TILE_WIDTH + tx];
        else
            Ads[ty][tx] = 0.0f;

        if((col < N) && ((tile_idx * TILE_WIDTH + ty) < K))
            Bds[ty][tx] = B[(N * (tile_idx * TILE_WIDTH + ty)) + col];
        else
            Bds[ty][tx] = 0.0f;

        __syncthreads();

        for(int i{}; i < TILE_WIDTH; ++i) {
            sum += Ads[ty][i] * Bds[i][tx];
        }

        __syncthreads();

    }

    if((row < M) && (col < N))
        C[row * N + col] = sum;
}

template <int TILE_WIDTH>
void tiledMatmulGPU(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int N,
    int K
) {
    dim3 dimBlock(
        TILE_WIDTH,
        TILE_WIDTH,
        1
    );

    dim3 dimGrid(
        (N + dimBlock.x - 1) / dimBlock.x,
        (M + dimBlock.y - 1) / dimBlock.y,
        1
    ); 

    tiledMatmulKernel<TILE_WIDTH><<<dimGrid, dimBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err{cudaGetLastError()};

    if(err != cudaSuccess)
        std::cout << "Tiled Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';
}

#endif