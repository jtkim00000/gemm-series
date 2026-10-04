#ifndef PEAK_MATMUL_CUH
#define PEAK_MATMUL_CUH

#include <iostream>
#include <cuda_runtime.h>

/*
    ==================================================
        PEAK GEMM KERNEL
    ==================================================

    A is an M x K matrix
    B is an K x N matrix

    This kernel computes the matrix multiplication A x B = C

    Thus, C is an M x N matrix

    The code below is for a GEMM kernel that attempts to maximize
    optimization using a combination of techniques from previous
    kernels and adds new, smaller optimizations. 

    Coalescing memory access had the added price of division and
    modulo operations, which actually caused a slowdown from the
    standard thread coarsening.
*/

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void peakMatmulKernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    const int M, 
    const int N,
    const int K
) {
    int tx{static_cast<int>(threadIdx.x)};
    int ty{static_cast<int>(threadIdx.y)};
    int bx{static_cast<int>(blockIdx.x)};
    int by{static_cast<int>(blockIdx.y)};

    constexpr int TILE_WIDTH{BLOCK_SIZE * COARSE_FACTOR};

    int row_start{TILE_WIDTH * by + ty * COARSE_FACTOR};
    int col_start{TILE_WIDTH * bx + tx * COARSE_FACTOR};

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    float sums[COARSE_FACTOR*COARSE_FACTOR];  // row-major output sums

    for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
        for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {
            sums[row_idx * COARSE_FACTOR + col_idx] = 0.0f;
        }
    }
    
    int row_current{};
    int col_current{};

    #pragma unroll
    for(int tile_idx{}; tile_idx < ((K + TILE_WIDTH - 1) / TILE_WIDTH); ++tile_idx) {

        #pragma unroll
        for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
            #pragma unroll
            for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {

                row_current = row_start + row_idx;
                col_current = col_start + col_idx;

                if((row_current < M) && ((tile_idx * TILE_WIDTH + tx * COARSE_FACTOR + col_idx) < K))
                    Ads[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = A[row_current * K + tile_idx * TILE_WIDTH + tx * COARSE_FACTOR + col_idx];
                else
                    Ads[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = 0.0f;

                if((col_current < N) && ((tile_idx * TILE_WIDTH + ty * COARSE_FACTOR + row_idx) < K))
                    Bds[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = B[(N * (tile_idx * TILE_WIDTH + ty * COARSE_FACTOR + row_idx)) + col_current];
                else
                    Bds[ty * COARSE_FACTOR + row_idx][tx * COARSE_FACTOR + col_idx] = 0.0f;
            }
        }

        __syncthreads();

        int sum_row{};
        int sum_col{};
        int sum_elem{};

        #pragma unroll
        for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
            #pragma unroll
            for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {
                sum_row = ty * COARSE_FACTOR + row_idx;
                sum_col = tx * COARSE_FACTOR + col_idx;
                sum_elem = row_idx * COARSE_FACTOR + col_idx;
                #pragma unroll
                for(int i{}; i < TILE_WIDTH; ++i) {
                    sums[sum_elem] += Ads[sum_row][i] * Bds[i][sum_col];
                }
            }
        }

        __syncthreads();

    }

    #pragma unroll
    for(int row_idx{}; row_idx < COARSE_FACTOR; ++row_idx) {
        #pragma unroll
        for(int col_idx{}; col_idx < COARSE_FACTOR; ++col_idx) {
            
            row_current = row_start + row_idx;
            col_current = col_start + col_idx;

            if((row_current < M) && (col_current < N))
                C[row_current * N + col_current] = sums[row_idx * COARSE_FACTOR + col_idx];

        }
    }
}

template <int BLOCK_SIZE, int COARSE_FACTOR>
void peakMatmulGPU(
    const float* A, 
    const float* B, 
    float* C, 
    const int M, 
    const int N,
    const int K
) {

    constexpr int TILE_WIDTH = BLOCK_SIZE * COARSE_FACTOR;

    dim3 dimBlock(
        BLOCK_SIZE, 
        BLOCK_SIZE, 
        1
    );
    dim3 dimGrid(
        (N + TILE_WIDTH - 1)/TILE_WIDTH,
        (M + TILE_WIDTH - 1)/TILE_WIDTH,
        1
    );

    peakMatmulKernel<BLOCK_SIZE, COARSE_FACTOR><<<dimGrid, dimBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaError_t err{cudaGetLastError()};

    if(err != cudaSuccess)
        std::cout << "Thread Coarsened Matmul Kernel Launch Error: " << cudaGetErrorString(err) << '\n';
}

#endif