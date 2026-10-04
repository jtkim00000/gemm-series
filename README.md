# GEMM Optimization Series in CUDA (Under Construction)

This document is an indepth description and documentation of my attempt at optimizing General Matrix Multiplication (GEMM) kernels in CUDA. Some of the optimization techniques that I implemented include tiling, thread granularity coarsening, memory coalescing and corner turning, and bank-confict avoidance.

Each of the kernels was validated and their throughput was benchmarked against cuBLAS. In the end, the best performing kernel I was able to make was a combination of tiling, thread coarsening, and loop unrolling. Many of the optimization techniques that I applied had additional costs that they carried with them, causing the overall performance to drop. These optimizations will also be covered in this document. 

## Table of Contents
- [Overview](#overview)
- [Tiling](#tiling)
- [Thread Coarsening](@thread-coarsening)
- [Memory Coalescing](#memory-coalescing)
- [Other Optimizations](#other-optimizations)
- [Future Plans](#future-plans)

## Overview
The following documentation fully described the progressive GEMM optimization project that I did in CUDA. Starting with a naive matrix multiplication kernel, I implemented various optimization techniques such as tiling, thread coarsening, memory coalescing, end bank conflict avoidance. Each of my kernels is thoroughly documented in the source code and I welcome readers to take a look at my implementations (and feel free to contact me if you find any errors or have questions).

Using a combination of techniques, I was able to achieve a throughput of 5.81 TFLOPS, which is a 5.95x speedup from the naive matrix multiplication kernel, and was 61.6% of the equivalent cuBLAS throughput on the same matrix size. 

For all the kernels, they implemented a GEMM where an M x K matrix A is multiplied by a K x N matrix B. The output matrix C is then a M x N matrix. The naive matrix multiplication kernel applied the finest granularity to threads, applying one thread per element in the output matrix C. Each of these threads then loops across a row of A and a column of B, multiplying corresponding elements and adding them to a collective sum. One should note that the accesses to matrix B are already coalesced since threads read from adjacent columns (multi-dimensional arrays are stored in row-major format). 

While many optimizations can be applied to the method that a kernel takes to multiply matrices, one must still consider effects of Streaming Multiprocessor (SM) occupancy.

One can find the naive kernel at `src/matmul/naive_matmul.cuh`

## Tiling
One of the primary downsides of naive matrix multiplication is that threads read from the same elements in global memory which has relatively high latency. If two threads are calculating elements that are in the same row of the output matrix C, then they would both need to read from the entire row of A. One way to combat this is to let all the threads in a block collectively load individual elements of matrix A and B, then store them into shared memory which had lower latency than global. Once the elements of A and B are stored into shared memory, each thread computed the dot products in the same way as the naive kernel.

One must note that shared memory is much smaller than global memory, and can often not hold all the data from A or B. Thus we break up our computation into smaller parts called tiles. In this implementation, a tile represent a square that corresponds to a subset of elements in our output matrix C. This tile would move across matrix A from left to right and move across matrix B from top to bottom. At each iteration the threads in the tile load a corresponding subset of matrix A and B into shared memory and then computes a portion of the dot product.

One must note that using shared memory and tiling has the added cost of thread synchronization. This is because all threads but finish storing data into shared memory before the partial dot product can be computed. Additionally, threads cannot move on to storing elements of the next tile into shared memory while other threads are still computing dot products. However, this tradeoff is still worth it in comparison to the naive matrix multiplication kernel due to the large overhead from repetitive reads from global memory.

## Thread Coarsening

## Memory Coalescing

## Other Optimizations

## Future Plans
