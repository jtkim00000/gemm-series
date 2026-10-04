# GEMM Optimization Series in CUDA

This document is an indepth description and documentation of my attempt at optimizing General Matrix Multiplication (GEMM) kernels in CUDA. Some of the optimization techniques that I implemented include tiling, thread granularity coarsening, memory coalescing and corner turning, and bank-confict avoidance.

Each of the kernels was validated and their throughput was benchmarked against cuBLAS. In the end, the best performing kernel I was able to make was a combination of tiling, thread coarsening, and loop unrolling. Many of the optimization techniques that I applied had additional costs that they carried with them, causing the overall performance to drop. These optimizations will also be covered in this document. 

## Table of Contents
- [Overview](#overview)
- [Tiling](#tiling)
- [Thread Coarsening](@thread-coarsening)
- [Memory Coalescing](#memory-coalescing)
- [Other Optimizations](#other-optimizations)
- [Future Plans](#future-plans)
- 
## Overview

## Tiling

## Thread Coarsening

## Memory Coalescing

## Other Optimizations

## Future Plans
