#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const double *A, const double *B, double* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float res_C = 0.0f;
    
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];


    for (int t = 0; t < (k - 1) / TILE_SIZE + 1; t++)
    {
        if(row < m && t * TILE_SIZE + tx < k)
        {
            A_shared[ty][tx] = A[row * k + t * TILE_SIZE + tx];
        }
        else
        {
            A_shared[ty][tx] = 0.0;
        }

        if (col < n && t * TILE_SIZE + ty < k)
        {
            B_shared[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        }
        else 
        {
            B_shared[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
        {
            res_C += A_shared[ty][i] * B_shared[i][tx];
        }

        __syncthreads();
    }

    if(row < m && col < n)
    {
        C[row * n + col] = res_C;
    }
    __syncthreads();
}

__global__ void sigmoid(const double* vec_size, double* vec_res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vec_size) {
        vec_res[idx] = 1.0 / (1.0 + exp(-vec[idx]));
    }
}
