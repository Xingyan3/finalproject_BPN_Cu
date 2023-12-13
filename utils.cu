#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const double *A, const double *B, double* C) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tx < n) {
        double res_C = 0.0;

        for (int i = 0; i < k; i++) {
            res_C += A[i] * B[i * n + tx];
        }
        C[tx] = res_C;
    }
}


__global__ void sigmoid(const double* vec_size, double* vec_res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vec_size) {
        vec_res[idx] = 1.0 / (1.0 + exp(-vec[idx]));
    }
}

__global__ void sigmoid_d(const double* vec_size, const double* gain, const double* vec_in, double* vec_res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vec_size) {
        double sigmoid_val = vec_in[idx];
        vec_res[idx] = gain * sigmoid_val * (1.0 - sigmoid_val) * vec_res[idx];
    }
}

__global__ void ComputeOutputErrorKernel(REAL* output, REAL* target, REAL* error, REAL gain, int units)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < units) {
        REAL out = output[i];
        REAL err = target[i] - out;
        error[i] = gain * out * (1 - out) * err; // 计算误差梯度
    }
}


// __global__ void mysgemm(int m, int n, int k, const double *A, const double *B, double* C) {

//     /********************************************************************
//      *
//      * Compute C = A x B
//      *   where A is a (m x k) matrix
//      *   where B is a (k x n) matrix
//      *   where C is a (m x n) matrix
//      *
//      * Use shared memory for tiling
//      *
//      ********************************************************************/

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     int row = by * TILE_SIZE + ty;
//     int col = bx * TILE_SIZE + tx;

//     float res_C = 0.0f;
    
//     __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
//     __shared__ float B_shared[TILE_SIZE][TILE_SIZE];


//     for (int t = 0; t < (k - 1) / TILE_SIZE + 1; t++)
//     {
//         if(row < m && t * TILE_SIZE + tx < k)
//         {
//             A_shared[ty][tx] = A[row * k + t * TILE_SIZE + tx];
//         }
//         else
//         {
//             A_shared[ty][tx] = 0.0;
//         }

//         if (col < n && t * TILE_SIZE + ty < k)
//         {
//             B_shared[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
//         }
//         else 
//         {
//             B_shared[ty][tx] = 0.0;
//         }

//         __syncthreads();

//         for (int i = 0; i < TILE_SIZE; i++)
//         {
//             res_C += A_shared[ty][i] * B_shared[i][tx];
//         }

//         __syncthreads();
//     }

//     if(row < m && col < n)
//     {
//         C[row * n + col] = res_C;
//     }
//     __syncthreads();
// }