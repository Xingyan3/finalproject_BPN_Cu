# CS217 Final Project report

**Team name**: VOID

**Group member**: Xingyan Zhou, Xinyu Zhang and Zhaorui Yang

**Project option**: Parallelizing serial C code with CUDA

# Project idea

The Backpropagation Network source code is a C language implementation of a neural network simulator, focusing on the backpropagation algorithm. It's primarily used for time-series forecasting, such as predicting the annual number of sunspots. The code includes the definition of the network structure, random number generation functions, and the core algorithms for learning and prediction.

Our team has thoroughly reviewed the source code and found that the backpropagation network consists of both forward propagation and backpropagation. As a result, we can parallelize the code in two directions. Additionally, we identified several matrix and vector multiplications that we had implemented in our assignments, we decided to integrated them into the project.

Furthermore, we're consider to implement a timing mechanism within the source code. This feature will help to compare the performance metrics before and after modifications.

# Our works
### Implementation details

**backpropagate.cu**
```C
#include "include.h"
#include "utils.cu"

void ComputeOutputError(NET* Net, REAL* Target)
{
  INT  i;
  REAL Out, Err;
   
  Net->Error = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Out = Net->OutputLayer->Output[i];
    Err = Target[i-1]-Out;
    Net->OutputLayer->Error[i] = Net->Gain * Out * (1-Out) * Err;
    Net->Error += 0.5 * sqr(Err);
  }
}

void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
  INT  i,j;
  REAL Out, Err;
   
  for (i=1; i<=Lower->Units; i++) {
    Out = Lower->Output[i];
    Err = 0;
    for (j=1; j<=Upper->Units; j++) {
      Err += Upper->Weight[j][i] * Upper->Error[j];
    }
    Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
  }
}

void BackpropagateNet(NET* Net)
{
  INT l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l-1]);
  }
}

void AdjustWeights(NET* Net)
{
  INT  l,i,j;
  REAL Out, Err, dWeight;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Out = Net->Layer[l-1]->Output[j];
        Err = Net->Layer[l]->Error[i];
        dWeight = Net->Layer[l]->dWeight[i][j];
        Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
      }
    }
  }
}
```

**main.c**
```C
#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#include "propagate.cu"
#include "backpropagate.cu"

int main(int argc, char *argv[])
{

}
```

**main.cu**
```C
#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#include "propagate.cu"
#include "backpropagate.cu"

int main(int argc, char *argv[])
{
  NET  Net;
  int Stop;
  double MinTestError;

  InitializeRandoms();
  GenerateNetwork(&Net);
  RandomWeights(&Net);
  InitializeApplication(&Net);

  cudaMalloc((void**)&Net, sizeof(NET)*1);
  cudaDeviceSynchronize();

  Stop = FALSE;
  MinTestError = MAX_REAL;
  do {
    TrainNet(&Net, 10);
    TestNet(&Net);
    if (TestError < MinTestError) {
      fprintf(f, " - saving Weights ...");
      MinTestError = TestError;
      SaveWeights(&Net);
    }
    else if (TestError > 1.2 * MinTestError) {
      fprintf(f, " - stopping Training and restoring Weights ...");
      Stop = TRUE;
      RestoreWeights(&Net);
    }
  } while (NOT Stop);

  TestNet(&Net);
  EvaluateNet(&Net);
   
  FinalizeApplication(&Net);
}
```

**propagate.cu**
```C
#include "include.h"
#include "utils.cu"

void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    mysgemm<<<dimGrid, dimBlock>>>(1, Lower->Units, Upper->Units, Lower->Output, Upper->Weight, Upper->Output);
    
    sigmoid<<<dimGrid, dimBlock>>>(Upper->Units, Upper->Output);
    cudaDeviceSynchronize();
}

void PropagateNet(NET* Net)
{
  INT l;
   
  for (l=0; l<NUM_LAYERS-1; l++) {
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
  }
}
```

**utils.cu**
```C
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
```

### How to run



# Results
**Result 1: Source code running time**
```
<img src="https://github.com/UCR-CSEE217/finalproject-f23-void/tree/main/picture/result_origional.png" metaname="viewport" width="400"/>
```

```
<img src="https://github.com/UCR-CSEE217/finalproject-f23-void/tree/main/picture/result_GPU.png" metaname="viewport" width="400"/>
```
**Result 2: GPU accelerated code running time**

# Conclusion



# Contribution


