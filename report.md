# CS217 Final Project report

**Team name**: VOID

**Group member**: Xingyan Zhou, Xinyu Zhang and Zhaorui Yang

**Project option**: Parallelizing serial C code with CUDA

# Project idea

The Backpropagation Network source code is a C language implementation of a neural network simulator, focusing on the backpropagation algorithm. It's primarily used for time-series forecasting, such as predicting the annual number of sunspots. The code includes the definition of the network structure, random number generation functions, and the core algorithms for learning and prediction.

We have thoroughly reviewed the source code and found that the backpropagation network consists of both propagation and backpropagation. As a result, we can parallelize the code in two directions. Additionally, we found several matrix and vector multiplications that we had implemented in our assignments, we decided to integrated them into the project. The platform we have chosen is Bender. It’s a powerful platform and we have gotten familiar with it, since it helps us a lot in the assignment. We still want to use it in the final project. Furthermore, we're consider to implement a timer within the source code. This feature will help to compare the performance before and after modifications.

# Our works
### Implementation details

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

  cudaMalloc((void**)&Net_d, sizeof(NET)*1);
  cudaDeviceSynchronize();

  Stop = FALSE;
  MinTestError = MAX_REAL;
  do {
    TrainNet(&Net, &Net_d, 10);
    TestNet(&Net, &Net_d);
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

**kernel.cu**
```C
#include <stdio.h>
#include "include.h"

#define TILE_SIZE 16

__global__ void PropagateLayerKernel(REAL* layerOutput, REAL* nextLayerOutput, REAL* weight, const REAL gain, const int units, const int nextUnits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < nextUnits) {
        REAL sum = 0;
        for (int j = 0; j < units; j++) {
            sum += weight[i * units + j] * layerOutput[j];
        }
        nextLayerOutput[i] = 1 / (1 + exp(-gain * sum));
    }
}


void PropagateNetCUDA(NET *Net, NET *Net_d, int NUM_LAYERS)
{
    int blockSize = TILE_SIZE;

    for (int l = 0; l < NUM_LAYERS - 1; l++)
    {
        int units = Net->Layer[l]->Units;
        int nextUnits = Net->Layer[l + 1]->Units;
        int numBlocks = (nextUnits + blockSize - 1) / blockSize;

        int size = nextUnits * units;
    
        REAL *d_weight, *d_layerOutput, *d_nextLayerOutput;
        cudaMalloc((REAL**)&d_weight, size * sizeof(REAL));
        cudaMalloc((REAL**)&d_layerOutput, units * sizeof(REAL));
        cudaMalloc((REAL**)&d_nextLayerOutput, nextUnits * sizeof(REAL));

        cudaMemcpy(d_layerOutput, Net->Layer[l]->Output, units * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, Net->Layer[l + 1]->Weight, size * sizeof(REAL), cudaMemcpyHostToDevice);

        PropagateLayerKernel<<<numBlocks, blockSize>>>(d_layerOutput, d_nextLayerOutput, d_weight, Net->Gain, units, nextUnits);

        cudaMemcpy(Net->Layer[l + 1]->Output, d_nextLayerOutput, nextUnits * sizeof(REAL), cudaMemcpyDeviceToHost);
            
        cudaFree(d_layerOutput);
        cudaFree(d_nextLayerOutput);
        cudaFree(d_weight);
    }
}
```

**supprot.cu**
```C
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, float *C, unsigned int m, unsigned int k,
  unsigned int n) {

  const float relativeTolerance = 1e-6;
  unsigned int count = 0;

  for(int row = 0; row < m; ++row) {
    for(int col = 0; col < n; ++col) {
      float sum = 0;
      for(unsigned int i = 0; i < k; ++i) {
        sum += A[row*k + i]*B[i*n + col];
      }
      count++;
      float relativeError = (sum - C[row*n + col])/sum;
      //printf("%f/%f ", sum, C[row*n + col]);
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
        printf("\nTEST FAILED %u\n\n",count);
        exit(1);
      }
    }
  }
  printf("TEST PASSED %u\n\n", count);

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
```

# Results


# Conclusion

Surprisingly, after our efforts, we found that there were no significant improvements in the overall running times. To investigate further, we focused on the running times for the propagation phase and discovered an unexpected result: the GPU running time was actually slower than the CPU running time. This led us to find the differences between the kernel, propagation, and serial running times.

For the kernel running time, which refers to the time taken for executing the core computation tasks on the GPU, was found to faster than propagation running time. This advantage was offset by the overheads associated with data transfer and memory management on the GPU, which are not present in CPU computations.

The propagation time showed a low performance. On the GPU, despite its parallel processing capabilities, the propagation was hindered by synchronization issues and the complex nature of data dependencies, leading to less efficient execution compared to the CPU.

The serial running time, which is the execution time on the CPU without parallelization, was consistently faster. This was attributed to the simpler architecture of the CPU that avoids the overheads present in GPU computations, and the nature of our tasks which might not have been ideally suited for parallelization.

These findings suggest that the optimization for GPU did not effectively leverage its parallel processing capabilities, and perhaps the tasks at hand were more suited for serial processing. Future efforts in optimization would need to carefully consider the nature of the tasks and the overheads involved in GPU computing.