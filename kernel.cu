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


__global__ void BackpropagateLayerKernel(REAL* output, REAL* error, REAL* weight, const REAL gain, int units, int prevUnits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < units) {
        REAL out = output[i];
        REAL err = 0;
        for (int j = 0; j < prevUnits; j++) {
            err += weight[j * units + i] * error[j];
        }
        error[i] = gain * out * (1 - out) * err;
    }
}

__global__ void AdjustWeightsKernel(REAL* output, REAL* error, REAL* weight, REAL* dWeight, const REAL eta, const REAL alpha, const int units, const int prevUnits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < units) {
        for (int j = 0; j < prevUnits; j++) {
            REAL out = output[j];
            REAL err = error[i];
            REAL dW = dWeight[i * prevUnits + j];
            weight[i * units + j] += eta * err * out + alpha * dW;
            dWeight[i * units + j] = eta * err * out;
        }
    }
}

void BackpropagateNetCUDA(NET *Net, int NUM_LAYERS)
{
    int blockSize = TILE_SIZE;

    for (int l = NUM_LAYERS - 1; l > 1; l--)
    {
        int units = Net->Layer[l]->Units;
        int prevUnits = Net->Layer[l - 1]->Units;

        int numBlocks = (prevUnits + blockSize - 1) / blockSize;

        int size = prevUnits * (units+1);
    
        REAL *d_weight, *d_dweight, *d_prevlayerOutput, *d_prevLayerError, *d_LayerError;
        cudaMalloc((REAL**)&d_weight, size * sizeof(REAL));
        cudaMalloc((REAL**)&d_dweight, size * sizeof(REAL));
        cudaMalloc((REAL**)&d_prevlayerOutput, prevUnits * sizeof(REAL));
        cudaMalloc((REAL**)&d_prevLayerError, prevUnits * sizeof(REAL));
        cudaMalloc((REAL**)&d_LayerError, units * sizeof(REAL));

        cudaMemcpy(d_prevLayerError, Net->Layer[l - 1]->Output, prevUnits * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prevlayerOutput, Net->Layer[l - 1]->Output, prevUnits * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, Net->Layer[l]->Weight, size * sizeof(REAL), cudaMemcpyHostToDevice);

        BackpropagateLayerKernel<<<numBlocks, blockSize>>>(d_prevlayerOutput, d_LayerError, d_weight, Net->Gain, units, prevUnits);
        AdjustWeightsKernel<<<numBlocks, blockSize>>>(d_prevlayerOutput, d_LayerError, d_weight, d_dweight, Net->Eta, Net->Alpha, units, prevUnits);
        // printf("jjj");
        // fflush(stdout);
        cudaMemcpy(Net->Layer[l]->Weight, d_weight, size * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(Net->Layer[l]->dWeight, d_dweight, size * sizeof(REAL), cudaMemcpyDeviceToHost);

        cudaFree(d_prevlayerOutput);
        cudaFree(d_prevLayerError);
        cudaFree(d_weight);
        cudaFree(d_dweight);
        cudaFree(d_LayerError);
    }
}
