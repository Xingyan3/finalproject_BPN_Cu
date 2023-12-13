#include <stdio.h>
#include "include.h"

#define TILE_SIZE 16

__global__ void PropagateLayerKernel(REAL* layerOutput, REAL* nextLayerOutput, REAL* weight, const REAL gain, const int units, const int nextUnits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i <= nextUnits && i) {
        REAL sum = 0;
        for (int j = 0; j <= units; j++) {
            sum += weight[i * (units+1) + j] * layerOutput[j];
        }
        nextLayerOutput[i] = 1 / (1 + exp(-gain * sum));

    }
}


void PropagateNetCUDA(NET *Net, int NUM_LAYERS)
{
    int blockSize = TILE_SIZE;

    float total_time = 0;

    for (int l = 0; l < NUM_LAYERS - 1; l++)
    {
        int units = Net->Layer[l]->Units + 1;
        int nextUnits = Net->Layer[l + 1]->Units + 1;
        int numBlocks = (nextUnits + blockSize - 1) / blockSize;

        int size = nextUnits * units;
    
        REAL *d_weight, *d_layerOutput, *d_nextLayerOutput;
        cudaMalloc((REAL**)&d_weight, size * sizeof(REAL));
        cudaMalloc((REAL**)&d_layerOutput, units * sizeof(REAL));
        cudaMalloc((REAL**)&d_nextLayerOutput, nextUnits * sizeof(REAL));

        REAL* flattenedWeight = (REAL*)malloc(size * sizeof(REAL));
        for (int i = 0; i < nextUnits; i++) {
            for (int j = 0; j < units; j++) {
                flattenedWeight[i * units + j] = Net->Layer[l + 1]->Weight[i][j];
            }
        }

        cudaMemcpy(d_weight, flattenedWeight, size * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_layerOutput, Net->Layer[l]->Output, units * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nextLayerOutput, Net->Layer[l + 1]->Output, nextUnits * sizeof(REAL), cudaMemcpyHostToDevice);

        // startTime(&timer_kernel);
        PropagateLayerKernel<<<numBlocks, blockSize>>>(d_layerOutput, d_nextLayerOutput, d_weight, Net->Gain, Net->Layer[l]->Units , Net->Layer[l + 1]->Units);
        // stopTime(&timer_kernel); 
        // total_time += elapsedTime(timer_kernel);

        cudaMemcpy(Net->Layer[l + 1]->Output, d_nextLayerOutput, nextUnits * sizeof(REAL), cudaMemcpyDeviceToHost);
            
        cudaFree(d_layerOutput);
        cudaFree(d_nextLayerOutput);
        cudaFree(d_weight);
    }
    // printf("kernel running time: %f s\n", total_time);
}


__global__ void BackpropagateLayerKernel(REAL* prevoutput, REAL* error, REAL* preverror, REAL* weight, const REAL gain, int units, int prevUnits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= prevUnits && i) {
        REAL out = prevoutput[i];
        REAL err = 0;
        for (int j = 1; j <= units; j++) {
            err += weight[j * (prevUnits+1) + i] * error[j];
        }
        preverror[i] = gain * out * (1 - out) * err;
    }
}

void BackpropagateNetCUDA(NET *Net, int NUM_LAYERS)
{
    int blockSize = TILE_SIZE;

    for (int l = NUM_LAYERS - 1; l > 1; l--)
    {
        int units = Net->Layer[l]->Units + 1;
        int prevUnits = Net->Layer[l - 1]->Units + 1;

        int numBlocks = (prevUnits + blockSize - 1) / blockSize;

        int size = prevUnits * units;
    
        REAL *d_weight, *d_prevLayerError, *d_prevlayerOutput, *d_LayerError;
        cudaMalloc((REAL**)&d_weight, size * sizeof(REAL));
        cudaMalloc((REAL**)&d_prevlayerOutput, prevUnits * sizeof(REAL));
        cudaMalloc((REAL**)&d_LayerError, units * sizeof(REAL));
        cudaMalloc((REAL**)&d_prevLayerError, prevUnits * sizeof(REAL));

        REAL* flattenedWeight = (REAL*)malloc(size * sizeof(REAL));
        for (int i = 0; i < units; i++) {
            for (int j = 0; j < prevUnits; j++) {
                flattenedWeight[i * prevUnits + j] = Net->Layer[l]->Weight[i][j];
            }
        }

        cudaMemcpy(d_weight, flattenedWeight, size * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prevlayerOutput, Net->Layer[l - 1]->Output, prevUnits * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_LayerError, Net->Layer[l]->Error, units * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prevLayerError, Net->Layer[l - 1]->Error, prevUnits * sizeof(REAL), cudaMemcpyHostToDevice);
        
        BackpropagateLayerKernel<<<numBlocks, blockSize>>>(d_prevlayerOutput, d_LayerError, d_prevLayerError, d_weight, Net->Gain, Net->Layer[l]->Units, Net->Layer[l - 1]->Units);

        cudaMemcpy(Net->Layer[l - 1]->Error, d_prevLayerError, prevUnits * sizeof(REAL), cudaMemcpyDeviceToHost);

        cudaFree(d_prevlayerOutput);
        cudaFree(d_weight);
        cudaFree(d_prevLayerError);
        cudaFree(d_LayerError);
    }
}

__global__ void AdjustWeightsKernel(REAL* output, REAL* error, REAL* weight, REAL* dWeight, const REAL eta, const REAL alpha, const int units, const int prevUnits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= units && i) {
        for (int j = 0; j <= prevUnits; j++) {
            REAL out = output[j];
            REAL err = error[i];
            REAL dW = dWeight[i * (prevUnits+1) + j];
            weight[i * (prevUnits+1) + j] += eta * err * out + alpha * dW;
            dWeight[i * (prevUnits+1) + j] = eta * err * out;
        }
    }
}

void AdjustWeightsCUDA(NET *Net, int NUM_LAYERS)
{
    int blockSize = TILE_SIZE;

    for (int l = 1; l < NUM_LAYERS; l++)
    {
        int units = Net->Layer[l]->Units + 1;
        int prevUnits = Net->Layer[l - 1]->Units + 1;

        int numBlocks = (units + blockSize - 1) / blockSize;

        int size = prevUnits * units;
    
        REAL *d_weight, *d_dweight, *d_prevlayerOutput, *d_LayerError;
        cudaMalloc((REAL**)&d_weight, size * sizeof(REAL));
        cudaMalloc((REAL**)&d_dweight, size * sizeof(REAL));
        cudaMalloc((REAL**)&d_prevlayerOutput, prevUnits * sizeof(REAL));
        cudaMalloc((REAL**)&d_LayerError, units * sizeof(REAL));

        REAL* flatteneddWeight = (REAL*)malloc(size * sizeof(REAL));
        REAL* flattenedWeight = (REAL*)malloc(size * sizeof(REAL));
        for (int i = 0; i < units; i++) {
            for (int j = 0; j < prevUnits; j++) {
                flatteneddWeight[i * prevUnits + j] = Net->Layer[l]->dWeight[i][j];
                flattenedWeight[i * prevUnits + j] = Net->Layer[l]->Weight[i][j];
            }
        }

        cudaMemcpy(d_LayerError, Net->Layer[l]->Error, units * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prevlayerOutput, Net->Layer[l - 1]->Output, prevUnits * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, flattenedWeight, size * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dweight, flatteneddWeight, size * sizeof(REAL), cudaMemcpyHostToDevice);
        
        AdjustWeightsKernel<<<numBlocks, blockSize>>>(d_prevlayerOutput, d_LayerError, d_weight, d_dweight, Net->Eta, Net->Alpha, Net->Layer[l]->Units, Net->Layer[l - 1]->Units);
    
        cudaMemcpy(flattenedWeight, d_weight, size * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(flatteneddWeight, d_dweight, size * sizeof(REAL), cudaMemcpyDeviceToHost);

        for (int i = 0; i < units; i++) {
            for (int j = 0; j < prevUnits; j++) {
                Net->Layer[l]->Weight[i][j] = flattenedWeight[i * prevUnits + j];
                Net->Layer[l]->dWeight[i][j] = flatteneddWeight[i * prevUnits + j];
            }
        }

        cudaFree(d_prevlayerOutput);
        cudaFree(d_weight);
        cudaFree(d_dweight);
        cudaFree(d_LayerError);
    }
}
