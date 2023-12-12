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
