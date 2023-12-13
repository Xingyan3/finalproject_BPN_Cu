#include "include.h"
#include "utils.cu"

void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
  const unsigned int BLOCK_SIZE = TILE_SIZE;
  const unsigned int numBlocks = (Upper->Units + BLOCK_SIZE - 1) / BLOCK_SIZE;

  mysgemm<<<BLOCK_SIZE, numBlocks>>>(1, Lower->Units, Upper->Units, Lower->Output, Upper->Weight, Upper->Output);

  sigmoid<<<BLOCK_SIZE, numBlocks>>>(Upper->Units, Upper->Output);
  cudaDeviceSynchronize();
}

void PropagateNet(NET* Net)
{
  INT l;
   
  for (l=0; l<NUM_LAYERS-1; l++) {
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
  }
}
