#include "include.h"
#include "utils.cu"

void propagate(NET* Net, LAYER* Lower, LAYER* Upper)
{
    const unsigned int BLOCK_SIZE = TILE_SIZE;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    mysgemm<<<dimGrid, dimBlock>>>()
}
