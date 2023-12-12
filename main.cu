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