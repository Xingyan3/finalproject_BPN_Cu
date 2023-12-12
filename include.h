/******************************************************************************
                            D E C L A R A T I O N S
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef int BOOL;
typedef int INT;
typedef double REAL;

#define FALSE 0
#define TRUE 1
#define NOT !
#define AND &&
#define OR ||

#define MIN_REAL -HUGE_VAL
#define MAX_REAL +HUGE_VAL
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define LO 0.1
#define HI 0.9
#define BIAS 1

#define sqr(x) ((x) * (x))

typedef struct
{                      /* A LAYER OF A NET:                     */
    INT Units;         /* - number of units in this layer       */
    REAL *Output;      /* - output of ith unit                  */
    REAL *Error;       /* - error term of ith unit              */
    REAL **Weight;     /* - connection weights to ith unit      */
    REAL **WeightSave; /* - saved weights for stopped training  */
    REAL **dWeight;    /* - last weight deltas for momentum     */
} LAYER;

typedef struct
{                       /* A NET:                                */
    LAYER **Layer;      /* - layers of this net                  */
    LAYER *InputLayer;  /* - input layer                         */
    LAYER *OutputLayer; /* - output layer                        */
    REAL Alpha;         /* - momentum factor                     */
    REAL Eta;           /* - learning rate                       */
    REAL Gain;          /* - gain of sigmoid function            */
    REAL Error;         /* - total net error                     */
} NET;
