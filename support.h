#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef int           BOOL;
typedef int           INT;
typedef double        REAL;

#define FALSE         0
#define TRUE          1
#define NOT           !
#define AND           &&
#define OR            ||

#define MIN_REAL      -HUGE_VAL
#define MAX_REAL      +HUGE_VAL
#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            0.1
#define HI            0.9
#define BIAS          1

#define sqr(x)        ((x)*(x))


typedef struct {                     /* A LAYER OF A NET:                     */
        INT           Units;         /* - number of units in this layer       */
        REAL*         Output;        /* - output of ith unit                  */
        REAL*         Error;         /* - error term of ith unit              */
        REAL**        Weight;        /* - connection weights to ith unit      */
        REAL**        WeightSave;    /* - saved weights for stopped training  */
        REAL**        dWeight;       /* - last weight deltas for momentum     */
} LAYER;

typedef struct {                     /* A NET:                                */
        LAYER**       Layer;         /* - layers of this net                  */
        LAYER*        InputLayer;    /* - input layer                         */
        LAYER*        OutputLayer;   /* - output layer                        */
        REAL          Alpha;         /* - momentum factor                     */
        REAL          Eta;           /* - learning rate                       */
        REAL          Gain;          /* - gain of sigmoid function            */
        REAL          Error;         /* - total net error                     */
} NET;
typedef int           BOOL;
typedef int           INT;
typedef double        REAL;

#define FALSE         0
#define TRUE          1
#define NOT           !
#define AND           &&
#define OR            ||

#define MIN_REAL      -HUGE_VAL
#define MAX_REAL      +HUGE_VAL
#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            0.1
#define HI            0.9
#define BIAS          1

#define sqr(x)        ((x)*(x))


typedef struct {                     /* A LAYER OF A NET:                     */
        INT           Units;         /* - number of units in this layer       */
        REAL*         Output;        /* - output of ith unit                  */
        REAL*         Error;         /* - error term of ith unit              */
        REAL**        Weight;        /* - connection weights to ith unit      */
        REAL**        WeightSave;    /* - saved weights for stopped training  */
        REAL**        dWeight;       /* - last weight deltas for momentum     */
} LAYER;

typedef struct {                     /* A NET:                                */
        LAYER**       Layer;         /* - layers of this net                  */
        LAYER*        InputLayer;    /* - input layer                         */
        LAYER*        OutputLayer;   /* - output layer                        */
        REAL          Alpha;         /* - momentum factor                     */
        REAL          Eta;           /* - learning rate                       */
        REAL          Gain;          /* - gain of sigmoid function            */
        REAL          Error;         /* - total net error                     */
} NET;

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void verify(float *A, float *B, float *C, unsigned int m, unsigned int k,
  unsigned int n);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
