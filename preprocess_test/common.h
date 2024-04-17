#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <math.h>

// #include <helper_cuda.h>
// #include <helper_functions.h>

#include <cusparse.h>
#include <cublas_v2.h>

#include "omp.h"
#include "biio.h"

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#define MAT_VAL_TYPE double
#define MAT_PTR_TYPE int
#define MAT_IDX_TYPE int
#define MAT_MAP_TYPE unsigned short

#define WARP_SIZE 32
#define WARP_NUM_SPMV 4
#define WARP_NUM_SPGM 4

#define BSR_M 4
#define BSR_N 4
#define BSR_NNZ 16

#define SpMV_Repeat 1000
#define SpMV_Warm 100

#define BIN_NUM 8

#define setbit(x,y) x|=(1<<y)      //set the yth bit of x is 1 
#define clrbit(x,y) x&=~(1<<y)     //set the yth bit of x is 0 
#define getbit(x,y) ((x) >> (y)&1) //get the yth bit of x 