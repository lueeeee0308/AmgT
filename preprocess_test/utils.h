#include "common.h"

typedef struct
{
    int row;
    int col;
    int nnz;
    int blc_row;
    int blc_col;
    int blc_num;
    MAT_PTR_TYPE *blcPtr;
    MAT_IDX_TYPE *blcIdx;
    MAT_VAL_TYPE *blcVal;
    MAT_MAP_TYPE *blcMap;
} bsrMAT;

typedef struct
{
    int row;
    int col;
    int nnz;
    int isSym;
    MAT_PTR_TYPE *RowPtr;
    MAT_IDX_TYPE *ColIdx;
    MAT_VAL_TYPE *Value;
} csrMAT;

#define Bitmap_col_x_row(col, row) (((col) * 0xf) & ((row) * 0x1111))


__device__ __forceinline__ void mma_m8n8k4(MAT_VAL_TYPE *acc, MAT_VAL_TYPE &frag_a, MAT_VAL_TYPE &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]):
        "d"(frag_a), "d"(frag_b)
    );
}

__device__ __forceinline__ void mma_m8n8k4_fp16(half *acc, uint32_t *A, uint32_t *B)
{
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}

__device__ __forceinline__ void mma_m8n8k4_fp16_rr(half *acc, uint32_t *A, uint32_t *B)
{
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}

__device__ __forceinline__ void mma_m8n8k4_fp16(float *acc, uint32_t *A, uint32_t *B)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
        " { %0, %1, %2, %3, %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %10, %11 }, "
        " { %0, %1, %2, %3, %4, %5, %6, %7 };"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}

__device__ __host__
int BinarySearch2(int *arr, int left, int right, int target) {
	int low = left;
	int high = right;
	int mid = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (target < arr[mid]) high = mid - 1;
		else if (target > arr[mid]) low = mid + 1;
		else return mid;
	}
	return -1;
}

__device__ __host__
int BinarySearch3(unsigned int *arr, int left, int right, unsigned int target) {
	int low = left;
	int high = right;
	int mid = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (target < arr[mid]) high = mid - 1;
		else if (target > arr[mid]) low = mid + 1;
		else return mid;
	}
	return -1;
}

int BinarySearch(int *arr, int len, int target) {
	int low = 0;
	int high = len;
	int mid = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (target < arr[mid]) high = mid - 1;
		else if (target > arr[mid]) low = mid + 1;
		else return mid;
	}
	return -1;
}

void swap_key(int *a , int *b)
{
    int tmp = *a;
    if (a != NULL && b != NULL)
    {
        *a = *b;
    }
    if (b != NULL)
    {
        *b = tmp;
    }
}

void swap_val(MAT_VAL_TYPE *a , MAT_VAL_TYPE *b)
{
    MAT_VAL_TYPE tmp = *a;
    if (a != NULL && b != NULL)
    {
        *a = *b;
    }
    if (b != NULL)
    {
        *b = tmp;
    }
}

int partition_key_val_pair1(int *key, MAT_VAL_TYPE *val, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    if (val != NULL && key != NULL)
    {
        swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
        swap_val(&val[pivot_index], &val[pivot_index + (length - 1)]);
    }

    for(; i < length; i++)
    {
        if ( key != NULL)
        {
            if(key[pivot_index+i] < pivot)
            {
                swap_key(&key[pivot_index+i], &key[small_length]);
                if (val != NULL)
                {
                    swap_val(&val[pivot_index+i], &val[small_length]);
                }
                small_length++;
            }
        }
    }

    if (key != NULL)
    {
        swap_key(&key[pivot_index + length - 1], &key[small_length]);
    }
    if (val != NULL)
    {
        swap_val(&val[pivot_index + length - 1], &val[small_length]);
    }

    return small_length;
}

void quick_sort_key_val_pair1(int *key, MAT_VAL_TYPE *val, int length)
{
    if(length == 0 || length == 1)
    {
        return;
    }

    int small_length = partition_key_val_pair1(key, val, length, 0);
    quick_sort_key_val_pair1(key, val, small_length);
    if (val != NULL && key != NULL)
    {
        quick_sort_key_val_pair1(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
    }
}

__device__ __forceinline__ MAT_VAL_TYPE warpReduceSum(MAT_VAL_TYPE sum){
    sum += __shfl_down_sync(0xffffffff,sum,16);
    sum += __shfl_down_sync(0xffffffff,sum,8);
    sum += __shfl_down_sync(0xffffffff,sum,4);
    sum += __shfl_down_sync(0xffffffff,sum,2);
    sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}

__device__ __forceinline__ int warpReduceSum(int sum){
    sum += __shfl_down_sync(0xffffffff,sum,16);
    sum += __shfl_down_sync(0xffffffff,sum,8);
    sum += __shfl_down_sync(0xffffffff,sum,4);
    sum += __shfl_down_sync(0xffffffff,sum,2);
    sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}

__device__ __host__
unsigned short bitMatrixMul(unsigned short bitMap_1, unsigned short bitMap_2)
{
    // bitMap_1: input bitmap 1
    // bitMap_2: input bitmap 2
    // return: bitmap 3

    // get every col of bitMap_1 and every row of bitMap_2
    unsigned short res = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        res |= Bitmap_col_x_row(bitMap_1 & 0x1111, bitMap_2 & 0xf);
        bitMap_1 >>= 1;
        bitMap_2 >>= 4;
    }
    return res;
}

void init_vector(MAT_VAL_TYPE *arr, int len)
{
    for (int i = 0; i < len; i ++)
    {
        arr[i] = 1.0;
    }
}

void CSR2BSR(csrMAT *csrmat, bsrMAT *bsrmat)
{
    struct timeval t1, t2;

    bsrmat->row = csrmat->row;
    bsrmat->col = csrmat->col;

    bsrmat->blc_row = (bsrmat->row + BSR_M - 1) / BSR_M;
    bsrmat->blc_col = (bsrmat->col + BSR_N - 1) / BSR_N;

    int bsr_m = bsrmat->blc_row;
    int bsr_n = bsrmat->blc_col;

    int m = bsrmat->row;

    bsrmat->blcPtr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
    memset(bsrmat->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));

    int len = (bsr_n + 31) / 32;
    gettimeofday(&t1, NULL);
    #pragma omp parallel for
    for (int i = 0; i < bsr_m; i ++)
    {
        int cur_row_block_num = 0;
        unsigned int *mask = (unsigned int *)malloc(sizeof(unsigned int) * len);
        memset(mask, 0, sizeof(unsigned int) * len);
        for (int mi = 0; mi < BSR_M; mi ++)
        {
            int cur_rowid = i * BSR_M + mi;
            if (cur_rowid >= m ) break;
            for (int j = csrmat->RowPtr[cur_rowid]; j < csrmat->RowPtr[cur_rowid + 1]; j ++)
            {gettimeofday(&t1, NULL);
                int cur_colid = csrmat->ColIdx[j];
                int key = cur_colid / BSR_N;
                setbit(mask[key / 32], key % 32);
            }
        }
        for (int j = 0; j < bsr_n; j ++)
        {
            if (getbit(mask[j/32], j%32) == 1)
            {
                cur_row_block_num++;
            }
        }
        bsrmat->blcPtr[i] = cur_row_block_num;
        free(mask);  
    }
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //printf("time === %.6lf\n",time);

    // printf("begin:\n");
    // for(int i=0;i<bsr_m;i++){
    //     printf("%d ",bsrmat->blcPtr[i]);
    // }
    // printf("\n");

    gettimeofday(&t1, NULL);
    exclusive_scan(bsrmat->blcPtr, bsr_m + 1);
    gettimeofday(&t2, NULL);
    time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //printf("time === %.6lf\n",time);

    bsrmat->blc_num = bsrmat->blcPtr[bsr_m];
    bsrmat->nnz = bsrmat->blc_num * BSR_M * BSR_N;

    int bsr_num = bsrmat->blc_num;
    int bsr_nnz = bsrmat->nnz;

    bsrmat->blcIdx = (MAT_IDX_TYPE *)malloc(sizeof(MAT_IDX_TYPE) * bsr_num);
    bsrmat->blcVal = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * bsr_nnz);
    bsrmat->blcMap = (MAT_MAP_TYPE *)malloc(sizeof(MAT_MAP_TYPE) * bsr_num);
    memset(bsrmat->blcVal, 0, sizeof(MAT_VAL_TYPE) * bsr_nnz);
    memset(bsrmat->blcMap, 0, sizeof(MAT_MAP_TYPE) * bsr_num);
    gettimeofday(&t1, NULL);
    #pragma omp parallel for
    for (int i = 0; i < bsr_m; i ++)
    {
        unsigned int *mask = (unsigned int *)malloc(sizeof(unsigned int) * len);
        memset(mask, 0, sizeof(unsigned int) * len);
        for (int mi = 0; mi < BSR_M; mi ++)
        {
            int cur_rowid = i * BSR_M + mi;
            if (cur_rowid >= m ) break;
            for (int j = csrmat->RowPtr[cur_rowid]; j < csrmat->RowPtr[cur_rowid + 1]; j ++)
            {
                int cur_colid = csrmat->ColIdx[j];
                int key = cur_colid / BSR_N;
                setbit(mask[key / 32], key % 32);
            }
        }
        // set colidx
        int cur_blockid = bsrmat->blcPtr[i];
        for (int j = 0; j < bsr_n; j ++)
        {
            if (getbit(mask[j/32], j%32) == 1)
            {
                bsrmat->blcIdx[cur_blockid++] = j;
            }
        }
        // load value
        for (int mi = 0; mi < BSR_M; mi ++)
        {
            int cur_rowid = i * BSR_M + mi;
            if (cur_rowid >= m ) break;
            for (int j = csrmat->RowPtr[cur_rowid]; j < csrmat->RowPtr[cur_rowid + 1]; j ++)
            {
                int cur_colid = csrmat->ColIdx[j];
                int key = cur_colid / BSR_N;
                int offset_cid = BinarySearch(bsrmat->blcIdx + bsrmat->blcPtr[i], bsrmat->blcPtr[i + 1] - bsrmat->blcPtr[i], key);
                int offset_blc = bsrmat->blcPtr[i] + offset_cid;
                int offset_idx = mi * BSR_M + cur_colid % BSR_N;
                int offset_val = offset_blc * (BSR_M * BSR_N) + offset_idx;
                bsrmat->blcVal[offset_val] = csrmat->Value[j];
                setbit(bsrmat->blcMap[offset_blc], offset_idx);
                // bsrmat->blcMap[offset_blc]
            }
        }
        free(mask);        
    }
    gettimeofday(&t2, NULL);
    time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //printf("time === %.6lf\n",time);



    // printf("begin:\n");
    // for (int i = 0; i < csrmat->row + 1; i++)
    // {
    //     printf("%d ", csrmat->RowPtr[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     printf("%d ", csrmat->ColIdx[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < bsr_m + 1; i++)
    // {
    //     printf("%d ", bsrmat->blcPtr[i]);
    // }
    // printf("\nidx\n");

    // for (int i = 0; i < bsr_num; i++)
    // {
    //     printf("%d ", bsrmat->blcIdx[i]);
    // }
    // printf("\nval\n");

    // for (int i = 0; i < bsr_num; i++)
    // {
    //     double now = 0.0;
    //     for(int j=0;j<16;j++){
    //         now+=bsrmat->blcVal[i*16+j];
    //     }
    //     printf("%d %lf\n",i, now);
    // }
    // printf("\nmap\n");

    // for (int i = 0; i < bsr_num; i++)
    // {
    //     printf("%d ", bsrmat->blcMap[i]);
    // }
    // printf("\n");
}

void blcMat_cpy_H2D(bsrMAT *d_mat, bsrMAT *h_mat)
{
    d_mat->row = h_mat->row;
    d_mat->col = h_mat->col;
    d_mat->nnz = h_mat->nnz;
    d_mat->blc_row = h_mat->blc_row;
    d_mat->blc_col = h_mat->blc_col;
    d_mat->blc_num = h_mat->blc_num;
    
    cudaMalloc((void **)&(d_mat->blcPtr), sizeof(MAT_PTR_TYPE) * (d_mat->blc_row + 1));
    cudaMalloc((void **)&(d_mat->blcIdx), sizeof(MAT_IDX_TYPE) * d_mat->blc_num);
    cudaMalloc((void **)&(d_mat->blcVal), sizeof(MAT_VAL_TYPE) * d_mat->nnz);
    cudaMalloc((void **)&(d_mat->blcMap), sizeof(MAT_MAP_TYPE) * d_mat->blc_num);

    cudaMemcpy(d_mat->blcPtr, h_mat->blcPtr, sizeof(MAT_PTR_TYPE) * (d_mat->blc_row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->blcIdx, h_mat->blcIdx, sizeof(MAT_IDX_TYPE) * d_mat->blc_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->blcVal, h_mat->blcVal, sizeof(MAT_VAL_TYPE) * d_mat->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->blcMap, h_mat->blcMap, sizeof(MAT_MAP_TYPE) * d_mat->blc_num, cudaMemcpyHostToDevice);
}

void readMatrix(csrMAT *mat, char *filename)
{
    read_Dmatrix_32(&(mat->row), &(mat->col), &(mat->nnz), 
                    &(mat->RowPtr), &(mat->ColIdx), &(mat->Value),
                    &(mat->isSym), filename);
    init_vector(mat->Value, mat->nnz);
    for (int i = 0; i < mat->row; i ++)
    {
        int len = mat->RowPtr[i + 1] - mat->RowPtr[i];
        quick_sort_key_val_pair1((mat->ColIdx) + mat->RowPtr[i], (mat->Value) + mat->RowPtr[i], len);
    }
}

void cusparse_spmv_all(MAT_VAL_TYPE *cu_ValA, MAT_PTR_TYPE *cu_RowPtrA, int *cu_ColIdxA, 
                       MAT_VAL_TYPE *cu_ValX, MAT_VAL_TYPE *cu_ValY, int rowA, int colA, MAT_PTR_TYPE nnzA,
                       MAT_VAL_TYPE alpha, MAT_VAL_TYPE beta, double *cu_time, double *best_time)
{
    struct timeval t1, t2;

    MAT_VAL_TYPE *dA_val, *dX, *dY;
    int *dA_cid;
    MAT_PTR_TYPE *dA_rpt;

    cudaMalloc((void **)&dA_val, sizeof(MAT_VAL_TYPE) * nnzA);
    cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA);
    cudaMalloc((void **)&dA_rpt, sizeof(MAT_PTR_TYPE) * (rowA + 1));
    cudaMalloc((void **)&dX, sizeof(MAT_VAL_TYPE) * colA);
    cudaMalloc((void **)&dY, sizeof(MAT_VAL_TYPE) * rowA);

    cudaMemcpy(dA_val, cu_ValA, sizeof(MAT_VAL_TYPE) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(MAT_PTR_TYPE) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, cu_ValX, sizeof(MAT_VAL_TYPE) * colA, cudaMemcpyHostToDevice);
    cudaMemset(dY, 0.0, sizeof(MAT_VAL_TYPE) * rowA);

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer = NULL;
    size_t               bufferSize = 0;

    gettimeofday(&t1, NULL);
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, colA, dX, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, rowA, dY, CUDA_R_64F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double cusparse_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    for (int i = 0; i < SpMV_Warm; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();

    double once_time = 0;
    
    for (int i = 0; i < SpMV_Repeat; ++i)
    {
        gettimeofday(&t1, NULL);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        once_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (once_time < *best_time) *best_time = once_time;
        *cu_time += once_time;
    }
    *cu_time /= SpMV_Repeat;
    
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaMemcpy(cu_ValY, dY, sizeof(MAT_VAL_TYPE) * rowA, cudaMemcpyDeviceToHost);

    cudaFree(dA_val);
    cudaFree(dA_cid);
    cudaFree(dA_rpt);
    cudaFree(dX);
    cudaFree(dY);
}

void cusparse_spgemm(double *time_cusp,
                     int A_num_rows, int A_num_cols, int A_nnz, MAT_PTR_TYPE *hA_csrOffsets, int *hA_columns, MAT_VAL_TYPE *hA_values,
                     int B_num_rows, int B_num_cols, int B_nnz, MAT_PTR_TYPE *hB_csrOffsets, int *hB_columns, MAT_VAL_TYPE *hB_values,
                     int C_num_rows, int C_num_cols, int *C_nnz, MAT_PTR_TYPE **hC_csrOffsets, int **hC_columns, MAT_VAL_TYPE **hC_values)
{
    struct timeval t1, t2;
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA, matB, matC;

    // Device memory management: Allocate and copy A, B
    int   *dA_csrOffsets, *dA_columns,
          *dB_csrOffsets, *dB_columns,
          *dC_csrOffsets, *dC_columns;
    MAT_MAP_TYPE *dA_values, *dB_values, *dC_values;

    // allocate A
    cudaMalloc((void**) &dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int));
    cudaMalloc((void**) &dA_values,  A_nnz * sizeof(MAT_VAL_TYPE));
    // allocate B
    cudaMalloc((void**) &dB_csrOffsets, (B_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int));
    cudaMalloc((void**) &dB_values,  B_nnz * sizeof(MAT_VAL_TYPE));

    // copy A
    cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    // copy B
    cudaMemcpy(dB_csrOffsets, hB_csrOffsets, (B_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_values, hB_values, B_nnz * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      dB_csrOffsets, dB_columns, dB_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                    //   dC_csrOffsets, NULL, NULL,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);
    double              alpha       = 1.0f;
    double              beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_64F;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    // allocate C offsets
    cudaMalloc((void**) &dC_csrOffsets, (A_num_rows + 1) * sizeof(int));

    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                &alpha, matA, matB, &beta, matC,
                                computeType, CUSPARSE_SPGEMM_DEFAULT,
                                spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void**) &dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, dBuffer1);
    
    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                            spgemmDesc, &bufferSize2, NULL);

    cudaMalloc((void**) &dBuffer2, bufferSize2);

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                            spgemmDesc, &bufferSize2, dBuffer2);
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
    // allocate matrix C
    cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int));
    cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(MAT_VAL_TYPE));

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
    
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    *time_cusp = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    printf("cusparse nnz %ld\n", C_nnz1);

    *C_nnz = C_nnz1;
    *hC_csrOffsets = (int *)malloc(sizeof(int) * (C_num_rows + 1));
    *hC_columns = (int *)malloc(C_nnz1 * sizeof(int));
    *hC_values = (MAT_VAL_TYPE *)malloc(C_nnz1 * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(*hC_csrOffsets, dC_csrOffsets, (C_num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*hC_columns, dC_columns, C_nnz1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*hC_values, dC_values, C_nnz1 * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // destroy matrix/vector descriptors
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);   

    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dB_csrOffsets);
    cudaFree(dB_columns);
    cudaFree(dB_values);
    cudaFree(dC_csrOffsets);
    cudaFree(dC_columns);
    cudaFree(dC_values);
}

long long int compute_nnzCub(MAT_PTR_TYPE nnzA, MAT_IDX_TYPE *cidA, MAT_PTR_TYPE *rptB)
{
    long long int nnzCub = 0;
    for (int i = 0; i < nnzA; i ++)
    {
        int colidx_A = cidA[i];
        nnzCub += rptB[colidx_A + 1] - rptB[colidx_A];
    }

    return nnzCub;
}

void gemm(MAT_VAL_TYPE *A, MAT_VAL_TYPE *B, MAT_VAL_TYPE *C, int m, int n, int k)
{
    for (int i = 0; i < m; i ++)
    {
        for (int j = 0; j < n; j ++)
        {
            MAT_VAL_TYPE sum = 0;
            for (int l = 0; l < k; l ++)
            {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] += sum;
        }
    }
}

void ref_spgemm(csrMAT *matA, csrMAT *matB, csrMAT *matC)
{
    int rowA = matA->row;
    int rowC = matA->row;

    MAT_PTR_TYPE *csrPtrA = matA->RowPtr;
    MAT_IDX_TYPE *csrCidA = matA->ColIdx;
    MAT_VAL_TYPE *csrValA = matA->Value;

    MAT_PTR_TYPE *csrPtrB = matB->RowPtr;
    MAT_IDX_TYPE *csrCidB = matB->ColIdx;
    MAT_VAL_TYPE *csrValB = matB->Value;

    matC->RowPtr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (rowC + 1));
    memset(matC->RowPtr, 0, sizeof(MAT_PTR_TYPE) * (rowC + 1));
    
    MAT_PTR_TYPE *csrPtrC = matC->RowPtr;

    int *Cub = (int *)malloc(sizeof(int) * rowC);
    memset(Cub, 0, sizeof(int) * rowC);
    for (int i = 0; i < rowA; i ++)
    {
        for (int j = csrPtrA[i]; j < csrPtrA[i + 1]; j ++)
        {
            int cidA = csrCidA[j];
            Cub[i] += csrPtrB[cidA + 1] - csrPtrB[cidA];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < rowA; i ++)
    {
        int rowsize = Cub[i];
        int hashsize = rowsize / 0.75;
        int *hashtable = (int *)malloc(sizeof(int) * hashsize);
        memset(hashtable, -1, sizeof(int) * hashsize);

        for (int j = csrPtrA[i]; j < csrPtrA[i + 1]; j ++)
        {
            int cidA = csrCidA[j];
            for (int k = csrPtrB[cidA]; k < csrPtrB[cidA + 1]; k ++)
            {
                int cidB = csrCidB[k];
                const int key = cidB;
                int hashadr = key % hashsize;
                while(1)
                {
                    const int keyexist = hashtable[hashadr];
                    if (keyexist == key) break;
                    else if (keyexist == -1)
                    {
                        hashtable[hashadr] = key;
                        csrPtrC[i] ++;
                        break;
                    }
                    else
                    {
                        hashadr = (hashadr + 1) % hashsize;
                    }
                }
            }
        }
        free(hashtable);
    }
    exclusive_scan(csrPtrC, rowC + 1);

    int nnzC = csrPtrC[rowC];
    matC->nnz = nnzC;

    matC->ColIdx = (MAT_IDX_TYPE *)malloc(sizeof(MAT_IDX_TYPE) * nnzC);
    matC->Value = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * nnzC);

    MAT_IDX_TYPE *csrCidC = matC->ColIdx;
    MAT_VAL_TYPE *csrValC = matC->Value;
    
    #pragma omp parallel for
    for (int rowid = 0; rowid < rowA; rowid ++)
    {
        //hash
        int hashsize_full_reg = (csrPtrC[rowid + 1] - csrPtrC[rowid]) / 0.75;
        int *tmpHashtable = (int *)malloc(hashsize_full_reg * sizeof(int));
        memset(tmpHashtable, -1, sizeof(int) * hashsize_full_reg);
        MAT_VAL_TYPE *tmpValue = (MAT_VAL_TYPE *)malloc(hashsize_full_reg * sizeof(MAT_VAL_TYPE));
        memset(tmpValue, 0, sizeof(MAT_VAL_TYPE) * hashsize_full_reg); 
        for (int blkj = csrPtrA[rowid]; blkj < csrPtrA[rowid + 1]; blkj ++)
        {
            int col = csrCidA[blkj];
            for(int l = csrPtrB[col]; l < csrPtrB[col+1]; l++)
            {
                const int key = csrCidB[l];
                int hashadr = key % hashsize_full_reg;
                while (1)
                {
                    const int keyexist = tmpHashtable[hashadr];
                    if (keyexist == key)
                    {
                        tmpValue[hashadr] += csrValB[l] * csrValA[blkj];
                        break;
                    }
                    else if (keyexist == -1)
                    {
                        tmpHashtable[hashadr] = key;
                        tmpValue[hashadr] = csrValB[l] * csrValA[blkj];
                        break;
                    }
                    else
                    {
                        hashadr = (hashadr + 1) % hashsize_full_reg;
                    }
                }

            }
        }

        int cptr = csrPtrC[rowid];
        for (int k = 0; k < hashsize_full_reg; k ++)
        {
            if (tmpHashtable[k] != -1)
            {
                csrCidC[cptr] = tmpHashtable[k];
                csrValC[cptr] = tmpValue[k];
                cptr ++;
            }
        }
        free(tmpHashtable);
        free(tmpValue);
        int nnzcnt = csrPtrC[rowid + 1] - csrPtrC[rowid];
        quick_sort_key_val_pair1(csrCidC + csrPtrC[rowid], csrValC + csrPtrC[rowid], nnzcnt);
    }
}

void print_bitmap(unsigned short bitmap)
{
    unsigned short mask = 0xf;

    for (int i = 0; i < 4; i ++)
    {
        unsigned short line = (bitmap & mask) >> (i << 2);

        for (int j = 0; j < 4; j ++)
        {
            printf("%d", line & 1);
            line >>= 1;
        }
        puts("");
        mask <<= 4;
    }
    puts("");
}

void print_bsrval(MAT_VAL_TYPE *bsrval)
{
    for (int i = 0; i < 4; i ++)
    {
        for (int j = 0; j < 4; j ++)
        {
            printf("%lf ", bsrval[i * 4 + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void release_device_bsrMAT(bsrMAT mat)
{
    cudaFree(mat.blcPtr);
    cudaFree(mat.blcIdx);
    cudaFree(mat.blcMap);
    cudaFree(mat.blcVal);
}

void release_host_bsrMAT(bsrMAT mat)
{
    free(mat.blcPtr);
    free(mat.blcIdx);
    free(mat.blcMap);
    free(mat.blcVal);
}

void release_host_csrMAT(csrMAT mat)
{
    free(mat.RowPtr);
    free(mat.ColIdx);
    free(mat.Value);
}