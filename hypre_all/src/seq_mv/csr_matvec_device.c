/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

double time_spmv_preprocess = 0;
double time_spmv_sum = 0;
int spmv_times = 0;

double csr2bsr_step1 = 0;
double csr2bsr_step2 = 0;
double csr2bsr_step3 = 0;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_ALG_DEFAULT
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG3

// #if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
// #define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_CSR_ALG2
// #define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG3

#elif CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_ALG_DEFAULT
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG1

#else
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_ALG_DEFAULT
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_CSRMM_ALG1
#endif

// #define gettimeofday1(a, b) \ 
//         cudaDeviceSynchronize(); \
//         gettimeofday(a, b)

/* y = alpha * A * x + beta * y
 * This function is supposed to be only used inside the other functions in this file
 */
static inline HYPRE_Int
hypre_CSRMatrixMatvecDevice2(HYPRE_Int trans,
                             HYPRE_Complex alpha,
                             hypre_CSRMatrix *A,
                             hypre_Vector *x,
                             HYPRE_Complex beta,
                             hypre_Vector *y,
                             HYPRE_Int offset)
{
    /* Sanity check */
    if (hypre_VectorData(x) == hypre_VectorData(y))
    {
        hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                          "ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice2");
    }

#if defined(HYPRE_USING_CUSPARSE) ||  \
    defined(HYPRE_USING_ROCSPARSE) || \
    defined(HYPRE_USING_ONEMKLSPARSE)

    /* Input variables */
    HYPRE_Int num_vectors_x = hypre_VectorNumVectors(x);
    HYPRE_Int num_vectors_y = hypre_VectorNumVectors(y);

    /* Local variables */
    HYPRE_Int use_vendor = hypre_HandleSpMVUseVendor(hypre_handle());

#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
    HYPRE_Int multivec_storage_x = hypre_VectorMultiVecStorageMethod(x);
    HYPRE_Int multivec_storage_y = hypre_VectorMultiVecStorageMethod(y);

    /* Force use of hypre's SpMV for row-wise multivectors */
    if ((num_vectors_x > 1 && multivec_storage_x == 1) ||
        (num_vectors_y > 1 && multivec_storage_y == 1))
    {
        use_vendor = 0;
    }
#else
    /* TODO - enable cuda 10, rocsparse, and onemkle sparse support for multi-vectors */
    if (num_vectors_x > 1 || num_vectors_y > 1)
    {
        use_vendor = 0;
    }
#endif

    if (use_vendor)
    {
#if defined(HYPRE_USING_CUSPARSE)
        hypre_CSRMatrixMatvecCusparse(trans, alpha, A, x, beta, y, offset);

#elif defined(HYPRE_USING_ROCSPARSE)
        hypre_CSRMatrixMatvecRocsparse(trans, alpha, A, x, beta, y, offset);

#elif defined(HYPRE_USING_ONEMKLSPARSE)
        hypre_CSRMatrixMatvecOnemklsparse(trans, alpha, A, x, beta, y, offset);
#endif
    }
    else
#endif // defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) ...
    {
#if defined(HYPRE_USING_GPU)
        hypre_CSRMatrixSpMVDevice(trans, alpha, A, x, beta, y, 0);

#elif defined(HYPRE_USING_DEVICE_OPENMP)
        hypre_CSRMatrixMatvecOMPOffload(trans, alpha, A, x, beta, y, offset);
#endif
    }

    return hypre_error_flag;
}

/* y = alpha * A * x + beta * b */
HYPRE_Int
hypre_CSRMatrixMatvecDevice(HYPRE_Int trans,
                            HYPRE_Complex alpha,
                            hypre_CSRMatrix *A,
                            hypre_Vector *x,
                            HYPRE_Complex beta,
                            hypre_Vector *b,
                            hypre_Vector *y,
                            HYPRE_Int offset)
{
    HYPRE_Int m_a = hypre_CSRMatrixNumRows(A);
    // printf("spmv %d\n",m_a);
    // hypre_GpuProfilingPushRange("CSRMatrixMatvec");
    HYPRE_Int num_vectors = hypre_VectorNumVectors(x);

    // TODO: RL: do we need offset > 0 at all?
    hypre_assert(offset == 0);

    // VPM: offset > 0 does not work with multivectors. Remove offset? See comment above
    hypre_assert(!(offset != 0 && num_vectors > 1));
    hypre_assert(num_vectors > 0);

    HYPRE_Int nx = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
    HYPRE_Int ny = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);

    // RL: Note the "<=", since the vectors sometimes can be temporary work spaces that have
    //     large sizes than the needed (such as in par_cheby.c)
    hypre_assert(ny <= hypre_VectorSize(y));
    hypre_assert(nx <= hypre_VectorSize(x));
    hypre_assert(ny <= hypre_VectorSize(b));

    // hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
    // hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
    // hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);
    // if (hypre_VectorData(b) != hypre_VectorData(y))
    //{
    //    hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
    // }

    if (hypre_VectorData(b) != hypre_VectorData(y))
    {
        hypre_TMemcpy(hypre_VectorData(y) + offset,
                      hypre_VectorData(b) + offset,
                      HYPRE_Complex,
                      (ny - offset) * num_vectors,
                      hypre_VectorMemoryLocation(y),
                      hypre_VectorMemoryLocation(b));
    }

    if (hypre_CSRMatrixNumNonzeros(A) <= 0 || alpha == 0.0)
    {
        hypre_SeqVectorScale(beta, y);
    }
    else
    {
        hypre_CSRMatrixMatvecDevice2(trans, alpha, A, x, beta, y, offset);
    }

#if defined(HYPRE_USING_GPU)
    hypre_SyncComputeStream(hypre_handle());
#endif

    // hypre_GpuProfilingPopRange();

    return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecCusparseNewAPI
 *
 * Sparse Matrix/(Multi)Vector interface to cusparse's API 11
 *
 * Note: The descriptor variables are not saved to allow for generic input
 *--------------------------------------------------------------------------*/

__device__ __forceinline__ void mma_m16n8k4_tf32_spmv(float *acc, uint32_t *frag_a, uint32_t *frag_b)
{
    asm volatile(

        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(frag_a[0]), "r"(frag_a[1]),
          "r"(frag_b[0]));
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
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]) : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
}

__forceinline__ __device__ int sum_32_shfl_int(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}
int BinarySearch(int *arr, int len, int target)
{
    int low = 0;
    int high = len;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__device__ __host__ int BinarySearch2(int *arr, int left, int right, int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}

__device__ __host__ int BinarySearch3(unsigned int *arr, int left, int right, unsigned int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__device__ __forceinline__ void mma_m8n8k4(MAT_VAL_TYPE *acc, MAT_VAL_TYPE &frag_a, MAT_VAL_TYPE &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}

__device__ __host__ int BinarySearch2_SpMV(int *arr, int left, int right, int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__global__ void bsr_spmv(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                         MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                         int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha, MAT_VAL_TYPE beta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[0] + beta * d_y[rowid];
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[1] + beta * d_y[rowid];
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[0] + beta * d_y[rowid];
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[1] + beta * d_y[rowid];
    }
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

void release_host_bsrMAT(bsrMAT mat)
{
    free(mat.blcPtr);
    free(mat.blcIdx);
    free(mat.blcMap);
    free(mat.blcVal);
}

template <int SM_SIZE>
__global__ void CSR_TO_BSR_get_rowptr(int blc_row, int blc_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                                      MAT_PTR_TYPE *csrPtr, int *csrIdx)
{

    int threadid = threadIdx.x;
    int blockid = blockIdx.x;
    int laneid = threadid & (WARP_SIZE - 1);

    __shared__ unsigned int mask[SM_SIZE];

    for (int j = threadid; j < ((blc_col + 31) >> 5); j += blockDim.x)
    {
        mask[j] = 0;
    }
    __syncthreads();

    int start = csrPtr[blockid * 4];

    int end = csrPtr[(blockid * 4 + 4) > csr_row ? csr_row : (blockid * 4 + 4)];

    for (int i = start + threadid; i < end; i += blockDim.x)
    {
        int colid = csrIdx[i];
        int key = colid / BSR_N;
        atomicOr(&(mask[key >> 5]), 1 << (key & 31));
    }

    __syncthreads();

    int sum = 0;
    for (int i = threadid; i < ((blc_col + 31) >> 5); i += blockDim.x)
    {
        unsigned int now = mask[i];
#pragma unroll
        for (int j = 0; j < 32; j++)
        {
            if ((now & 1) == 1)
            {
                sum++;
            }
            now /= 2;
        }
    }
    __syncthreads();
    sum = sum_32_shfl_int(sum);
    __syncthreads();

    if (laneid == 0)
    {
        atomicAdd(&blcPtr[blockid], sum);
    }
    return;
}

__global__ void CSR_TO_BSR_get_rowptr_noshare(int blc_row, int blc_col, int csr_row, int csr_col, int every_layer, MAT_PTR_TYPE *blcPtr,
                                              MAT_PTR_TYPE *csrPtr, int *csrIdx, int *sum_tmp_memory)
{
    int *tmp_memory = sum_tmp_memory + blockIdx.x * ((blc_col + 31) >> 5);
    int threadid = threadIdx.x;
    int laneid = threadid & (WARP_SIZE - 1);
    for (int blockid = blockIdx.x; blockid < blc_row; blockid += BLOCK_NUM_NO_SHARE)
    {
        __syncthreads();
        for (int j = threadid; j < ((blc_col + 31) >> 5); j += blockDim.x)
        {
            tmp_memory[j] = 0;
        }
        __syncthreads();

        int start = csrPtr[blockid * 4];

        int end = csrPtr[(blockid * 4 + 4) > csr_row ? csr_row : (blockid * 4 + 4)];

        for (int i = start + threadid; i < end; i += blockDim.x)
        {
            int colid = csrIdx[i];
            int key = colid / BSR_N;
            atomicOr(&(tmp_memory[key >> 5]), 1 << (key & 31));
        }

        __syncthreads();

        int sum = 0;
        for (int i = threadid; i < ((blc_col + 31) >> 5); i += blockDim.x)
        {
            unsigned int now = tmp_memory[i];
#pragma unroll
            for (int j = 0; j < 32; j++)
            {
                if ((now & 1) == 1)
                {
                    sum++;
                }
                now /= 2;
            }
        }
        __syncthreads();
        sum = sum_32_shfl_int(sum);
        __syncthreads();

        if (laneid == 0)
        {
            atomicAdd(&blcPtr[blockid], sum);
        }
    }
    return;
}

template <int SM_SIZE>
__global__ void CSR_TO_CSR_getidx(int blc_row, int blc_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                                  MAT_IDX_TYPE *blcIdx, MAT_VAL_TYPE *blcVal, int *blcMap,
                                  MAT_PTR_TYPE *csrPtr, MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal)
{

    int threadid = threadIdx.x;
    int blockid = blockIdx.x;
    int landid = threadid & 31;
    int warpid = threadid >> 5;

    __shared__ unsigned int mask[SM_SIZE];
    __shared__ unsigned char colidx[128];
    __shared__ int block_Sum[1];

    if (threadid == 0)
    {
        block_Sum[0] = 0;
    }

    for (int j = threadid; j < ((blc_col + 31) >> 5); j += blockDim.x)
    {
        mask[j] = 0;
    }
    __syncthreads();

    int start = csrPtr[blockid * 4];

    int end = csrPtr[(blockid * 4 + 4) > csr_row ? csr_row : (blockid * 4 + 4)];

    for (int i = start + threadid; i < end; i += blockDim.x)
    {
        int colid = csrIdx[i];
        int key = colid / BSR_N;
        atomicOr(&(mask[key >> 5]), 1 << (key & 31));
    }

    __syncthreads();

    int block_nnz = blcPtr[blockid];
    MAT_IDX_TYPE *now_blcIdx = blcIdx + block_nnz;
    MAT_VAL_TYPE *now_blcVal = blcVal + (block_nnz * (BSR_M * BSR_N));
    int *now_blcMap = blcMap + (block_nnz / 2);
    int Map_flag = block_nnz & 1;
    block_nnz = blcPtr[blockid + 1] - block_nnz;

    int offset;

    for (int i = 0; i < ((blc_col + 31) >> 5); i += 4)
    {
        offset = block_Sum[0];
        __syncthreads();
        unsigned int now;
        if ((i * 32 + threadid) < blc_col)
        {
            now = mask[i + warpid];
        }
        else
        {
            now = 0;
        }
        int flag = (now >> landid) & 1;
        if (flag == 1)
        {
            atomicAdd(&block_Sum[0], 1);
            colidx[threadid] = 1;
        }
        else
        {
            colidx[threadid] = 2;
        }
        __syncthreads();
        if (flag == 1)
        {
            for (int j = 0; j < threadid; j++)
            {
                if (colidx[j] == 1)
                {
                    offset = offset + 1;
                }
            }
            now_blcIdx[offset] = (i << 5) + threadid;
        }

        __syncthreads();
    }

    __syncthreads();

    for (int j = 0; j < BSR_M; j++)
    {
        if ((blockid * 4 + j) >= csr_row)
        {
            break;
        }
        start = csrPtr[(blockid * 4 + j)];
        end = csrPtr[blockid * 4 + j + 1];
        for (int i = start + threadid; i < end; i += blockDim.x)
        {
            int colid = csrIdx[i];
            int key = colid / BSR_N;
            int offset_cid = BinarySearch2_SpMV(now_blcIdx, 0, block_nnz, key);
            int offset_idx = (j * BSR_M) + (colid % BSR_N);

            now_blcVal[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = csrVal[i];
            atomicOr(&(now_blcMap[(offset_cid + Map_flag) >> 1]), 1 << ((offset_idx + 16 * ((offset_cid + Map_flag) & 1)) & 31));
        }
        __syncthreads();
    }

    __syncthreads();
    return;
}

__global__ void CSR_TO_BSR_getidx_noshare(int *bin_rowidx, int *bin_offset, int bin,
                                          int blc_row, int blc_col, int csr_row, int csr_col, int every_layer,
                                          MAT_PTR_TYPE *blcPtr, MAT_IDX_TYPE *blcIdx, MAT_VAL_TYPE *blcVal, int *blcMap,
                                          MAT_PTR_TYPE *csrPtr, MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal, int *sum_tmp_memory)
{
    int *tmp_memory = sum_tmp_memory + blockIdx.x * ((blc_col + 31) >> 5);
    int threadid = threadIdx.x;
    int landid = threadid & 31;
    int warpid = threadid >> 5;

    __shared__ unsigned char colidx[128];
    __shared__ int block_Sum[1];

    for (int bin_row_offset = bin_offset[bin] + blockIdx.x; bin_row_offset < bin_offset[bin + 1]; bin_row_offset += BLOCK_NUM_NO_SHARE)
    {
        int blockid = bin_rowidx[bin_row_offset];
        if (threadid == 0)
        {
            block_Sum[0] = 0;
        }
        for (int j = threadid; j < ((blc_col + 31) >> 5); j += blockDim.x)
        {
            tmp_memory[j] = 0;
        }
        __syncthreads();

        int start = csrPtr[blockid * 4];

        int end = csrPtr[(blockid * 4 + 4) > csr_row ? csr_row : (blockid * 4 + 4)];

        for (int i = start + threadid; i < end; i += blockDim.x)
        {
            int colid = csrIdx[i];
            int key = colid / BSR_N;
            atomicOr(&(tmp_memory[key >> 5]), 1 << (key & 31));
        }

        __syncthreads();

        int block_nnz = blcPtr[blockid];
        MAT_IDX_TYPE *now_blcIdx = blcIdx + block_nnz;
        MAT_VAL_TYPE *now_blcVal = blcVal + (block_nnz * (BSR_M * BSR_N));
        int *now_blcMap = blcMap + (block_nnz / 2);
        int Map_flag = block_nnz & 1;
        block_nnz = blcPtr[blockid + 1] - block_nnz;

        int offset;

        for (int i = 0; i < ((blc_col + 31) >> 5); i += 4)
        {
            offset = block_Sum[0];
            __syncthreads();
            unsigned int now;
            if ((i * 32 + threadid) < blc_col)
            {
                now = tmp_memory[i + warpid];
            }
            else
            {
                now = 0;
            }
            int flag = (now >> landid) & 1;
            if (flag == 1)
            {
                atomicAdd(&block_Sum[0], 1);
                colidx[threadid] = 1;
            }
            else
            {
                colidx[threadid] = 2;
            }
            __syncthreads();
            if (flag == 1)
            {
                for (int j = 0; j < threadid; j++)
                {
                    if (colidx[j] == 1)
                    {
                        offset = offset + 1;
                    }
                }
                now_blcIdx[offset] = (i << 5) + threadid;
            }

            __syncthreads();
        }

        __syncthreads();

        for (int j = 0; j < BSR_M; j++)
        {
            if ((blockid * 4 + j) >= csr_row)
            {
                break;
            }
            start = csrPtr[(blockid * 4 + j)];
            end = csrPtr[blockid * 4 + j + 1];
            for (int i = start + threadid; i < end; i += blockDim.x)
            {
                int colid = csrIdx[i];
                int key = colid / BSR_N;
                int offset_cid = BinarySearch2(now_blcIdx, 0, block_nnz, key);
                int offset_idx = (j * BSR_M) + (colid % BSR_N);
                now_blcVal[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = csrVal[i];
                atomicOr(&(now_blcMap[(offset_cid + Map_flag) >> 1]), 1 << ((offset_idx + 16 * ((offset_cid + Map_flag) & 1)) & 31));
            }
        }

        __syncthreads();
    }
    return;
}
void CSR_TO_BSR_step1(int bsr_row, int bsr_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                      MAT_PTR_TYPE *csrPtr, MAT_PTR_TYPE *csrIdx, int *sum_tmp_memory)
{

    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum = bsr_row;

    if (csr_col < 128)
    {
        CSR_TO_BSR_get_rowptr<32><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else if (csr_col < 256)
    {
        CSR_TO_BSR_get_rowptr<64><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else if (csr_col < 512)
    {
        CSR_TO_BSR_get_rowptr<128><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else if (csr_col < 1024)
    {
        CSR_TO_BSR_get_rowptr<256><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else if (csr_col < 2048)
    {
        CSR_TO_BSR_get_rowptr<512><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else if (csr_col < 4096)
    {
        CSR_TO_BSR_get_rowptr<1024><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else if (csr_col < 8192)
    {
        CSR_TO_BSR_get_rowptr<2048><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else if (csr_col < 16384)
    {
        CSR_TO_BSR_get_rowptr<4096><<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
    }
    else
    {
        BlockNum = BLOCK_NUM_NO_SHARE;
        int every_layer = (bsr_row + BlockNum - 1) / BlockNum;
        CSR_TO_BSR_get_rowptr_noshare<<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, every_layer, blcPtr, csrPtr, csrIdx, sum_tmp_memory);
    }
}

__global__ void CSR2BSR_compute_bin(MAT_PTR_TYPE *BlcPtr, int m, int *bin_offset)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= m)
        return;

    int cur_Cub = BlcPtr[threadid + 1] - BlcPtr[threadid];

    if (cur_Cub < 128)
    {
        atomicAdd(&bin_offset[0], 1);
    }
    else if (cur_Cub >= 128 && cur_Cub < 256)
    {
        atomicAdd(&bin_offset[1], 1);
    }
    else if (cur_Cub >= 256 && cur_Cub < 512)
    {
        atomicAdd(&bin_offset[2], 1);
    }
    else if (cur_Cub >= 512 && cur_Cub < 1024)
    {
        atomicAdd(&bin_offset[3], 1);
    }
    else if (cur_Cub >= 1024 && cur_Cub < 2048)
    {
        atomicAdd(&bin_offset[4], 1);
    }
    else if (cur_Cub >= 2048 && cur_Cub < 4096)
    {
        atomicAdd(&bin_offset[5], 1);
    }
    else
    {
        atomicAdd(&bin_offset[6], 1);
    }
    __syncthreads();
}

__global__ void CSR2BSR_set_bin(int m, MAT_PTR_TYPE *BlcPtr, MAT_IDX_TYPE *bin_rowidx, int *bin_offset, int *bin_size, int *max_num)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= m)
        return;

    int cur_Cub = BlcPtr[threadid + 1] - BlcPtr[threadid];
    int idx = 0;

    if (cur_Cub < 128)
    {
        idx = atomicAdd(&bin_size[0], 1);
        bin_rowidx[bin_offset[0] + idx] = threadid;
    }
    else if (cur_Cub >= 128 && cur_Cub < 256)
    {
        idx = atomicAdd(&bin_size[1], 1);
        bin_rowidx[bin_offset[1] + idx] = threadid;
    }
    else if (cur_Cub >= 256 && cur_Cub < 512)
    {
        idx = atomicAdd(&bin_size[2], 1);
        bin_rowidx[bin_offset[2] + idx] = threadid;
    }
    else if (cur_Cub >= 512 && cur_Cub < 1024)
    {
        idx = atomicAdd(&bin_size[3], 1);
        bin_rowidx[bin_offset[3] + idx] = threadid;
    }
    else if (cur_Cub >= 1024 && cur_Cub < 2048)
    {
        idx = atomicAdd(&bin_size[4], 1);
        bin_rowidx[bin_offset[4] + idx] = threadid;
    }
    else if (cur_Cub >= 2048 && cur_Cub < 4096)
    {
        idx = atomicAdd(&bin_size[5], 1);
        bin_rowidx[bin_offset[5] + idx] = threadid;
    }
    else
    {
        idx = atomicAdd(&bin_size[6], 1);
        bin_rowidx[bin_offset[6] + idx] = threadid;
        atomicMax(max_num, cur_Cub);
    }

    // printf("thread %d idx %d\n",threadid,idx);
}
#define CSR2BSR_BIN_NUM 7

template <int SM_SIZE>
__global__ void CSR_TO_CSR_bin_getidx(int *bin_rowidx, int *bin_offset, int bin,
                                      int blc_row, int blc_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                                      MAT_IDX_TYPE *blcIdx, MAT_VAL_TYPE *blcVal, unsigned short *blcMap,
                                      MAT_PTR_TYPE *csrPtr, MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal)
{
    int threadid = threadIdx.x;
    int blockid = blockIdx.x;
    int laneid = threadid & 31;
    int bin_row_offset = bin_offset[bin] + blockid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    __shared__ int hashtable[SM_SIZE];
    __shared__ unsigned int maptable[SM_SIZE];
    __shared__ int nz_num[1];

    if (threadid == 0)
    {
        nz_num[0] = 0;
    }

    for (int i = threadid; i < SM_SIZE; i += blockDim.x)
    {
        hashtable[i] = -1;
    }

    for (int i = threadid; i < SM_SIZE; i += blockDim.x)
    {
        maptable[i] = 0;
    }
    __syncthreads();

    int rowid = bin_rowidx[bin_row_offset];

    int end_row = (rowid * 4 + 4) < csr_row ? (4) : (csr_row - rowid * 4);
    for (int i = 0; i < end_row; i++)
    {
        int start = csrPtr[i + rowid * 4];
        int end = csrPtr[i + rowid * 4 + 1];
        for (int j = start + threadid; j < end; j += blockDim.x)
        {
            int col = csrIdx[j];
            int key = col / BSR_N;
            int hashadr = key & (SM_SIZE - 1);
            while (1)
            {
                int keyexist = hashtable[hashadr];
                if (keyexist == key)
                {
                    atomicOr(maptable + hashadr, 1 << (i * 4 + (col % 4)));
                    break;
                }

                else if (keyexist == -1)
                {
                    int idx = atomicCAS(hashtable + hashadr, -1, key);
                    if (idx == -1)
                    {
                        atomicOr(maptable + hashadr, 1 << (i * 4 + (col % 4)));
                        break;
                    }
                }
                else
                {
                    hashadr = (hashadr + 1) & (SM_SIZE - 1);
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x < WARP_SIZE)
    {
        for (int i = 0; i < SM_SIZE; i += WARP_SIZE)
        {
            unsigned int res_map = 0;
            int res = -1;
            if ((i + laneid) < SM_SIZE)
            {
                res_map = maptable[i + laneid];
                res = hashtable[i + laneid];
            }
            __syncwarp();
            if (res != -1)
            {
                int ind = atomicAdd(&nz_num[0], 1);
                hashtable[ind] = res;
                maptable[ind] = res_map;
            }
        }
    }
    __syncthreads();

    int len = nz_num[0];

    int offset = blcPtr[rowid];
    int target, count;
    unsigned int target_map;
    unsigned short set_num = 0x0000ffff;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        target = hashtable[i];
        target_map = maptable[i];
        count = 0;
        for (int j = 0; j < len; j++)
        {
            count += ((unsigned int)(hashtable[j] - target) >> 31);
        }
        blcIdx[offset + count] = target;
        blcMap[offset + count] = target_map & set_num;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        target = hashtable[i];
        count = 0;
        for (int j = 0; j < len; j++)
        {
            count += ((unsigned int)(hashtable[j] - target) >> 31);
        }
        maptable[count] = (unsigned int)target;
    }
    __syncthreads();

    double *now_blcVal = blcVal + (offset * (BSR_M * BSR_N));
    for (int i = 0; i < end_row; i++)
    {
        int start = csrPtr[i + rowid * 4];
        int end = csrPtr[i + rowid * 4 + 1];
        for (int j = start + threadid; j < end; j += blockDim.x)
        {
            int col = csrIdx[j];
            double value = csrVal[j];
            unsigned int key = col / BSR_N;
            int offset_cid = BinarySearch3(maptable, 0, len, key);
            int offset_idx = (i * BSR_M) + (col % BSR_N);
            now_blcVal[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = value;
        }
    }
    __syncthreads();
    return;
}
double CSR_TO_BSR_step2_new(int bsr_row, int bsr_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                            MAT_IDX_TYPE *blcIdx, MAT_VAL_TYPE *blcVal, MAT_MAP_TYPE *blcMap,
                            MAT_PTR_TYPE *csrPtr, MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal, int *sum_tmp_memory)
{
    struct timeval t1, t2;

    int *bin_offset;
    cudaMalloc((void **)&bin_offset, sizeof(int) * (CSR2BSR_BIN_NUM + 1));
    cudaMemset(bin_offset, 0, sizeof(int) * (CSR2BSR_BIN_NUM + 1));
    int *bin_size;
    cudaMalloc((void **)&bin_size, sizeof(int) * CSR2BSR_BIN_NUM);
    cudaMemset(bin_size, 0, sizeof(int) * CSR2BSR_BIN_NUM);
    MAT_IDX_TYPE *bin_rowidx;
    cudaMalloc((void **)&bin_rowidx, sizeof(MAT_IDX_TYPE) * bsr_row);
    int *max_num;
    cudaMalloc((void **)&max_num, sizeof(int));

    int ThreadNum = 128;
    int BlockNum = (bsr_row + ThreadNum - 1) / ThreadNum;

    double sum_time = 0.0;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    CSR2BSR_compute_bin<<<BlockNum, ThreadNum>>>(blcPtr, bsr_row, bin_offset);

    thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + (CSR2BSR_BIN_NUM + 1), bin_offset, 0);

    CSR2BSR_set_bin<<<BlockNum, ThreadNum>>>(bsr_row, blcPtr, bin_rowidx, bin_offset, bin_size, max_num);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    sum_time += time;

    // printf("preprocess time %lf ms\n", time);

    int max_len;
    cudaMemcpy(&max_len, max_num, sizeof(int), cudaMemcpyDeviceToHost);
    int *offset = (int *)malloc(sizeof(int) * (CSR2BSR_BIN_NUM + 1));
    cudaMemcpy(offset, bin_offset, sizeof(int) * (CSR2BSR_BIN_NUM + 1), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    for (int i = CSR2BSR_BIN_NUM - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = 128;
        BlockNum = row_num;

        if (row_num)
        {
            switch (i)
            {
            case 0:
                CSR_TO_CSR_bin_getidx<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    bsr_row, bsr_col, csr_row, csr_col,
                                                                    blcPtr, blcIdx, blcVal, blcMap,
                                                                    csrPtr, csrIdx, csrVal);
                break;
            case 1:
                CSR_TO_CSR_bin_getidx<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    bsr_row, bsr_col, csr_row, csr_col,
                                                                    blcPtr, blcIdx, blcVal, blcMap,
                                                                    csrPtr, csrIdx, csrVal);
                break;
            case 2:
                CSR_TO_CSR_bin_getidx<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    bsr_row, bsr_col, csr_row, csr_col,
                                                                    blcPtr, blcIdx, blcVal, blcMap,
                                                                    csrPtr, csrIdx, csrVal);
                break;
            case 3:
                CSR_TO_CSR_bin_getidx<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     bsr_row, bsr_col, csr_row, csr_col,
                                                                     blcPtr, blcIdx, blcVal, blcMap,
                                                                     csrPtr, csrIdx, csrVal);
                break;
            case 4:
                CSR_TO_CSR_bin_getidx<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     bsr_row, bsr_col, csr_row, csr_col,
                                                                     blcPtr, blcIdx, blcVal, blcMap,
                                                                     csrPtr, csrIdx, csrVal);
                break;
            case 5:
                CSR_TO_CSR_bin_getidx<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     bsr_row, bsr_col, csr_row, csr_col,
                                                                     blcPtr, blcIdx, blcVal, blcMap,
                                                                     csrPtr, csrIdx, csrVal);
                break;
            case 6:
            {
                if (max_len == 0)
                {
                    max_len = 4097;
                }
                // unsigned int *TMP_maptable = NULL;
                // int *TMP_hashtable = NULL;
                // int *TMP_nz_num = NULL;

                // max_len = max_len + (max_len >> 2);

                // cudaMalloc((void **)&TMP_nz_num, sizeof(int) * row_num);
                // cudaMalloc((void **)&TMP_hashtable, sizeof(int) * max_len * row_num);
                // cudaMalloc((void **)&TMP_maptable, sizeof(unsigned int) * max_len * row_num);

                // CSR_TO_CSR_bin_getidx_noshare<<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                //                                                        bsr_row, bsr_col, csr_row, csr_col,
                //                                                        blcPtr, blcIdx, blcVal, blcMap,
                //                                                        csrPtr, csrIdx, csrVal,
                //                                                        TMP_hashtable, TMP_maptable, TMP_nz_num, max_len);

                // cudaFree(TMP_maptable);
                // cudaFree(TMP_hashtable);
                // cudaFree(TMP_nz_num);

                if (sum_tmp_memory == NULL)
                {
                    cudaMalloc((void **)&sum_tmp_memory, sizeof(MAT_PTR_TYPE) * (BLOCK_NUM_NO_SHARE * ((row_num + 31) >> 5)));
                    int every_layer = (row_num + BlockNum - 1) / BlockNum;
                    CSR_TO_BSR_getidx_noshare<<<BLOCK_NUM_NO_SHARE, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                                 bsr_row, bsr_col, csr_row, csr_col, every_layer,
                                                                                 blcPtr, blcIdx, blcVal, (int *)blcMap,
                                                                                 csrPtr, csrIdx, csrVal, sum_tmp_memory);
                    cudaFree(sum_tmp_memory);
                }
                else
                {
                    int every_layer = (row_num + BlockNum - 1) / BlockNum;
                    CSR_TO_BSR_getidx_noshare<<<BLOCK_NUM_NO_SHARE, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                                 bsr_row, bsr_col, csr_row, csr_col, every_layer,
                                                                                 blcPtr, blcIdx, blcVal, (int *)blcMap,
                                                                                 csrPtr, csrIdx, csrVal, sum_tmp_memory);
                }
            }
            break;
            default:
                break;
            }
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    sum_time += time;
    // printf("run time %lf ms\n", time);

    cudaFree(bin_offset);
    cudaFree(bin_size);
    cudaFree(bin_rowidx);
    cudaFree(max_num);

    // printf("step preprocess + run time %lf ms\n", sum_time);
    return sum_time;
}
__global__ void bsr_spmv_balanced_cc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, MAT_VAL_TYPE *d_blcVal,
                                          MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);
    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE res = 0;

    for (int i = start + groupid; i < end; i += 8)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        atomicAdd(&d_y[blc_rid * BSR_M + laneid], res * alpha);
    }
}
__global__ void bsr_spmv_tc_fp64(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                                 MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                                 int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            d_y[rowid] += fragC[0] * alpha;
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            d_y[rowid] += fragC[1] * alpha;
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            d_y[rowid] += fragC[0] * alpha;
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            d_y[rowid] += fragC[1] * alpha;
    }
}

__global__ void bsr_spmv_cc_fp64(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, MAT_VAL_TYPE *d_blcVal,
                                 MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                                 int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE res = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        d_y[blc_rid * BSR_M + laneid] += alpha * res;
    }
}
__global__ void bsr_spmv_balanced_tc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                                          MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;

    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
}

__global__ void get_rowPtrbyWarp(MAT_PTR_TYPE *d_blcPtr, int *rowPtrbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    rowPtrbyWarp[rowid] = (d_blcPtr[rowid + 1] - d_blcPtr[rowid] + WARP_CAPACITY - 1) / WARP_CAPACITY;
}

__global__ void get_rowIdxbyWarp(int *rowPtrbyWarp, int *rowIdxbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    int offset = rowPtrbyWarp[rowid];
    int stride = rowPtrbyWarp[rowid + 1] - rowPtrbyWarp[rowid];

    for (int i = offset; i < (offset + stride); i++)
    {
        rowIdxbyWarp[i] = rowid;
    }
}
__global__ void getStand(MAT_PTR_TYPE *rowptr, double *sum, double avg_len, int N)
{

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ MAT_VAL_TYPE partialSum[256];

    if (idx < N)
    {
        // partialSum[threadIdx.x] = a[idx] * b[idx];
        partialSum[threadIdx.x] = pow(rowptr[idx + 1] - rowptr[idx] - avg_len, 2);
    }
    else
    {
        partialSum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&sum[0], partialSum[0]);
    }
}
__global__ void beta_vecY(MAT_VAL_TYPE *d_y, MAT_VAL_TYPE beta, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_y[rowid] *= beta;
}
// int preprocess_num = 0;
void CSR2BSR_GPU(hypre_CSRMatrix *A)
{
    if (!hypre_BSRTAG(A))
    {
        // preprocess_num++;
        // // printf("%d\n", preprocess_num);
        // int row = hypre_CSRMatrixNumRows(A);
        // int column = hypre_CSRMatrixNumCols(A);
        // int A_nnz = hypre_CSRMatrixNumNonzeros(A);
        // MAT_PTR_TYPE *RowPtr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (row + 1));
        // cudaMemcpy(RowPtr, hypre_CSRMatrixI(A), sizeof(MAT_PTR_TYPE) * (row + 1), cudaMemcpyDeviceToHost);

        // MAT_IDX_TYPE *ColIdx = (MAT_IDX_TYPE *)malloc(sizeof(MAT_IDX_TYPE) * A_nnz);
        // cudaMemcpy(ColIdx, hypre_CSRMatrixJ(A), sizeof(MAT_PTR_TYPE) * (A_nnz), cudaMemcpyDeviceToHost);

        // MAT_VAL_TYPE *Value = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * A_nnz);
        // cudaMemcpy(Value, hypre_CSRMatrixData(A), sizeof(MAT_VAL_TYPE) * (A_nnz), cudaMemcpyDeviceToHost);
        // char file_name[] = "nd6k_A_a.mtx";
        // file_name[7] += preprocess_num;

        // FILE *fp = fopen(file_name, "w+");
        // fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
        // fprintf(fp, "%ld ", row);
        // fprintf(fp, "%ld ", column);
        // fprintf(fp, "%ld\n", A_nnz);

        // for (int i = 0; i < row; i++)
        // {
        //     for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        //     {
        //         fprintf(fp, "%d %d %lf\n", i + 1, ColIdx[j] + 1, Value[j]);
        //     }
        // }

        // fclose(fp);

        struct timeval t_start, t_end;
        hypre_BSRTAG(A) = 1;
        (hypre_BSR(A)) = (bsrMAT *)malloc(sizeof(bsrMAT));
        bsrMAT *bsrmat = (hypre_BSR(A));
        int *d_csrptr = hypre_CSRMatrixI(A);
        int *d_csridx = hypre_CSRMatrixJ(A);
        MAT_VAL_TYPE *d_csrval = hypre_CSRMatrixData(A);
        bsrmat->row = hypre_CSRMatrixNumRows(A);
        bsrmat->col = hypre_CSRMatrixNumCols(A);
        bsrmat->blc_row = (bsrmat->row + BSR_M - 1) / BSR_M;
        bsrmat->blc_col = (bsrmat->col + BSR_N - 1) / BSR_N;

        gettimeofday(&t_start, NULL);
        cudaMalloc((void **)&(bsrmat->blcPtr), sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        cudaMemset(bsrmat->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));

        int *sum_tmp_memory = NULL;
        if (bsrmat->col >= 16384)
        {
            cudaMalloc((void **)&sum_tmp_memory, sizeof(MAT_PTR_TYPE) * (BLOCK_NUM_NO_SHARE * (bsrmat->blc_col + 31) >> 5));
        }
        CSR_TO_BSR_step1(bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, bsrmat->blcPtr, d_csrptr, d_csridx, sum_tmp_memory);
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        csr2bsr_step1 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

        gettimeofday(&t_start, NULL);
        thrust::exclusive_scan(thrust::device, bsrmat->blcPtr, bsrmat->blcPtr + (bsrmat->blc_row + 1), bsrmat->blcPtr, 0);
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        csr2bsr_step2 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

        cudaMemcpy(&(bsrmat->blc_num), &bsrmat->blcPtr[bsrmat->blc_row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
        bsrmat->nnz = bsrmat->blc_num * BSR_M * BSR_N;

        bsrmat->avg_nnz = (double)hypre_CSRMatrixNumNonzeros(A) / (double)(bsrmat->blc_num);

        HYPRE_Real *result_gpu;
        cudaMalloc((void **)&result_gpu, sizeof(HYPRE_Real));
        cudaMemset(result_gpu, 0.0, sizeof(HYPRE_Real));
        int thread_num_stand = 256;
        int block_num_stand = (bsrmat->blc_row + thread_num_stand - 1) / thread_num_stand;
        double avg_len = (double)bsrmat->blc_num / (double)bsrmat->blc_row;

        getStand<<<block_num_stand, thread_num_stand>>>(bsrmat->blcPtr, result_gpu, avg_len, bsrmat->blc_row);
        cudaDeviceSynchronize();
        cudaMemcpy(&bsrmat->stand, result_gpu, sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

        bsrmat->stand = sqrtf(bsrmat->stand / bsrmat->blc_row);

        // printf("avg_len: %lf\tavg_nnz: %lf\tstand:%lf\n", avg_len, bsrmat->avg_nnz, bsrmat->stand);c;

        gettimeofday(&t_start, NULL);
        cudaMalloc((void **)&bsrmat->blcIdx, sizeof(MAT_IDX_TYPE) * bsrmat->blc_num);
        cudaMalloc((void **)&bsrmat->blcVal, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
        cudaMalloc((void **)&bsrmat->blcMap, sizeof(MAT_MAP_TYPE) * (bsrmat->blc_num + 1));

        cudaMemset(bsrmat->blcVal, 0, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
        cudaMemset(bsrmat->blcMap, 0, sizeof(MAT_MAP_TYPE) * (bsrmat->blc_num + 1));

        CSR_TO_BSR_step2_new(bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, bsrmat->blcPtr,
                             bsrmat->blcIdx, bsrmat->blcVal, bsrmat->blcMap,
                             d_csrptr, d_csridx, d_csrval, sum_tmp_memory);
        cudaDeviceSynchronize();

        if (bsrmat->col >= 16384)
        {
            cudaFree(sum_tmp_memory);
            sum_tmp_memory = NULL;
        }

        gettimeofday(&t_end, NULL);
        csr2bsr_step3 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
        // load balanced preprocess

        //   cudaMalloc((void **)&bsrmat->rowPtrbyWarp, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        //   cudaMemset(bsrmat->rowPtrbyWarp, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));

        //   int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
        //   int BlockNum = (bsrmat->blc_row + ThreadNum - 1) / ThreadNum;

        //   get_rowPtrbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->rowPtrbyWarp, bsrmat->blc_row);
        //   cudaDeviceSynchronize();

        //   thrust::exclusive_scan(thrust::device, bsrmat->rowPtrbyWarp, bsrmat->rowPtrbyWarp + (bsrmat->blc_row + 1), bsrmat->rowPtrbyWarp, 0);
        //   cudaDeviceSynchronize();

        //   cudaMemcpy(&bsrmat->warpnum, (bsrmat->rowPtrbyWarp) + bsrmat->blc_row, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);

        //   cudaMalloc((void **)&bsrmat->rowIdxbyWarp, sizeof(int) * bsrmat->warpnum);

        //   get_rowIdxbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->blc_row);
        //   cudaDeviceSynchronize();
    }
}

void BSR_BALANCED_PREPROCESS_GPU(hypre_CSRMatrix *A)
{
#ifdef ADAPTIVE_AMGT_SPMV
    if (!hypre_BSRBALANCEDTAG(A) && hypre_BSR(A)->stand >= 12)
#else
    if (!hypre_BSRBALANCEDTAG(A))
#endif
    {
        hypre_BSRBALANCEDTAG(A) = 1;
        bsrMAT *bsrmat = (hypre_BSR(A));
        // load balanced preprocess

        cudaMalloc((void **)&bsrmat->rowPtrbyWarp, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        cudaMemset(bsrmat->rowPtrbyWarp, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));

        int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
        int BlockNum = (bsrmat->blc_row + ThreadNum - 1) / ThreadNum;

        get_rowPtrbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->rowPtrbyWarp, bsrmat->blc_row);
        cudaDeviceSynchronize();

        thrust::exclusive_scan(thrust::device, bsrmat->rowPtrbyWarp, bsrmat->rowPtrbyWarp + (bsrmat->blc_row + 1), bsrmat->rowPtrbyWarp, 0);
        cudaDeviceSynchronize();

        cudaMemcpy(&bsrmat->warpnum, (bsrmat->rowPtrbyWarp) + bsrmat->blc_row, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);

        cudaMalloc((void **)&bsrmat->rowIdxbyWarp, sizeof(int) * bsrmat->warpnum);

        get_rowIdxbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->blc_row);
        cudaDeviceSynchronize();
    }
}
void spmv_amgT_fp64(HYPRE_Int trans,
                    HYPRE_Complex alpha,
                    hypre_CSRMatrix *A,
                    hypre_Vector *x,
                    HYPRE_Complex beta,
                    hypre_Vector *y,
                    HYPRE_Int offset)
{
    struct timeval t1, t2;
    gettimeofday1(&t1, NULL);
    CSR2BSR_GPU(A);
    BSR_BALANCED_PREPROCESS_GPU(A);
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    gettimeofday1(&t1, NULL);
    bsrMAT *bsrmat = (hypre_BSR(A));
    MAT_VAL_TYPE *dvecX = hypre_VectorData(x);
    MAT_VAL_TYPE *dvecY = hypre_VectorData(y);
    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_b = (bsrmat->warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum = (bsrmat->blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum2 = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    if (beta != 1)
    {
        beta_vecY<<<BlockNum2, ThreadNum>>>(dvecY, beta, bsrmat->row);
        cudaDeviceSynchronize();
    }
    double stand = bsrmat->stand;
    double avgnz = bsrmat->avg_nnz;

#ifdef ADAPTIVE_AMGT_SPMV
    if (stand >= 12 && avgnz >= 10)
    {
        // ===tensor core, balanced===
        bsr_spmv_balanced_tc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand >= 12 && avgnz < 10)
    {
        // ===cuda core, balanced===
        bsr_spmv_balanced_cc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand < 12 && avgnz >= 10)
    {
        // ===tensor core===
        bsr_spmv_tc_fp64<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else
    {
        // ===cuda core===
        bsr_spmv_cc_fp64<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }

#else

    bsr_spmv_balanced_tc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
    cudaDeviceSynchronize();
#endif
    gettimeofday1(&t2, NULL);
    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", bsrmat->row);
    printf("spmv_kernel_n=%d\n", bsrmat->col);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;
}
__global__ void bsr_spmv_balanced_tc_fp32(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, float *d_blcVal,
                                          float *d_x, float *d_y, int blc_row, int blc_col, int row, int col,
                                          float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    uint32_t fragA[2] = {0}, fragB[2] = {0};
    float fragC[4] = {0};

    for (int i = start; i < end; i += 2)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragA[0]) : "f"((i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid]));

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragB[0]) : "f"(d_x[xid + laneid_mod_4]));

        mma_m16n8k4_tf32_spmv(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
}
__global__ void bsr_spmv_balanced_cc_fp32(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, float *d_blcVal,
                                          float *d_x, float *d_y, int blc_row, int blc_col, int row, int col,
                                          float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);
    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    float res = 0;

    for (int i = start + groupid; i < end; i += 8)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        atomicAdd(&d_y[blc_rid * BSR_M + laneid], res * alpha);
    }
}
__global__ void bsr_spmv_tc_fp32(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, float *d_blcVal,
                                 float *d_x, float *d_y,
                                 int blc_row, int blc_col, int row, int col, float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    uint32_t fragA[2] = {0}, fragB[2] = {0};
    float fragC[4] = {0};

    for (int i = start; i < end; i += 2)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragA[0]) : "f"((i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid]));

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragB[0]) : "f"(d_x[xid + laneid_mod_4]));

        mma_m16n8k4_tf32_spmv(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            d_y[rowid] = fragC[0] * alpha;
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            d_y[rowid] = fragC[1] * alpha;
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            d_y[rowid] = fragC[0] * alpha;
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            d_y[rowid] = fragC[1] * alpha;
    }
}
__global__ void bsr_spmv_cc_fp32(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, float *d_blcVal,
                                 float *d_x, float *d_y,
                                 int blc_row, int blc_col, int row, int col, float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    float res = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        d_y[blc_rid * BSR_M + laneid] = alpha * res;
    }
}
__global__ void vec_64_to_32(MAT_VAL_TYPE *d_x_csr, float *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_x_bsr[rowid] = d_x_csr[rowid];
}

__global__ void vec_64_to_16(MAT_VAL_TYPE *d_x_csr, uint32_t *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    half *d_x_half = reinterpret_cast<half *>(&d_x_bsr[0]);
    d_x_half[rowid] = d_x_csr[rowid];
}

__global__ void vec_16_to_64(MAT_VAL_TYPE *d_x_csr, uint32_t *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    half *d_x_half = reinterpret_cast<half *>(&d_x_bsr[0]);
    d_x_csr[rowid] = d_x_half[rowid];
}

__global__ void vec_32_to_64(MAT_VAL_TYPE *d_x_csr, float *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_x_csr[rowid] = d_x_bsr[rowid];
}
__global__ void bsr_spmv_balanced_tc_fp16(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, uint32_t *d_blcVal,
                                          uint32_t *d_x, uint32_t *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);
    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    half res, fragC[8] = {0};
    uint32_t fragA[2], fragB[2];

    int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;

    for (int i = start; i < end; i += 8)
    {
        int cur_blockid = laneid / 4 + i;
        uint32_t *cur_val = d_blcVal + (i * BSR_NNZ / 2);
        fragA[0] = cur_blockid < end ? cur_val[laneid * 2] : 0;
        fragA[1] = cur_blockid < end ? cur_val[laneid * 2 + 1] : 0;

        int xid = cur_blockid < end ? (d_blcCid[cur_blockid] * BSR_N / 2) : (d_blcCid[i] * BSR_N / 2);
        fragB[0] = d_x[xid];
        fragB[1] = d_x[xid + 1];

        mma_m8n8k4_fp16(fragC, fragA, fragB);
    }
    res = fragC[target_idx];

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        int rowid = blc_rid * 4 + laneid;
        if (rowid < row)
            atomicAdd(&d_y_half[rowid], res * (half)alpha);
    }
}
__global__ void bsr_spmv_cc_fp16(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, uint32_t *d_blcVal,
                                 uint32_t *d_x, uint32_t *d_y,
                                 int blc_row, int blc_col, int row, int col, half alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    half *d_A_half = reinterpret_cast<half *>(&d_blcVal[0]);
    half *d_x_half = reinterpret_cast<half *>(&d_x[0]);
    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);

    half res = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        half *cur_val = d_A_half + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x_half[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        d_y_half[blc_rid * BSR_M + laneid] = alpha * res;
    }
}
__global__ void bsr_spmv_balanced_cc_fp16(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, uint32_t *d_blcVal,
                                          uint32_t *d_x, uint32_t *d_y, int blc_row, int blc_col, int row, int col,
                                          half alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);
    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    half *d_A_half = reinterpret_cast<half *>(&d_blcVal[0]);
    half *d_x_half = reinterpret_cast<half *>(&d_x[0]);
    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);

    half res = 0;

    for (int i = start + groupid; i < end; i += 8)
    {
        half *cur_val = d_A_half + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x_half[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        atomicAdd(&d_y_half[blc_rid * BSR_M + laneid], res * alpha);
    }
}
__global__ void bsr_spmv_tc_fp16(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, uint32_t *d_blcVal,
                                 uint32_t *d_x, uint32_t *d_y,
                                 int blc_row, int blc_col, int row, int col, half alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    half res, fragC[8] = {0};
    uint32_t fragA[2], fragB[2];

    int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;

    for (int i = start; i < end; i += 8)
    {
        int cur_blockid = laneid / 4 + i;
        uint32_t *cur_val = d_blcVal + (i * BSR_NNZ / 2);
        fragA[0] = cur_blockid < end ? cur_val[laneid * 2] : 0;
        fragA[1] = cur_blockid < end ? cur_val[laneid * 2 + 1] : 0;

        int xid = cur_blockid < end ? (d_blcCid[cur_blockid] * BSR_N / 2) : (d_blcCid[i] * BSR_N / 2);
        fragB[0] = d_x[xid];
        fragB[1] = d_x[xid + 1];

        mma_m8n8k4_fp16(fragC, fragA, fragB);
    }
    res = fragC[target_idx];

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        int rowid = blc_rid * 4 + laneid;
        if (rowid < row)
            d_y_half[rowid] = res * (half)alpha;
    }
}
void spmv_amgT_fp16(HYPRE_Int trans,
                    HYPRE_Complex alpha,
                    hypre_CSRMatrix *A,
                    hypre_Vector *x,
                    HYPRE_Complex beta,
                    hypre_Vector *y,
                    HYPRE_Int offset)
{
    struct timeval t1, t2;
    gettimeofday1(&t1, NULL);
    // printf("SpMV 0\n");
    CSR2BSR_GPU(A);
    // printf("SpMV 0.1\n");
    BSR_BALANCED_PREPROCESS_GPU(A);
    // printf("SpMV 1\n");
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    bsrMAT *bsrmat = (hypre_BSR(A));
    if (!hypre_VectorSpaceTag(A)) // 判断 A的辅助x和y空间是否已经存在
    {
        cudaMalloc((void **)&bsrmat->dVecX_fp16, sizeof(uint32_t) * ((bsrmat->col + 1) / 2));
        cudaMalloc((void **)&bsrmat->dVecY_fp16, sizeof(uint32_t) * ((bsrmat->row + 1) / 2));
        hypre_VectorSpaceTag(A) = 1;
    }

    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_b = (bsrmat->warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum = (bsrmat->blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum2 = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    int BlockNum_x = (bsrmat->col + ThreadNum - 1) / ThreadNum;
    int BlockNum_y = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    MAT_VAL_TYPE *dvecX = hypre_VectorData(x);
    MAT_VAL_TYPE *dvecY = hypre_VectorData(y);
    if (beta != 1)
    {
        beta_vecY<<<BlockNum2, ThreadNum>>>(dvecY, beta, bsrmat->row);
        cudaDeviceSynchronize();
    }
    vec_64_to_16<<<BlockNum_x, ThreadNum>>>(dvecX, bsrmat->dVecX_fp16, bsrmat->col); // 将double向量中的数赋值到float中

    cudaDeviceSynchronize();

    vec_64_to_16<<<BlockNum_y, ThreadNum>>>(dvecY, bsrmat->dVecY_fp16, bsrmat->row);
    cudaDeviceSynchronize();

    double stand = bsrmat->stand;
    double avgnz = bsrmat->avg_nnz;
    // printf("SpMV 4\n");
    if (!hypre_MixedPrecisionTag(A) || hypre_BSR(A)->blcVal_fp16 == NULL) // 申请额外空间
    {
        int ThreadNum_val_convert = WARP_SIZE * WARP_NUM_SPMV;
        int BlockNum_spgemm_A = (bsrmat->nnz + ThreadNum_val_convert - 1) / ThreadNum_val_convert;
        hypre_MixedPrecisionTag(A) = 1;
        cudaMalloc((void **)&bsrmat->blcVal_fp16, sizeof(uint32_t) * ((bsrmat->nnz + 1) / 2));
        cudaMemset(bsrmat->blcVal_fp16, 0, sizeof(uint32_t) * ((bsrmat->nnz + 1) / 2));
        bsr_val_fp64_to_16<<<BlockNum_spgemm_A, ThreadNum_val_convert>>>(bsrmat->blcVal, bsrmat->blcVal_fp16, bsrmat->nnz);
        cudaDeviceSynchronize();
    }
    gettimeofday1(&t1, NULL);
#ifdef ADAPTIVE_AMGT_SPMV
    // printf("half!!!!!\n");
    if (stand >= 12 && avgnz >= 10)
    {
        // ===tensor core, balanced===
        bsr_spmv_balanced_tc_fp16<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand >= 12 && avgnz < 10)
    {
        // ===cuda core, balanced===
        bsr_spmv_balanced_cc_fp16<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand < 12 && avgnz >= 10)
    {
        // ===tensor core===
        bsr_spmv_tc_fp16<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else
    {
        // ===cuda core===
        bsr_spmv_cc_fp16<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }

#else
    // check value
    bsr_spmv_balanced_tc_fp16<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
    cudaDeviceSynchronize();

#endif
    gettimeofday1(&t2, NULL);
    vec_16_to_64<<<BlockNum2, ThreadNum>>>(dvecY, bsrmat->dVecY_fp16, bsrmat->row);
    cudaDeviceSynchronize();

    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", bsrmat->row);
    printf("spmv_kernel_n=%d\n", bsrmat->col);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;
}

void spmv_amgT_fp32(HYPRE_Int trans,
                    HYPRE_Complex alpha,
                    hypre_CSRMatrix *A,
                    hypre_Vector *x,
                    HYPRE_Complex beta,
                    hypre_Vector *y,
                    HYPRE_Int offset)
{
    struct timeval t1, t2;
    gettimeofday1(&t1, NULL);
    // printf("SpMV 0\n");
    CSR2BSR_GPU(A);
    // printf("SpMV 0.1\n");
    BSR_BALANCED_PREPROCESS_GPU(A);
    // printf("SpMV 1\n");
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    bsrMAT *bsrmat = (hypre_BSR(A));
    if (!hypre_VectorSpaceTag(A)) // 判断 A的辅助x和y空间是否已经存在
    {
        cudaMalloc((void **)&bsrmat->dVecX_fp32, sizeof(float) * bsrmat->col);
        cudaMalloc((void **)&bsrmat->dVecY_fp32, sizeof(float) * bsrmat->row);
        hypre_VectorSpaceTag(A) = 1;
    }

    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_b = (bsrmat->warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum = (bsrmat->blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum2 = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    int BlockNum_x = (bsrmat->col + ThreadNum - 1) / ThreadNum;
    int BlockNum_y = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    MAT_VAL_TYPE *dvecX = hypre_VectorData(x);
    MAT_VAL_TYPE *dvecY = hypre_VectorData(y);
    if (beta != 1)
    {
        beta_vecY<<<BlockNum2, ThreadNum>>>(dvecY, beta, bsrmat->row);
        cudaDeviceSynchronize();
    }
    vec_64_to_32<<<BlockNum_x, ThreadNum>>>(dvecX, bsrmat->dVecX_fp32, bsrmat->col); // 将double向量中的数赋值到float中

    cudaDeviceSynchronize();

    vec_64_to_32<<<BlockNum_y, ThreadNum>>>(dvecY, bsrmat->dVecY_fp32, bsrmat->row);
    cudaDeviceSynchronize();

    double stand = bsrmat->stand;
    double avgnz = bsrmat->avg_nnz;
    // printf("SpMV 4\n");
    if (!hypre_MixedPrecisionTag(A) || hypre_BSR(A)->blcVal_fp32 == NULL) // 申请额外空间
    {
        // printf("no mixed!!!!!!\n");
        int ThreadNum_val_convert = WARP_SIZE * WARP_NUM_SPMV;
        int BlockNum_spgemm_A = (bsrmat->nnz + ThreadNum_val_convert - 1) / ThreadNum_val_convert;
        hypre_MixedPrecisionTag(A) = 1;
        cudaMalloc((void **)&bsrmat->blcVal_fp32, sizeof(float) * bsrmat->nnz);
        cudaMemset(bsrmat->blcVal_fp32, 0, sizeof(float) * bsrmat->nnz);
        bsr_val_fp64_to_32<<<BlockNum_spgemm_A, ThreadNum_val_convert>>>(bsrmat->blcVal, bsrmat->blcVal_fp32, bsrmat->nnz);
        cudaDeviceSynchronize();
    }
    gettimeofday1(&t1, NULL);
#ifdef ADAPTIVE_AMGT_SPMV
    if (stand >= 12 && avgnz >= 10)
    {
        // ===tensor core, balanced===
        bsr_spmv_balanced_tc_fp32<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand >= 12 && avgnz < 10)
    {
        // ===cuda core, balanced===
        bsr_spmv_balanced_cc_fp32<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand < 12 && avgnz >= 10)
    {
        // ===tensor core===
        bsr_spmv_tc_fp32<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else
    {
        // ===cuda core===
        bsr_spmv_cc_fp32<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }

#else
    // check value
    bsr_spmv_balanced_tc_fp32<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, (float)alpha);
    cudaDeviceSynchronize();

#endif
    gettimeofday1(&t2, NULL);
    vec_32_to_64<<<BlockNum2, ThreadNum>>>(dvecY, bsrmat->dVecY_fp32, bsrmat->row);
    cudaDeviceSynchronize();
    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", bsrmat->row);
    printf("spmv_kernel_n=%d\n", bsrmat->col);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;
}

HYPRE_Int
hypre_CSRMatrixMatvecCusparseNewAPI(HYPRE_Int trans,
                                    HYPRE_Complex alpha,
                                    hypre_CSRMatrix *A,
                                    hypre_Vector *x,
                                    HYPRE_Complex beta,
                                    hypre_Vector *y,
                                    HYPRE_Int offset)
{
    spmv_times++;
    // printf("begin: row:%d\tcol:%d\tpreprocess time %lf\t spmv time %lf \n", hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A), time_spmv_preprocess, time_spmv_sum);
    // printf("SpMV begin\n");

    struct timeval t1, t2;
#ifdef Hypre_AMGT

#ifdef MIXED_PRESION
    AMGT_PRECISION precision = get_calculationPrecison(hypre_Level(A));
    // spmv_amgT_fp64(trans, alpha, A, x, beta, y, offset);
    if (precision == AMGT_DOUBLE)
    {
        // printf("DOUBLE\n");
        spmv_amgT_fp64(trans, alpha, A, x, beta, y, offset);
    }
    else if (precision == AMGT_FLOAT)
    {
        // printf("%d\n", hypre_Level(A));
        // printf("FLOAT SpMV\n");
        spmv_amgT_fp32(trans, alpha, A, x, beta, y, offset);

        // printf("End SpMV\n");
    }
    else
    {
        // printf("half spmv\n");
        spmv_amgT_fp16(trans, alpha, A, x, beta, y, offset);
        // printf("half spmv end\n");
    }
#else

    spmv_amgT_fp64(trans, alpha, A, x, beta, y, offset);

#endif

    // printf("time_spmv:%lf\n", time_spmv_sum);
#else
    // #if 0
    gettimeofday1(&t1, NULL);
    HYPRE_Int num_vectors = hypre_VectorNumVectors(x);
    HYPRE_Int num_cols = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
    HYPRE_Int num_rows = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);
    hypre_CSRMatrix *AT;
    hypre_CSRMatrix *B;
    /* SpMV data */
    size_t bufferSize = 0;
    char *dBuffer = hypre_CSRMatrixGPUMatSpMVBuffer(A);
    cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
    const cudaDataType data_type = hypre_HYPREComplexToCudaDataType();
    const cusparseIndexType_t index_type = hypre_HYPREIntToCusparseIndexType();

    /* Local cusparse descriptor variables */
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseDnMatDescr_t matX, matY;

    /* We handle the transpose explicitly to ensure the same output each run
     * and for potential performance improvement memory for AT */
    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &AT, 1);
        B = AT;
    }
    else
    {
        B = A;
    }

    /* Create cuSPARSE vector data structures */
    matA = hypre_CSRMatrixToCusparseSpMat(B, offset);
    if (num_vectors == 1)
    {
        // printf("matrix-vec\n");
        vecX = hypre_VectorToCusparseDnVec(x, 0, num_cols);
        vecY = hypre_VectorToCusparseDnVec(y, offset, num_rows - offset);
    }
    else
    {
        // printf("matrix-matrix\n");
        matX = hypre_VectorToCusparseDnMat(x);
        matY = hypre_VectorToCusparseDnMat(y);
    }

    if (!dBuffer)
    {
        // printf("!dBuffer\n");
        if (num_vectors == 1)
        {
            HYPRE_CUSPARSE_CALL(cusparseSpMV_bufferSize(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        matA,
                                                        vecX,
                                                        &beta,
                                                        vecY,
                                                        data_type,
                                                        HYPRE_CUSPARSE_SPMV_ALG,
                                                        &bufferSize));
        }
        else
        {
            HYPRE_CUSPARSE_CALL(cusparseSpMM_bufferSize(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        matA,
                                                        matX,
                                                        &beta,
                                                        matY,
                                                        data_type,
                                                        HYPRE_CUSPARSE_SPMM_ALG,
                                                        &bufferSize));
        }

        dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
        hypre_CSRMatrixGPUMatSpMVBuffer(A) = dBuffer;

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
        if (num_vectors > 1)
        {
            HYPRE_CUSPARSE_CALL(cusparseSpMM_preprocess(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        matA,
                                                        matX,
                                                        &beta,
                                                        matY,
                                                        data_type,
                                                        HYPRE_CUSPARSE_SPMM_ALG,
                                                        dBuffer));
        }
#endif
    }
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday1(&t1, NULL);
    if (num_vectors == 1)
    {
        HYPRE_CUSPARSE_CALL(cusparseSpMV(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         vecX,
                                         &beta,
                                         vecY,
                                         data_type,
                                         HYPRE_CUSPARSE_SPMV_ALG,
                                         dBuffer));
        // cudaDeviceSynchronize();
    }
    else
    {
        HYPRE_CUSPARSE_CALL(cusparseSpMM(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matX,
                                         &beta,
                                         matY,
                                         data_type,
                                         HYPRE_CUSPARSE_SPMM_ALG,
                                         dBuffer));
        //   cudaDeviceSynchronize();
    }
    gettimeofday1(&t2, NULL);
    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", num_rows);
    printf("spmv_kernel_n=%d\n", num_cols);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;
    //    printf("time_spmv:%lf\n", time_spmv_sum);
    //    printf("end preprocess time %lf\t spmv time %lf\n", time_spmv_preprocess, time_spmv_sum);
    gettimeofday1(&t1, NULL);
#if defined(HYPRE_USING_GPU)
    hypre_SyncComputeStream(hypre_handle());
#endif

    /* Free memory */
    HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
    if (num_vectors == 1)
    {
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecX));
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecY));
    }
    else
    {
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnMat(matX));
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnMat(matY));
    }
    if (trans)
    {
        hypre_CSRMatrixDestroy(AT);
    }
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //    gettimeofday1(&t2, NULL);
    //    time_spmv_sum += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //    printf("time_spmv:%lf\n",time_spmv_sum);

#endif
    return hypre_error_flag;
}

#else // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

HYPRE_Int
hypre_CSRMatrixMatvecCusparseOldAPI(HYPRE_Int trans,
                                    HYPRE_Complex alpha,
                                    hypre_CSRMatrix *A,
                                    hypre_Vector *x,
                                    HYPRE_Complex beta,
                                    hypre_Vector *y,
                                    HYPRE_Int offset)
{

    printf("==============old one \n");
#ifdef HYPRE_BIGINT
#error "ERROR: cusparse old API should not be used when bigint is enabled!"
#endif
    cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
    cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A);
    hypre_CSRMatrix *B;

    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &B, 1);
    }
    else
    {
        B = A;
    }

    HYPRE_CUSPARSE_CALL(hypre_cusparse_csrmv(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             hypre_CSRMatrixNumRows(B) - offset,
                                             hypre_CSRMatrixNumCols(B),
                                             hypre_CSRMatrixNumNonzeros(B),
                                             &alpha,
                                             descr,
                                             hypre_CSRMatrixData(B),
                                             hypre_CSRMatrixI(B) + offset,
                                             hypre_CSRMatrixJ(B),
                                             hypre_VectorData(x),
                                             &beta,
                                             hypre_VectorData(y) + offset));

    if (trans)
    {
        hypre_CSRMatrixDestroy(B);
    }

    return hypre_error_flag;
}

#endif // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

HYPRE_Int
hypre_CSRMatrixMatvecCusparse(HYPRE_Int trans,
                              HYPRE_Complex alpha,
                              hypre_CSRMatrix *A,
                              hypre_Vector *x,
                              HYPRE_Complex beta,
                              hypre_Vector *y,
                              HYPRE_Int offset)
{
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
    /* Luke E: The generic API is techinically supported on 10.1,10.2 as a preview,
     * with Dscrmv being deprecated. However, there are limitations.
     * While in Cuda < 11, there are specific mentions of using csr2csc involving
     * transposed matrix products with dcsrm*,
     * they are not present in SpMV interface.
     */
    hypre_CSRMatrixMatvecCusparseNewAPI(trans, alpha, A, x, beta, y, offset);

#else
    hypre_CSRMatrixMatvecCusparseOldAPI(trans, alpha, A, x, beta, y, offset);
#endif

    return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUSPARSE)

#if defined(HYPRE_USING_ROCSPARSE)
HYPRE_Int
hypre_CSRMatrixMatvecRocsparse(HYPRE_Int trans,
                               HYPRE_Complex alpha,
                               hypre_CSRMatrix *A,
                               hypre_Vector *x,
                               HYPRE_Complex beta,
                               hypre_Vector *y,
                               HYPRE_Int offset)
{
    rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());
    rocsparse_mat_descr descr = hypre_CSRMatrixGPUMatDescr(A);
    rocsparse_mat_info info = hypre_CSRMatrixGPUMatInfo(A);

    hypre_CSRMatrix *B;

    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &B, 1);
    }
    else
    {
        B = A;
    }

    HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrmv(handle,
                                               rocsparse_operation_none,
                                               hypre_CSRMatrixNumRows(B) - offset,
                                               hypre_CSRMatrixNumCols(B),
                                               hypre_CSRMatrixNumNonzeros(B),
                                               &alpha,
                                               descr,
                                               hypre_CSRMatrixData(B),
                                               hypre_CSRMatrixI(B) + offset,
                                               hypre_CSRMatrixJ(B),
                                               info,
                                               hypre_VectorData(x),
                                               &beta,
                                               hypre_VectorData(y) + offset));

    if (trans)
    {
        hypre_CSRMatrixDestroy(B);
    }

    return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)

#if defined(HYPRE_USING_ONEMKLSPARSE)
HYPRE_Int
hypre_CSRMatrixMatvecOnemklsparse(HYPRE_Int trans,
                                  HYPRE_Complex alpha,
                                  hypre_CSRMatrix *A,
                                  hypre_Vector *x,
                                  HYPRE_Complex beta,
                                  hypre_Vector *y,
                                  HYPRE_Int offset)
{
    sycl::queue *compute_queue = hypre_HandleComputeStream(hypre_handle());
    hypre_CSRMatrix *AT;
    oneapi::mkl::sparse::matrix_handle_t matA_handle = hypre_CSRMatrixGPUMatHandle(A);
    hypre_GPUMatDataSetCSRData(A);

    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &AT, 1);
        hypre_GPUMatDataSetCSRData(AT);
        matA_handle = hypre_CSRMatrixGPUMatHandle(AT);
    }

    HYPRE_ONEMKL_CALL(oneapi::mkl::sparse::gemv(*compute_queue,
                                                oneapi::mkl::transpose::nontrans,
                                                alpha,
                                                matA_handle,
                                                hypre_VectorData(x),
                                                beta,
                                                hypre_VectorData(y) + offset)
                          .wait());

    if (trans)
    {
        hypre_CSRMatrixDestroy(AT);
    }

    return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)

#endif // #if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
