#include "utils.h"

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

void spmv(double *best_time, double *time_spmv, bsrMAT *hmatA, MAT_VAL_TYPE *hvecX, MAT_VAL_TYPE *hvecY, MAT_VAL_TYPE alpha, MAT_VAL_TYPE beta)
{
    int row = hmatA->row;
    int col = hmatA->col;

    int blc_row = hmatA->blc_row;
    int blc_col = hmatA->blc_col;

    struct timeval t1, t2;

    bsrMAT dmatA;
    blcMat_cpy_H2D(&dmatA, hmatA);

    int x_len = ((col + BSR_N - 1) / BSR_N) * BSR_N;
    MAT_VAL_TYPE *dvecX, *dvecY;
    cudaMalloc((void **)&dvecX, sizeof(MAT_VAL_TYPE) * x_len);
    cudaMemcpy(dvecX, hvecX, sizeof(MAT_VAL_TYPE) * x_len, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dvecY, sizeof(MAT_VAL_TYPE) * row);
    cudaMemcpy(dvecY, hvecY, sizeof(MAT_VAL_TYPE) * row, cudaMemcpyHostToDevice);

    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum = (blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;

    printf("device - warmup...\n");

    for (int i = 0; i < SpMV_Warm; i++)
        bsr_spmv<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatA.blcVal, dvecX, dvecY, blc_row, blc_col, row, col, alpha, beta);
    cudaDeviceSynchronize();

    printf("device - compute...\n");
    double once_time = 0;

    for (int i = 0; i < SpMV_Repeat; i++)
    {
        cudaMemcpy(dvecY, hvecY, sizeof(MAT_VAL_TYPE) * row, cudaMemcpyHostToDevice);
        gettimeofday(&t1, NULL);
        bsr_spmv<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatA.blcVal, dvecX, dvecY, blc_row, blc_col, row, col, alpha, beta);
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        once_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (once_time < *best_time)
            *best_time = once_time;
        *time_spmv += once_time;
    }
    *time_spmv /= SpMV_Repeat;

    cudaMemcpy(hvecY, dvecY, sizeof(MAT_VAL_TYPE) * row, cudaMemcpyDeviceToHost);

    cudaFree(dmatA.blcPtr);
    cudaFree(dmatA.blcIdx);
    cudaFree(dmatA.blcVal);
    cudaFree(dvecX);
    cudaFree(dvecY);
}

__forceinline__ __device__ int sum_32_shfl_int(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
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

#define BLOCK_NUM_NO_SHARE 300 // memory limit

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
            int offset_cid = BinarySearch2(now_blcIdx, 0, block_nnz, key);
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

#define GET_SIZE 128

__global__ void CSR_TO_BSR_get_4rowptr(int blc_row, int blc_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                                       MAT_PTR_TYPE *csrPtr, int *csrIdx)
{
    int threadid = threadIdx.x;
    int laneid = threadid & (WARP_SIZE - 1);
    int blockid = blockIdx.x;

    __shared__ int mask[GET_SIZE];

    int start1 = csrPtr[(blockid * 4) < csr_row ? (blockid * 4) : csr_row] + threadid;
    int end1 = csrPtr[(blockid * 4 + 1) < csr_row ? (blockid * 4 + 1) : csr_row];

    int start2 = csrPtr[(blockid * 4 + 1) < csr_row ? (blockid * 4 + 1) : csr_row] + threadid;
    int end2 = csrPtr[(blockid * 4 + 2) < csr_row ? (blockid * 4 + 2) : csr_row];

    int start3 = csrPtr[(blockid * 4 + 2) < csr_row ? (blockid * 4 + 2) : csr_row] + threadid;
    int end3 = csrPtr[(blockid * 4 + 3) < csr_row ? (blockid * 4 + 3) : csr_row];

    int start4 = csrPtr[(blockid * 4 + 3) < csr_row ? (blockid * 4 + 3) : csr_row] + threadid;
    int end4 = csrPtr[(blockid * 4 + 4) < csr_row ? (blockid * 4 + 4) : csr_row];

    int sum = 0;

    for (int i = 0; i < csr_col; i += GET_SIZE * 32 * 4)
    {
        int end_col = (i + GET_SIZE * 32 * 4) < csr_col ? (i + GET_SIZE * 32 * 4) : csr_col;
        for (int j = threadid; j < GET_SIZE; j += blockDim.x)
        {
            mask[j] = 0;
        }

        __syncthreads();
        for (; start1 < end1; start1 += blockDim.x)
        {
            int col = csrIdx[start1];
            if (col < end_col)
            {
                int key = (col - i) / BSR_N;
                atomicOr(&(mask[key >> 5]), 1 << (key & 31));
            }
            else
            {
                break;
            }
        }

        for (; start2 < end2; start2 += blockDim.x)
        {
            int col = csrIdx[start2];
            if (col < end_col)
            {
                int key = (col - i) / BSR_N;
                atomicOr(&(mask[key >> 5]), 1 << (key & 31));
            }
            else
            {
                break;
            }
        }

        for (; start3 < end3; start3 += blockDim.x)
        {
            int col = csrIdx[start3];
            if (col < end_col)
            {
                int key = (col - i) / BSR_N;
                atomicOr(&(mask[key >> 5]), 1 << (key & 31));
            }
            else
            {
                break;
            }
        }

        for (; start4 < end4; start4 += blockDim.x)
        {
            int col = csrIdx[start4];
            if (col < end_col)
            {
                int key = (col - i) / BSR_N;
                atomicOr(&(mask[key >> 5]), 1 << (key & 31));
            }
            else
            {
                break;
            }
        }
        __syncthreads();

        for (int j = threadid; j < GET_SIZE; j += blockDim.x)
        {
            unsigned int now = mask[j];
            sum += __popc(now);
        }
        __syncthreads();
    }
    sum = sum_32_shfl_int(sum);
    __syncthreads();
    if (laneid == 0)
    {
        atomicAdd(&blcPtr[blockid], sum);
    }
    return;
}

void CSR_TO_BSR_step1(int bsr_row, int bsr_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                      MAT_PTR_TYPE *csrPtr, MAT_PTR_TYPE *csrIdx)
{

    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum = bsr_row;

    CSR_TO_BSR_get_4rowptr<<<BlockNum, ThreadNum>>>(bsr_row, bsr_col, csr_row, csr_col, blcPtr, csrPtr, csrIdx);
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

__global__ void CSR_TO_CSR_bin_getidx_noshare(int *bin_rowidx, int *bin_offset, int bin,
                                              int blc_row, int blc_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                                              MAT_IDX_TYPE *blcIdx, MAT_VAL_TYPE *blcVal, unsigned short *blcMap,
                                              MAT_PTR_TYPE *csrPtr, MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal,
                                              int *TMP_hashtable, unsigned int *TMP_maptable, int *TMP_nz_num, int SM_SIZE)
{

    int threadid = threadIdx.x;
    int blockid = blockIdx.x;
    int laneid = threadid & 31;
    int bin_row_offset = bin_offset[bin] + blockid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    int *hashtable = TMP_hashtable + blockid * SM_SIZE;
    unsigned int *maptable = TMP_maptable + blockid * SM_SIZE;
    int *nz_num = TMP_nz_num + blockid;

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
            int hashadr = (key < SM_SIZE) ? key : (key - SM_SIZE);
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
                    hashadr = ((hashadr + 1) < SM_SIZE) ? (hashadr + 1) : ((hashadr + 1) - SM_SIZE);
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

double CSR_TO_BSR_step2(int bsr_row, int bsr_col, int csr_row, int csr_col, MAT_PTR_TYPE *blcPtr,
                      MAT_IDX_TYPE *blcIdx, MAT_VAL_TYPE *blcVal, unsigned short *blcMap,
                      MAT_PTR_TYPE *csrPtr, MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal)
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

    //printf("preprocess time %lf ms\n", time);

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
                if(max_len == 0){
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

                int *sum_tmp_memory = NULL;
                cudaMalloc((void **)&sum_tmp_memory, sizeof(MAT_PTR_TYPE) * (BLOCK_NUM_NO_SHARE * ((row_num + 31) >> 5)));
                int every_layer = (row_num + BlockNum - 1) / BlockNum;
                CSR_TO_BSR_getidx_noshare<<<BLOCK_NUM_NO_SHARE, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                             bsr_row, bsr_col, csr_row, csr_col, every_layer, 
                                                                             blcPtr, blcIdx, blcVal, (int *)blcMap, 
                                                                             csrPtr, csrIdx, csrVal, sum_tmp_memory);
                cudaFree(sum_tmp_memory);

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

    cudaFree(bin_offset);
    cudaFree(bin_size);
    cudaFree(bin_rowidx);
    cudaFree(max_num);

    // printf("run time %lf ms\n", time);

    // printf("step preprocess + run time %lf ms\n", sum_time);
    return sum_time;
}

void cuda_CSR2BSR(csrMAT *csrmat, bsrMAT *bsrmat)
{

    double sum_time = 0.0;

    struct timeval t1, t2;

    bsrmat->row = csrmat->row;
    bsrmat->col = csrmat->col;

    bsrmat->blc_row = (bsrmat->row + BSR_M - 1) / BSR_M;
    bsrmat->blc_col = (bsrmat->col + BSR_N - 1) / BSR_N;

    bsrmat->blcPtr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
    memset(bsrmat->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));

    MAT_PTR_TYPE *d_bsrptr = NULL, *d_csrptr = NULL;
    MAT_IDX_TYPE *d_csridx = NULL;
    MAT_VAL_TYPE *d_csrval = NULL;
    cudaMalloc((void **)&d_bsrptr, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
    cudaMalloc((void **)&d_csrptr, sizeof(MAT_PTR_TYPE) * (csrmat->row + 1));
    cudaMalloc((void **)&d_csridx, sizeof(MAT_IDX_TYPE) * csrmat->nnz);
    cudaMalloc((void **)&d_csrval, sizeof(MAT_VAL_TYPE) * csrmat->nnz);

    cudaMemset(d_bsrptr, 0.0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
    cudaMemcpy(d_csrptr, csrmat->RowPtr, sizeof(MAT_PTR_TYPE) * (csrmat->row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csridx, csrmat->ColIdx, sizeof(MAT_IDX_TYPE) * csrmat->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrval, csrmat->Value, sizeof(MAT_VAL_TYPE) * csrmat->nnz, cudaMemcpyHostToDevice);

    gettimeofday(&t1, NULL);
    CSR_TO_BSR_step1(bsrmat->blc_row, bsrmat->blc_col, csrmat->row, csrmat->col, d_bsrptr, d_csrptr, d_csridx);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    sum_time = sum_time + time;
    //printf("time step1 === %.6lf ms\n", time);
    gettimeofday(&t1, NULL);

    thrust::exclusive_scan(thrust::device, d_bsrptr, d_bsrptr + (bsrmat->blc_row + 1), d_bsrptr, 0);
    gettimeofday(&t2, NULL);
    time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    sum_time = sum_time + time;
    //printf("time scan === %.6lf ms\n", time);

    // exit(0);
    cudaMemcpy(&(bsrmat->blc_num), &d_bsrptr[bsrmat->blc_row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    bsrmat->nnz = bsrmat->blc_num * BSR_M * BSR_N;

    bsrmat->blcIdx = (MAT_IDX_TYPE *)malloc(sizeof(MAT_IDX_TYPE) * bsrmat->blc_num);
    bsrmat->blcVal = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
    bsrmat->blcMap = (MAT_MAP_TYPE *)malloc(sizeof(MAT_MAP_TYPE) * bsrmat->blc_num);

    MAT_IDX_TYPE *d_blcIdx = NULL;
    MAT_VAL_TYPE *d_blcVal = NULL;
    MAT_MAP_TYPE *d_blcMap = NULL;

    cudaMalloc((void **)&d_blcIdx, sizeof(MAT_IDX_TYPE) * bsrmat->blc_num);
    cudaMalloc((void **)&d_blcVal, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
    cudaMalloc((void **)&d_blcMap, sizeof(MAT_MAP_TYPE) * (bsrmat->blc_num + 1));

    cudaMemset(d_blcVal, 0, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
    cudaMemset(d_blcMap, 0, sizeof(MAT_MAP_TYPE) * bsrmat->blc_num);

    gettimeofday(&t1, NULL);

    time = CSR_TO_BSR_step2(bsrmat->blc_row, bsrmat->blc_col, csrmat->row, csrmat->col, d_bsrptr,
                     d_blcIdx, d_blcVal, d_blcMap,
                     d_csrptr, d_csridx, d_csrval);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //printf("time step2 === %.6lf ms\n", time);

    sum_time = sum_time + time;

    printf("uscsr2bsr sum time %lf ms\n", sum_time);

    cudaMemcpy(bsrmat->blcPtr, d_bsrptr, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(bsrmat->blcIdx, d_blcIdx, sizeof(MAT_IDX_TYPE) * bsrmat->blc_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(bsrmat->blcVal, d_blcVal, sizeof(MAT_VAL_TYPE) * bsrmat->nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(bsrmat->blcMap, d_blcMap, sizeof(MAT_MAP_TYPE) * bsrmat->blc_num, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < csrmat->row; i++)
    // {
    //     double f=0.0;
    //     for(int j=csrmat->RowPtr[i];j<csrmat->RowPtr[i+1];j++){
    //         f+=csrmat->Value[j];
    //     }
    //     printf("%d %lf\n",i,f);

    // }
    // printf("\n");

    // printf("begin:\n");
    //  for (int i = 0; i < csrmat->row; i++)
    //  {
    //      printf("%d %d\n", i, csrmat->RowPtr[i + 1] - csrmat->RowPtr[i]);
    //  }
    //  for (int i = 0; i < csrmat->row + 1; i++)
    //  {
    //      printf("%d ", csrmat->RowPtr[i]);
    //  }
    //  printf("\n");

    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     printf("%d %d\n", i,csrmat->ColIdx[i]);
    // }
    // printf("\nvalue\n");

    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     printf("%lf ", csrmat->Value[i]);
    // }
    // printf("\n");

    // for (int i = 0; i < bsr_m + 1; i++)
    //  {
    //      printf("%d ", bsrmat->blcPtr[i]);
    //  }
    //  printf("\nidx\n");

    // for (int i = 0; i < bsrmat->blc_num; i++)
    // {
    //     printf("%d ", bsrmat->blcIdx[i]);
    // }
    // printf("\nval\n");

    // for (int i = 0; i < bsrmat->blc_num; i++)
    // {
    //     double now = 0.0;
    //     for(int j=0;j<16;j++){
    //         now+=bsrmat->blcVal[i*16+j];
    //     }
    //     printf("%d %lf\n",i, now);
    // }
    // printf("\nmap\n");

    // for (int i = 0; i < bsrmat->blc_num; i++)
    // {
    //     printf("%d ", bsrmat->blcMap[i]);
    // }
    // printf("\n");

    return;
}

__global__ void BSR_TO_CSR_get_rowptr(int csr_row, int csr_col, int bsr_row, int bsr_col,
                                      MAT_PTR_TYPE *csrPtr, MAT_PTR_TYPE *blcPtr, MAT_MAP_TYPE *blcMap)
{

    __shared__ MAT_MAP_TYPE tmp_blcMap[128]; // every get two block blcmap

    int threadid = threadIdx.x;
    int blockid = blockIdx.x;
    int laneid = threadid & (WARP_SIZE - 1);
    int warpid = threadid >> 5;
    int start = blcPtr[blockid];
    int end = blcPtr[blockid + 1];

    int warp_idx = laneid / 4;
    int warp_offset = laneid % 4;

    int row_nnz_sum = 0;
    for (int blcidx = start; blcidx < end; blcidx += blockDim.x)
    {
        tmp_blcMap[threadid] = ((blcidx + threadid) < end) ? blcMap[blcidx + threadid] : 0;
        __syncthreads();
        int getidend = (blcidx + 128) < end ? 128 : (end - blcidx);
        for (int getid = 0; getid < getidend; getid += 8)
        {
            MAT_MAP_TYPE now_blcMap = tmp_blcMap[getid + warp_idx];
            now_blcMap = now_blcMap >> (warpid * 4 + warp_offset);
            if ((now_blcMap & 1) == 1)
            {
                row_nnz_sum++;
            }
        }
        __syncthreads();
    }

    row_nnz_sum = sum_32_shfl_int(row_nnz_sum);
    __syncthreads();

    if (laneid == 0 && ((blockid * 4 + warpid) < csr_row))
    {
        csrPtr[blockid * 4 + warpid] = row_nnz_sum;
    }
}

__global__ void BSR_TO_CSR_get_getidx(int csr_row, int csr_col, int bsr_row, int bsr_col,
                                      MAT_PTR_TYPE *csrPtr, MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal,
                                      MAT_PTR_TYPE *blcPtr, MAT_IDX_TYPE *blcIdx,
                                      MAT_VAL_TYPE *blcVal, MAT_MAP_TYPE *blcMap)
{

    __shared__ MAT_MAP_TYPE tmp_blcMap[128]; // every get two block blcmap
    __shared__ MAT_IDX_TYPE tmp_blcIdx[128]; // every get two block blcIdx
    __shared__ MAT_IDX_TYPE tmp_csrIdx[128]; // every get two block csrIdx
    __shared__ MAT_VAL_TYPE tmp_csrVal[128]; // every get two block csrVal

    int threadid = threadIdx.x;
    int blockid = blockIdx.x;
    int laneid = threadid & (WARP_SIZE - 1);
    int warpid = threadid >> 5;
    int start = blcPtr[blockid];
    int end = blcPtr[blockid + 1];

    int warp_idx = laneid / 4;
    int warp_offset = laneid % 4;

    int fill_offset = ((blockid * 4 + warpid) < csr_row) ? csrPtr[blockid * 4 + warpid] : 0;

    MAT_IDX_TYPE *now_tmp_csrIdx = tmp_csrIdx + warpid * WARP_SIZE;
    MAT_VAL_TYPE *now_tmp_csrVal = tmp_csrVal + warpid * WARP_SIZE;

    for (int blcidx = start; blcidx < end; blcidx += blockDim.x)
    {
        tmp_blcMap[threadid] = ((blcidx + threadid) < end) ? blcMap[blcidx + threadid] : 0;
        tmp_blcIdx[threadid] = ((blcidx + threadid) < end) ? blcIdx[blcidx + threadid] : 0;
        __syncthreads();
        int getidend = (blcidx + 128) < end ? 128 : (end - blcidx);
        for (int getid = 0; getid < getidend; getid += 8)
        {
            int row_nnz_sum = 0;
            MAT_MAP_TYPE now_blcMap = tmp_blcMap[getid + warp_idx];
            MAT_IDX_TYPE now_blcidx = tmp_blcIdx[getid + warp_idx];
            now_blcMap = now_blcMap >> (warpid * 4 + warp_offset);
            if ((now_blcMap & 1) == 1)
            {
                row_nnz_sum++;
                now_tmp_csrIdx[laneid] = now_blcidx * 4 + warp_offset;
                now_tmp_csrVal[laneid] = blcVal[(blcidx + getid + warp_idx) * 16 + warpid * 4 + warp_offset];
            }
            else
            {
                now_tmp_csrIdx[laneid] = -1;
            }
            __syncthreads();

            int set_offset = 0;
            for (int i = 0; i < laneid; i++)
            {
                if (now_tmp_csrIdx[i] != -1)
                {
                    set_offset++;
                }
            }

            set_offset += fill_offset;
            if ((now_blcMap & 1) == 1)
            {
                csrIdx[set_offset] = now_tmp_csrIdx[laneid];
                csrVal[set_offset] = now_tmp_csrVal[laneid];
            }
            __syncthreads();

            row_nnz_sum = sum_32_shfl_int(row_nnz_sum);
            fill_offset += row_nnz_sum;

            __syncthreads();
        }
    }
}

void BSR_TO_CSR_step1(int csr_row, int csr_col, int bsr_row, int bsr_col, MAT_PTR_TYPE *csrPtr,
                      MAT_PTR_TYPE *blcPtr, MAT_MAP_TYPE *blcMap)
{
    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum = bsr_row;
    BSR_TO_CSR_get_rowptr<<<BlockNum, ThreadNum>>>(csr_row, csr_col, bsr_row, bsr_col, csrPtr, blcPtr, blcMap);
}

void BSR_TO_CSR_step2(int csr_row, int csr_col, int bsr_row, int bsr_col, MAT_PTR_TYPE *csrPtr,
                      MAT_IDX_TYPE *csrIdx, MAT_VAL_TYPE *csrVal,
                      MAT_PTR_TYPE *blcPtr, MAT_IDX_TYPE *blcIdx,
                      MAT_VAL_TYPE *blcVal, MAT_MAP_TYPE *blcMap)
{
    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum = bsr_row;
    BSR_TO_CSR_get_getidx<<<BlockNum, ThreadNum>>>(csr_row, csr_col, bsr_row, bsr_col,
                                                   csrPtr, csrIdx, csrVal,
                                                   blcPtr, blcIdx, blcVal, blcMap);
}

void cuda_BSR2CSR(bsrMAT *bsrmat, csrMAT *csrmat)
{
    struct timeval t1, t2;
    csrmat->row = bsrmat->row;
    csrmat->col = bsrmat->col;

    double sum_time = 0.0;

    MAT_PTR_TYPE *d_csrptr = NULL;
    cudaMalloc((void **)&d_csrptr, sizeof(MAT_PTR_TYPE) * (csrmat->row + 1));
    cudaMemset(d_csrptr, 0.0, sizeof(MAT_PTR_TYPE) * (csrmat->row + 1));

    MAT_PTR_TYPE *d_bsrptr = NULL;
    MAT_IDX_TYPE *d_blcIdx = NULL;
    MAT_VAL_TYPE *d_blcVal = NULL;
    MAT_MAP_TYPE *d_blcMap = NULL;

    cudaMalloc((void **)&d_bsrptr, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
    cudaMalloc((void **)&d_blcIdx, sizeof(MAT_IDX_TYPE) * bsrmat->blc_num);
    cudaMalloc((void **)&d_blcVal, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
    cudaMalloc((void **)&d_blcMap, sizeof(MAT_MAP_TYPE) * bsrmat->blc_num);

    cudaMemcpy(d_bsrptr, bsrmat->blcPtr, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blcMap, bsrmat->blcMap, sizeof(MAT_MAP_TYPE) * bsrmat->blc_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blcIdx, bsrmat->blcIdx, sizeof(MAT_IDX_TYPE) * bsrmat->blc_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blcVal, bsrmat->blcVal, sizeof(MAT_VAL_TYPE) * bsrmat->nnz, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    BSR_TO_CSR_step1(csrmat->row, csrmat->col, bsrmat->blc_row, bsrmat->blc_col, d_csrptr, d_bsrptr, d_blcMap);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    sum_time += time;

    //printf("usbsr2csr step1 %lf ms\n", time);

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    thrust::exclusive_scan(thrust::device, d_csrptr, d_csrptr + (bsrmat->row + 1), d_csrptr, 0);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    sum_time += time;

    //printf("usbsr2csr step2 %lf ms\n", time);

    cudaMemcpy(&(csrmat->nnz), &d_csrptr[csrmat->row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);

    MAT_IDX_TYPE *d_csridx = NULL;
    MAT_VAL_TYPE *d_csrval = NULL;
    cudaMalloc((void **)&d_csridx, sizeof(MAT_IDX_TYPE) * csrmat->nnz);
    cudaMalloc((void **)&d_csrval, sizeof(MAT_VAL_TYPE) * csrmat->nnz);

    // cudaMemcpy(d_csridx, bsrmat->blcIdx, sizeof(MAT_IDX_TYPE) * csrmat->nnz, cudaMemcpyDeviceToHost);

    cudaMemset(d_csrval, 0.0, sizeof(MAT_VAL_TYPE) * csrmat->nnz);

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    BSR_TO_CSR_step2(csrmat->row, csrmat->col, bsrmat->blc_row, bsrmat->blc_col,
                     d_csrptr, d_csridx, d_csrval,
                     d_bsrptr, d_blcIdx, d_blcVal, d_blcMap);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    sum_time += time;

    //printf("usbsr2csr step3 %lf ms\n", time);

    printf("usbsr2csr sum time %lf ms\n", sum_time);

    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     printf("%d %d\n", i, csrmat->ColIdx[i]);
    // }
    // printf("\nvalue\n");

    // for (int i = 0; i < csrmat->row; i++)
    // {
    //     double f = 0.0;
    //     for (int j = csrmat->RowPtr[i]; j < csrmat->RowPtr[i + 1]; j++)
    //     {
    //         f += csrmat->Value[j];
    //     }
    //     printf("%d %lf\n", i, f);
    // }
    // printf("\n");

    MAT_PTR_TYPE *RowPtr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (csrmat->row + 1));
    MAT_IDX_TYPE *ColIdx = (MAT_IDX_TYPE *)malloc(sizeof(MAT_IDX_TYPE) * csrmat->nnz);
    MAT_VAL_TYPE *Value = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * csrmat->nnz);

    cudaMemcpy(RowPtr, d_csrptr, sizeof(MAT_PTR_TYPE) * (csrmat->row + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(ColIdx, d_csridx, sizeof(MAT_IDX_TYPE) * csrmat->nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(Value, d_csrval, sizeof(MAT_VAL_TYPE) * csrmat->nnz, cudaMemcpyDeviceToHost);

    int flag = 0;

    for (int i = 0; i < csrmat->row + 1; i++)
    {
        if (RowPtr[i] != csrmat->RowPtr[i])
        {
            flag = 1;
            //printf("error1 %d %d\n", RowPtr[i], csrmat->RowPtr[i]);
        }
    }
    //printf("\n");

    for (int i = 0; i < csrmat->nnz; i++)
    {
        if (ColIdx[i] != csrmat->ColIdx[i])
        {
            flag = 1;
            //printf("error2 %d %d\n", ColIdx[i], csrmat->ColIdx[i]);
        }
    }
    //printf("\n");

    if(flag == 1){
        printf("bsr2csr error\n");
    }
    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     if (Value[i] != csrmat->Value[i])
    //     {
    //         printf("error3 %lf %lf\n", Value[i], csrmat->Value[i]);
    //     }
    // }
    // printf("\n");

    // printf("begin\n");

    // for(int i=0;i<bsrmat->blc_row+1;i++){
    //     printf("%d ",bsrmat->blcPtr[i]);
    // }

    // printf("\n");

    // for(int i=0;i<bsrmat->blc_row;i++){
    //     printf("next \n");
    //     for(int j=bsrmat->blcPtr[i];j<bsrmat->blcPtr[i+1];j++){
    //         printf("%d ",j);
    //         for(int k=0;k<16;k++){
    //             printf("%lf ",bsrmat->blcVal[j*16+k]);
    //         }
    //         printf("\n");
    //         unsigned short p=bsrmat->blcMap[j];
    //         for(int k=0;k<16;k++){
    //             printf("%d ",p&1);
    //             p=p/2;
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}

#define CUSPARSE_WARMUP 10
#define CUSPARSE_RUN_TIME 100

void cusparse_csr2bsr(csrMAT *csrmat)
{

    int *csrRowPtrA;
    int *csrColIndA;
    double *csrValA;

    cudaMalloc((void **)&csrRowPtrA, sizeof(MAT_PTR_TYPE) * (csrmat->row + 1));
    cudaMalloc((void **)&csrColIndA, sizeof(MAT_IDX_TYPE) * csrmat->nnz);
    cudaMalloc((void **)&csrValA, sizeof(MAT_VAL_TYPE) * csrmat->nnz);

    cudaMemcpy(csrRowPtrA, csrmat->RowPtr, sizeof(MAT_PTR_TYPE) * (csrmat->row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColIndA, csrmat->ColIdx, sizeof(MAT_IDX_TYPE) * csrmat->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(csrValA, csrmat->Value, sizeof(MAT_VAL_TYPE) * csrmat->nnz, cudaMemcpyHostToDevice);

    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseMatDescr_t descrC;
    cusparseCreateMatDescr(&descrC);

    int nnzb;

    int *bsrRowPtrC = NULL;
    int *bsrColIndC = NULL;
    double *bsrValC = NULL;

    int m = csrmat->row;
    int n = csrmat->col;

    int blockDim = 4;
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    struct timeval t1, t2;
    int mb = (csrmat->row + blockDim - 1) / blockDim;
    int nb = (csrmat->col + blockDim - 1) / blockDim;
    cudaMalloc((void **)&bsrRowPtrC, sizeof(int) * (mb + 1));

    cudaDeviceSynchronize();

    for (int i = 0; i < CUSPARSE_WARMUP; i++)
    {
        cusparseXcsr2bsrNnz(handle, dirA, m, n,
                            descrA, csrRowPtrA, csrColIndA, blockDim,
                            descrC, bsrRowPtrC, &nnzb);

        cudaDeviceSynchronize();
    }

    gettimeofday(&t1, NULL);

    for (int i = 0; i < CUSPARSE_RUN_TIME; i++)
    {
        cusparseXcsr2bsrNnz(handle, dirA, m, n,
                            descrA, csrRowPtrA, csrColIndA, blockDim,
                            descrC, bsrRowPtrC, &nnzb);

        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    cudaMalloc((void **)&bsrColIndC, sizeof(int) * nnzb);
    cudaMalloc((void **)&bsrValC, sizeof(double) * (blockDim * blockDim) * nnzb);

    for (int i = 0; i < CUSPARSE_WARMUP; i++)
    {
        cusparseDcsr2bsr(handle, dirA, m, n,
                         descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
                         descrA, bsrValC, bsrRowPtrC, bsrColIndC);

        cudaDeviceSynchronize();
    }

    gettimeofday(&t1, NULL);

    for (int i = 0; i < CUSPARSE_RUN_TIME; i++)
    {
        cusparseDcsr2bsr(handle, dirA, m, n,
                         descrA, csrValA, csrRowPtrA, csrColIndA, blockDim,
                         descrA, bsrValC, bsrRowPtrC, bsrColIndC);

        cudaDeviceSynchronize();
    }

    gettimeofday(&t2, NULL);

    time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    MAT_PTR_TYPE *c_bsrptr = (int *)malloc(sizeof(int)*(mb+1));
    MAT_IDX_TYPE *c_blcIdx = (int *)malloc(sizeof(int)*nnzb);
    MAT_VAL_TYPE *c_blcVal = (double *)malloc(sizeof(double)*nnzb*(blockDim*blockDim));

    cudaMemcpy(c_bsrptr, bsrRowPtrC, sizeof(MAT_PTR_TYPE) * (mb + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_blcIdx, bsrColIndC, sizeof(MAT_IDX_TYPE) * nnzb, cudaMemcpyDeviceToHost);
    cudaMemcpy(c_blcVal, bsrValC, sizeof(MAT_VAL_TYPE) *nnzb*(blockDim*blockDim), cudaMemcpyDeviceToHost);

    printf("\ncusparse csr2bsr time : %lf ms\n", time / ((double)CUSPARSE_RUN_TIME));

    cusparseCreate(&handle);

    //printf("nnzb = %d\n", nnzb);

    int *csrRowPtrB;
    int *csrColIndB;
    double *csrValB;

    cudaMalloc((void **)&csrRowPtrB, sizeof(MAT_PTR_TYPE) * ( (mb * 4) + 1));
    cudaMalloc((void **)&csrColIndB, sizeof(MAT_IDX_TYPE) * nnzb * blockDim * blockDim);
    cudaMalloc((void **)&csrValB, sizeof(MAT_VAL_TYPE) * nnzb * blockDim * blockDim);

    cusparseMatDescr_t descrB;
    cusparseCreateMatDescr(&descrB);

    cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    cusparseHandle_t handle1 = NULL;
    cusparseCreate(&handle1);

    for (int i = 0; i < CUSPARSE_WARMUP; i++)
    {
        cusparseDbsr2csr(handle1, dir, mb, nb,
                         descrC,
                         bsrValC, bsrRowPtrC, bsrColIndC,
                         blockDim,
                         descrB,
                         csrValB, csrRowPtrB, csrColIndB);

        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    for (int i = 0; i < CUSPARSE_RUN_TIME; i++)
    {
        cusparseDbsr2csr(handle1, dir, mb, nb,
                         descrC,
                         bsrValC, bsrRowPtrC, bsrColIndC,
                         blockDim,
                         descrB,
                         csrValB, csrRowPtrB, csrColIndB);

        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cusparse bsr2csr time : %lf ms\n", time / ((double)CUSPARSE_RUN_TIME));


    MAT_PTR_TYPE *b_csrptr = (int *)malloc(sizeof(int)*((mb * 4) + 1));
    MAT_IDX_TYPE *b_clcIdx = (int *)malloc(sizeof(int)*nnzb * blockDim * blockDim);
    MAT_VAL_TYPE *b_clcVal = (double *)malloc(sizeof(double)*nnzb * blockDim * blockDim);

    cudaMemcpy(b_csrptr, csrRowPtrB, sizeof(MAT_PTR_TYPE) *  ((mb * 4) + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_clcIdx, csrColIndB, sizeof(MAT_IDX_TYPE) * nnzb * blockDim * blockDim, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_clcVal, csrValB, sizeof(MAT_VAL_TYPE) *nnzb * blockDim * blockDim, cudaMemcpyDeviceToHost);


    // printf("rowptr:\n");
    // for (int i = 0; i < ((mb * 4) + 1); i++)
    // {
    //     printf("%d ",b_csrptr[i]);
    // }
    // printf("colidx:\n");
    // for (int i = 0; i < nnzb * blockDim * blockDim; i++)
    // {
    //     printf("%d ",b_clcIdx[i]);
    // }
    // printf("value:\n");
    // for (int i = 0; i < nnzb * blockDim * blockDim; i++)
    // {
    //     printf("%lf ",b_clcVal[i]);
    // }
    // printf("\n");
    // printf("rowptr:\n");
    // for (int i = 0; i < csrmat->row + 1; i++)
    // {
    //     printf("%d ",csrmat->RowPtr[i]);
    // }
    // printf("colidx:\n");
    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     printf("%d ",csrmat->ColIdx[i]);
    // }
    // printf("value:\n");
    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     printf("%lf ",csrmat->Value[i]);
    // }
    // printf("\n");
    // int flag = 0;

    // for (int i = 0; i < csrmat->row + 1; i++)
    // {
    //     if (csrmat->RowPtr[i] != b_csrptr[i])
    //     {
    //         flag = 1;
    //         //printf("error1 %d %d %d\n", i, bsr_matA.blcPtr[i], bsr_matA1.blcPtr[i]);
    //     }
    // }
    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     if (csrmat->ColIdx[i] != b_clcIdx[i])
    //     {
    //         flag = 1;
    //         //printf("error2 %d %d %d\n", i, bsr_matA.blcIdx[i], bsr_matA1.blcIdx[i]);
    //     }
    // }
    // for (int i = 0; i < csrmat->nnz; i++)
    // {
    //     if (csrmat->Value[i] != b_clcVal[i])
    //     {
    //         flag = 1;
    //         //printf("error4 %d %lf %lf\n", i, bsr_matA.blcVal[i], bsr_matA1.blcVal[i]);
    //     }
    // }
    // if(flag == 1){
    //     printf("error cusparse translate\n");
    // }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Run the code by './test_sparse matrixA.mtx(cbd) matrixB.mtx(cbd) ischeck'.\n");
        return 0;
    }

    char *filename;
    filename = argv[1];
    int ischeck = atoi(argv[2]);

    MAT_VAL_TYPE alpha = 3.0;
    MAT_VAL_TYPE beta = 0.0;

    // read matrix
    csrMAT csr_matA;
    readMatrix(&csr_matA, filename);
    for (int i = 0; i < csr_matA.nnz; i++)
    {
        csr_matA.Value[i] = (double)i;
    }

    printf("read Matrix\n");
    // csr_matA.col=20398;
    // //csr_matA.row=60;
    printf("%d %d\n", csr_matA.row, csr_matA.col);
    // format convert: csr -> bsr
    bsrMAT bsr_matA;
    bsrMAT bsr_matA1;
    cuda_CSR2BSR(&csr_matA, &bsr_matA);
    // CSR2BSR(&csr_matA, &bsr_matA1);

    int flag = 0;
    // for (int i = 0; i < bsr_matA1.blc_row + 1; i++)
    // {
    //     if (bsr_matA.blcPtr[i] != bsr_matA1.blcPtr[i])
    //     {
    //         flag = 1;
    //         //printf("error1 %d %d %d\n", i, bsr_matA.blcPtr[i], bsr_matA1.blcPtr[i]);
    //     }
    // }
    // for (int i = 0; i < bsr_matA1.blc_num; i++)
    // {
    //     if (bsr_matA.blcIdx[i] != bsr_matA1.blcIdx[i])
    //     {
    //         flag = 1;
    //         //printf("error2 %d %d %d\n", i, bsr_matA.blcIdx[i], bsr_matA1.blcIdx[i]);
    //     }
    // }
    // for (int i = 0; i < bsr_matA1.blc_num; i++)
    // {
    //     if (bsr_matA.blcMap[i] != bsr_matA1.blcMap[i])
    //     {
    //         flag = 1;
    //         // printf("error3 %d ", i);
    //         // unsigned short now = bsr_matA.blcMap[i];
    //         // for (int j = 0; j < 16; j++)
    //         // {
    //         //     printf("%d", now & 1);
    //         //     now = now / 2;
    //         // }
    //         // printf(" ");
    //         // now = bsr_matA1.blcMap[i];
    //         // for (int j = 0; j < 16; j++)
    //         // {
    //         //     printf("%d", now & 1);
    //         //     now = now / 2;
    //         // }
    //         // printf("\n");
    //     }
    // }
    // for (int i = 0; i < bsr_matA1.blc_num * 4; i++)
    // {
    //     if (bsr_matA.blcVal[i] != bsr_matA1.blcVal[i])
    //     {
    //         flag = 1;
    //         //printf("error4 %d %lf %lf\n", i, bsr_matA.blcVal[i], bsr_matA1.blcVal[i]);
    //     }
    // }

    if(flag == 1){
        printf("csr2bsr error\n");
    }

    cuda_BSR2CSR(&bsr_matA, &csr_matA);
    cusparse_csr2bsr(&csr_matA);

    return 0;
    printf("format convert\n");

    int x_len = ((csr_matA.col + BSR_N - 1) / BSR_N) * BSR_N;
    MAT_VAL_TYPE *vecX = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * x_len);
    init_vector(vecX, csr_matA.col);
    for (int i = csr_matA.col; i < x_len; i++)
        vecX[i] = 0;
    MAT_VAL_TYPE *vecY = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * csr_matA.row);
    MAT_VAL_TYPE *cu_vecY = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * csr_matA.row);

    long long data_csr = (csr_matA.row + csr_matA.col + csr_matA.nnz) * sizeof(MAT_VAL_TYPE) + (csr_matA.row + 1) * sizeof(MAT_PTR_TYPE) + csr_matA.nnz * sizeof(MAT_IDX_TYPE);
    long long data_bsr = (bsr_matA.row + bsr_matA.col + bsr_matA.nnz) * sizeof(MAT_VAL_TYPE) +
                         (bsr_matA.blc_row + 1) * sizeof(MAT_PTR_TYPE) + bsr_matA.blc_num * sizeof(MAT_IDX_TYPE);

    printf("cuSPARSE SpMV computing ...\n");
    double time_cu = 0, best_time_cu = 9999;
    cusparse_spmv_all(csr_matA.Value, csr_matA.RowPtr, csr_matA.ColIdx, vecX, cu_vecY, csr_matA.row, csr_matA.col, csr_matA.nnz, alpha, beta, &time_cu, &best_time_cu);

    double time_spmv = 0, best_time = 9999;
    // int id = 32;
    printf("double precision spmv \n");
    spmv(&best_time, &time_spmv, &bsr_matA, vecX, vecY, alpha, beta);

    double gflops_cu = (double)((long)csr_matA.nnz * 2) / (best_time_cu * 1e6);
    double gflops_cu_avg = (double)((long)csr_matA.nnz * 2) / (time_cu * 1e6);
    double bdw_cu = (double)data_csr / (best_time_cu * 1e6);
    double gflops_spmv = (double)((long)csr_matA.nnz * 2) / (best_time * 1e6);
    double gflops_spmv_avg = (double)((long)csr_matA.nnz * 2) / (time_spmv * 1e6);
    double bdw_spmv = (double)data_bsr / (best_time * 1e6);

    printf("cuSpMV: %.2lf ms, %.2lf Gflops, %.2lf Gflops, %.2lf GB/s\n", best_time_cu, gflops_cu, gflops_cu_avg, bdw_cu);
    printf("usSpMV: %.2lf ms, %.2lf Gflops, %.2lf Gflops, %.2lf GB/s\n", best_time, gflops_spmv, gflops_spmv_avg, bdw_spmv);

    printf("checking...\n");
    if (ischeck)
    {
        MAT_VAL_TYPE *ref_vecY = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * csr_matA.row);
        memset(ref_vecY, 0, sizeof(MAT_VAL_TYPE) * csr_matA.row);
        int i;
        for (i = 0; i < csr_matA.row; i++)
        {
            MAT_VAL_TYPE sum = 0;
            for (int j = csr_matA.RowPtr[i]; j < csr_matA.RowPtr[i + 1]; j++)
            {
                sum += csr_matA.Value[j] * vecX[csr_matA.ColIdx[j]];
            }
            ref_vecY[i] = alpha * sum + beta * ref_vecY[i];

            double temp_cuda_val = vecY[i];
            // double temp_cuda_val = cu_vecY[i];
            double temp_ref_val = ref_vecY[i];

            if (fabs(temp_ref_val - temp_cuda_val) > 1e-1)
            {
                printf("error in row %d, refY=%lf, Y=%lf\n", i, temp_ref_val, temp_cuda_val);
                break;
            }
        }

        if (i == csr_matA.row)
            printf("result correct!\n");
        free(ref_vecY);
    }

    release_host_csrMAT(csr_matA);
    release_host_bsrMAT(bsr_matA);
    free(vecX);
    free(vecY);
    free(cu_vecY);
    fflush(NULL);
    return 0;
}