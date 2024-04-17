/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "seq_mv.hpp"
#include <sys/time.h>
double time_spgemm = 0;
double time_spgemm_preprocess = 0;
int spgemm_times = 0;
double bsr2csr_step1 = 0;
double bsr2csr_step2 = 0;
double bsr2csr_step3 = 0;

#if defined(HYPRE_USING_GPU)

__forceinline__ __device__ int sum_32_shfl_int(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
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
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]) : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
}
__device__ __forceinline__ void mma_m16n8k4_tf32(float *acc, uint32_t *frag_a, uint32_t *frag_b)
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
__device__ __forceinline__ void mma_m8n8k4_SpGEMM(MAT_VAL_TYPE *acc, MAT_VAL_TYPE &frag_a, MAT_VAL_TYPE &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}
__device__ __host__ int BinarySearch2_SpGEMM(int *arr, int left, int right, int target)
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
__global__ void compute_Cub_bin(MAT_PTR_TYPE *BlcPtrA, int *BlcCidA, MAT_PTR_TYPE *BlcPtrB,
                                MAT_PTR_TYPE *BlcCub, int m, int *bin_offset)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rowid = global_tid;
    if (rowid >= m)
        return;

    int start_blc = BlcPtrA[rowid];
    int end_blc = BlcPtrA[rowid + 1];

    int cur_Cub = 0;
    for (int i = start_blc; i < end_blc; i++)
    {
        int cidA = BlcCidA[i];
        cur_Cub += BlcPtrB[cidA + 1] - BlcPtrB[cidA];
    }
    BlcCub[rowid] = cur_Cub;

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
    else if (cur_Cub >= 4096 && cur_Cub < 8192)
    {
        atomicAdd(&bin_offset[6], 1);
    }
    else
    {
        atomicAdd(&bin_offset[7], 1);
    }
    __syncthreads();
}

__global__ void set_bin(int m, MAT_PTR_TYPE *BlcCub, MAT_IDX_TYPE *bin_rowidx, int *bin_offset, int *bin_size, int *max_num)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rowid = global_tid;
    if (rowid >= m)
        return;

    int cur_Cub = BlcCub[rowid];
    int idx = 0;

    if (cur_Cub < 128)
    {
        idx = atomicAdd(&bin_size[0], 1);
        bin_rowidx[bin_offset[0] + idx] = rowid;
    }
    else if (cur_Cub >= 128 && cur_Cub < 256)
    {
        idx = atomicAdd(&bin_size[1], 1);
        bin_rowidx[bin_offset[1] + idx] = rowid;
    }
    else if (cur_Cub >= 256 && cur_Cub < 512)
    {
        idx = atomicAdd(&bin_size[2], 1);
        bin_rowidx[bin_offset[2] + idx] = rowid;
    }
    else if (cur_Cub >= 512 && cur_Cub < 1024)
    {
        idx = atomicAdd(&bin_size[3], 1);
        bin_rowidx[bin_offset[3] + idx] = rowid;
    }
    else if (cur_Cub >= 1024 && cur_Cub < 2048)
    {
        idx = atomicAdd(&bin_size[4], 1);
        bin_rowidx[bin_offset[4] + idx] = rowid;
    }
    else if (cur_Cub >= 2048 && cur_Cub < 4096)
    {
        idx = atomicAdd(&bin_size[5], 1);
        bin_rowidx[bin_offset[5] + idx] = rowid;
    }
    else if (cur_Cub >= 4096 && cur_Cub < 8192)
    {
        idx = atomicAdd(&bin_size[6], 1);
        bin_rowidx[bin_offset[6] + idx] = rowid;
    }
    else
    {
        idx = atomicAdd(&bin_size[7], 1);
        bin_rowidx[bin_offset[7] + idx] = rowid;
        atomicMax(max_num, cur_Cub);
    }
}

__device__ __host__ unsigned short bitMatrixMul(unsigned short bitMap_1, unsigned short bitMap_2)
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
template <int SM_SIZE>
__global__ void symbolic_spgemm_step1(int *bin_rowidx, int *bin_offset, int bin,
                                      MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA,
                                      MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB,
                                      MAT_PTR_TYPE *BlcPtrC, int m, int n, int k)
{
    int bid = blockIdx.x;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    int rowid = bin_rowidx[bin_row_offset];

    __shared__ int hashtable[SM_SIZE];

    int warpid = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    for (int i = threadIdx.x; i < SM_SIZE; i += blockDim.x)
    {
        hashtable[i] = -1;
    }
    __syncthreads();

    int start_blc = BlcPtrA[rowid];
    int end_blc = BlcPtrA[rowid + 1];

    MAT_PTR_TYPE local_nz_num = 0;

    for (int i = start_blc + warpid; i < end_blc; i += WARP_NUM_SPGM)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        for (int j = BlcPtrB[cidA] + laneid; j < BlcPtrB[cidA + 1]; j += WARP_SIZE)
        {
            MAT_IDX_TYPE cidB = BlcCidB[j];
            MAT_MAP_TYPE mapB = BlcMapB[j];

            MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);

            if (mapC)
            {
                const int key = cidB;
                int hashadr = key & (SM_SIZE - 1);
                while (1)
                {
                    int keyexist = hashtable[hashadr];
                    if (keyexist == key)
                        break;
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
                            local_nz_num++;
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
    }
    __syncthreads();

    local_nz_num = warpReduceSum(local_nz_num);
    if (laneid == 0)
    {
        atomicAdd(BlcPtrC + rowid, local_nz_num);
    }
}

template <int SM_SIZE>
__global__ void symbolic_spgemm_step1_large(int *bin_rowidx, int *bin_offset, int bin,
                                            int *over_num, int *over_rid,
                                            MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA,
                                            MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB,
                                            MAT_PTR_TYPE *BlcPtrC, int m, int n, int k)
{
    int bid = blockIdx.x;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    int rowid = bin_rowidx[bin_row_offset];

    __shared__ int hashtable[SM_SIZE];
    __shared__ int check_num[1];

    int warpid = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    for (int i = threadIdx.x; i < SM_SIZE; i += blockDim.x)
    {
        hashtable[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        check_num[0] = 0;
    }
    __syncthreads();

    int start_blc = BlcPtrA[rowid];
    int end_blc = BlcPtrA[rowid + 1];

    int edge = SM_SIZE * 0.75;

    for (int i = start_blc + warpid; i < end_blc; i += WARP_NUM_SPGM)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        for (int j = BlcPtrB[cidA] + laneid; j < BlcPtrB[cidA + 1]; j += WARP_SIZE)
        {
            MAT_IDX_TYPE cidB = BlcCidB[j];
            MAT_MAP_TYPE mapB = BlcMapB[j];

            MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);

            if (mapC)
            {
                const int key = cidB;
                int hashadr = key & (SM_SIZE - 1);
                while (check_num[0] < edge)
                {
                    int keyexist = hashtable[hashadr];
                    if (keyexist == key)
                        break;
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
                            atomicAdd(&check_num[0], 1);
                            break;
                        }
                    }
                    else
                    {
                        hashadr = (hashadr + 1) & (SM_SIZE - 1);
                    }
                }
            }
            if (check_num[0] >= edge)
                break;
        }
        if (check_num[0] >= edge)
            break;
    }
    __syncthreads();

    if (check_num[0] >= edge)
    {
        if (threadIdx.x == 0)
        {
            int id = atomicAdd(over_num, 1);
            over_rid[id] = rowid;
        }
    }
    else
    {
        if (threadIdx.x == 0)
            BlcPtrC[rowid] = check_num[0];
    }
}

__global__ void symbolic_spgemm_step1_gl(int *over_rowidx, int *hashtable, int hashsize,
                                         MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA,
                                         MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB,
                                         MAT_PTR_TYPE *BlcPtrC, int m, int n, int k)
{
    int bid = blockIdx.x;
    int rowid = over_rowidx[bid];

    int warpid = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    int start_blc = BlcPtrA[rowid];
    int end_blc = BlcPtrA[rowid + 1];

    MAT_PTR_TYPE local_nz_num = 0;

    int *cur_hashtable = hashtable + hashsize * bid;

    for (int i = start_blc + warpid; i < end_blc; i += WARP_NUM_SPGM)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        for (int j = BlcPtrB[cidA] + laneid; j < BlcPtrB[cidA + 1]; j += WARP_SIZE)
        {
            MAT_IDX_TYPE cidB = BlcCidB[j];
            MAT_MAP_TYPE mapB = BlcMapB[j];

            MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);

            if (mapC)
            {
                const int key = cidB;
                int hashadr = key % hashsize;
                while (1)
                {
                    int keyexist = cur_hashtable[hashadr];
                    if (keyexist == key)
                        break;
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(cur_hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
                            local_nz_num++;
                            break;
                        }
                    }
                    else
                    {
                        hashadr = (hashadr + 1) % hashsize;
                    }
                }
            }
        }
    }
    __syncwarp();

    local_nz_num = warpReduceSum(local_nz_num);

    if (laneid == 0)
    {
        atomicAdd(BlcPtrC + rowid, local_nz_num);
    }
}

template <int SM_SIZE>
__global__ void symbolic_spgemm_step2(int *bin_rowidx, int *bin_offset, int bin,
                                      MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA,
                                      MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB,
                                      MAT_PTR_TYPE *BlcPtrC, MAT_IDX_TYPE *BlcCidC, int m, int n, int k)
{
    int bid = blockIdx.x;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    int rowid = bin_rowidx[bin_row_offset];

    __shared__ MAT_IDX_TYPE hashtable[SM_SIZE];
    __shared__ MAT_PTR_TYPE nz_num[1];

    int warpid = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    for (int i = threadIdx.x; i < SM_SIZE; i += blockDim.x)
    {
        hashtable[i] = -1;
    }
    if (threadIdx.x == 0)
        nz_num[0] = 0;
    __syncthreads();

    int start_blc = BlcPtrA[rowid];
    int end_blc = BlcPtrA[rowid + 1];

    for (int i = start_blc + warpid; i < end_blc; i += WARP_NUM_SPGM)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        for (int j = BlcPtrB[cidA] + laneid; j < BlcPtrB[cidA + 1]; j += WARP_SIZE)
        {
            MAT_IDX_TYPE cidB = BlcCidB[j];
            MAT_MAP_TYPE mapB = BlcMapB[j];

            MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);

            if (mapC)
            {
                const int key = cidB;
                int hashadr = key & (SM_SIZE - 1);
                while (1)
                {
                    int keyexist = hashtable[hashadr];
                    if (keyexist == key)
                        break;
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
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
    }
    __syncthreads();

    if (threadIdx.x < WARP_SIZE)
    {
        for (int i = laneid; i < SM_SIZE; i += WARP_SIZE)
        {
            if (hashtable[i] != -1)
            {
                int ind = atomicAdd(&nz_num[0], 1);
                hashtable[ind] = hashtable[i];
            }
        }
    }
    __syncthreads();

    int len = nz_num[0];

    int offset = BlcPtrC[rowid];
    int target, count;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        target = hashtable[i];
        count = 0;
        for (int j = 0; j < len; j++)
        {
            count += (unsigned int)(hashtable[j] - target) >> 31;
        }
        BlcCidC[offset + count] = target;
    }
}

template <int SM_SIZE>
__global__ void symbolic_spgemm_step2_large(int *bin_rowidx, int *bin_offset, int bin,
                                            int *over_num, int *over_rid,
                                            MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA,
                                            MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB,
                                            MAT_PTR_TYPE *BlcPtrC, MAT_IDX_TYPE *BlcCidC, int m, int n, int k)
{
    int bid = blockIdx.x;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    int rowid = bin_rowidx[bin_row_offset];

    __shared__ MAT_IDX_TYPE hashtable[SM_SIZE];
    __shared__ MAT_PTR_TYPE nz_num[1];
    __shared__ int check_num[1];

    int warpid = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    for (int i = threadIdx.x; i < SM_SIZE; i += blockDim.x)
    {
        hashtable[i] = -1;
    }
    if (threadIdx.x == 0)
    {
        nz_num[0] = 0;
        check_num[0] = 0;
    }
    __syncthreads();

    int start_blc = BlcPtrA[rowid];
    int end_blc = BlcPtrA[rowid + 1];

    int edge = SM_SIZE * 0.75;
    // int loopnum = 0;

    for (int i = start_blc + warpid; i < end_blc; i += WARP_NUM_SPGM)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        for (int j = BlcPtrB[cidA] + laneid; j < BlcPtrB[cidA + 1]; j += WARP_SIZE)
        {
            MAT_IDX_TYPE cidB = BlcCidB[j];
            MAT_MAP_TYPE mapB = BlcMapB[j];

            MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);

            if (mapC)
            {
                const int key = cidB;
                int hashadr = key & (SM_SIZE - 1);
                // while(check_num[0] < edge && loopnum < edge)
                while (check_num[0] < edge)
                {
                    int keyexist = hashtable[hashadr];
                    if (keyexist == key)
                        break;
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
                            atomicAdd(&check_num[0], 1);
                            break;
                        }
                    }
                    else
                    {
                        hashadr = (hashadr + 1) & (SM_SIZE - 1);
                        // loopnum ++;
                    }
                }
            }
            if (check_num[0] >= edge)
                break;
        }
        if (check_num[0] >= edge)
            break;
    }
    __syncthreads();

    if (check_num[0] >= edge)
    {
        if (threadIdx.x == 0)
        {
            int id = atomicAdd(over_num, 1);
            over_rid[id] = rowid;
        }
    }
    else
    {
        if (threadIdx.x < WARP_SIZE)
        {
            for (int i = laneid; i < SM_SIZE; i += WARP_SIZE)
            {
                if (hashtable[i] != -1)
                {
                    int ind = atomicAdd(&nz_num[0], 1);
                    hashtable[ind] = hashtable[i];
                }
            }
        }
        __syncthreads();

        int len = nz_num[0];

        int offset = BlcPtrC[rowid];
        int target, count;
        for (int i = threadIdx.x; i < len; i += blockDim.x)
        {
            target = hashtable[i];
            count = 0;
            for (int j = 0; j < len; j++)
            {
                count += (unsigned int)(hashtable[j] - target) >> 31;
            }
            BlcCidC[offset + count] = target;
        }
    }
}

__global__ void symbolic_spgemm_step2_gl(int *over_rowidx, int *hashtable, int hashsize,
                                         MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA,
                                         MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB,
                                         MAT_PTR_TYPE *BlcPtrC, MAT_IDX_TYPE *BlcCidC, int m, int n, int k)
{
    int bid = blockIdx.x;
    int rowid = over_rowidx[bid];

    __shared__ MAT_PTR_TYPE nz_num[1];

    int warpid = threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    if (threadIdx.x == 0)
        nz_num[0] = 0;
    __syncthreads();

    int start_blc = BlcPtrA[rowid];
    int end_blc = BlcPtrA[rowid + 1];

    int *cur_hashtable = hashtable + hashsize * bid;

    for (int i = start_blc + warpid; i < end_blc; i += WARP_NUM_SPGM)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        for (int j = BlcPtrB[cidA] + laneid; j < BlcPtrB[cidA + 1]; j += WARP_SIZE)
        {
            MAT_IDX_TYPE cidB = BlcCidB[j];
            MAT_MAP_TYPE mapB = BlcMapB[j];

            MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);

            if (mapC)
            {
                const int key = cidB;
                int hashadr = key % hashsize;
                while (1)
                {
                    int keyexist = cur_hashtable[hashadr];
                    if (keyexist == key)
                        break;
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(cur_hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
                            break;
                        }
                    }
                    else
                    {
                        hashadr = (hashadr + 1) % hashsize;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < WARP_SIZE)
    {
        for (int i = laneid; i < hashsize; i += WARP_SIZE)
        {
            if (cur_hashtable[i] != -1)
            {
                int ind = atomicAdd(&nz_num[0], 1);
                cur_hashtable[ind] = cur_hashtable[i];
            }
        }
    }
    __syncthreads();

    int len = nz_num[0];

    int offset = BlcPtrC[rowid];
    int target, count;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        target = cur_hashtable[i];
        count = 0;
        for (int j = 0; j < len; j++)
        {
            count += (unsigned int)(cur_hashtable[j] - target) >> 31;
        }
        BlcCidC[offset + count] = target;
    }
}
__global__ void numeric_spgemm_hybrid_f16(MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA, uint32_t *BlcValA,
                                          MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB, uint32_t *BlcValB,
                                          MAT_PTR_TYPE *BlcPtrC, MAT_IDX_TYPE *BlcCidC, MAT_MAP_TYPE *BlcMapC, uint32_t *BlcValC,
                                          int m, int n, int k)
{
    int gl_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = gl_tid / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    int rowid = warpid;
    if (rowid >= m)
        return;

    half *half_valA = reinterpret_cast<half *>(&BlcValA[0]);
    half *half_valB = reinterpret_cast<half *>(&BlcValB[0]);
    half *half_valC = reinterpret_cast<half *>(&BlcValC[0]);

    MAT_IDX_TYPE *curBlcCid = BlcCidC + BlcPtrC[rowid];
    MAT_MAP_TYPE *curBlcMap = BlcMapC + BlcPtrC[rowid];
    half *curBlcVal = half_valC + (BlcPtrC[rowid] * BSR_NNZ);

    int start = BlcPtrA[rowid];
    int end = BlcPtrA[rowid + 1];

    uint32_t fragA[2], fragB[2];

    int laneid_div_4 = laneid / 4;
    int laneid_mod_4 = laneid & 3;

    for (int i = start; i < end; i++)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        int startB = BlcPtrB[cidA];
        int endB = BlcPtrB[cidA + 1];

        int left = 0, right = BlcPtrC[rowid + 1] - BlcPtrC[rowid];

        if (__builtin_popcount(mapA) >= 10)
        {
            int offsetA = i * BSR_NNZ / 2;
            fragA[0] = BlcValA[offsetA + laneid_mod_4 * 2];
            fragA[1] = BlcValA[offsetA + laneid_mod_4 * 2 + 1];

            int strideB = endB - startB;
            int tail = 0;

            if (strideB > 0 && strideB % 8 != 0)
            {
                tail = strideB % 8;
                endB = endB - tail;
            }

            for (int j = startB; j < endB; j += 8)
            {
                uint32_t *cur_valB = BlcValB + (j * BSR_NNZ / 2);

                MAT_IDX_TYPE cidB = BlcCidB[j + laneid_div_4];
                MAT_MAP_TYPE mapB = BlcMapB[j + laneid_div_4];
                MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);

                fragB[0] = cur_valB[laneid * 2];
                fragB[1] = cur_valB[laneid * 2 + 1];

                half fragC[8] = {0};

                mma_m8n8k4_fp16_rr(fragC, fragA, fragB);

                int offsetC = 0;

                if (mapC)
                {
                    if (laneid_mod_4 == 0)
                    {
                        offsetC = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB);
                        left = offsetC;
                        curBlcMap[offsetC] |= mapC;
                    }
                }
                offsetC = __shfl_sync(0xffffffff, offsetC, 0, 4);

                int offsetC_val = offsetC * BSR_NNZ + laneid_mod_4 * 4;
                if (mapC)
                {
                    int startid = laneid < 16 ? 0 : 4;
                    curBlcVal[offsetC_val] += fragC[startid];
                    curBlcVal[offsetC_val + 1] += fragC[startid + 1];
                    curBlcVal[offsetC_val + 2] += fragC[startid + 2];
                    curBlcVal[offsetC_val + 3] += fragC[startid + 3];
                }
            }
            if (tail)
            {
                // int blcid = endB + laneid_div_4;
                uint32_t *cur_valB = BlcValB + (endB * BSR_NNZ / 2);

                MAT_IDX_TYPE cidB = laneid_div_4 < tail ? BlcCidB[endB + laneid_div_4] : 0;
                MAT_MAP_TYPE mapB = laneid_div_4 < tail ? BlcMapB[endB + laneid_div_4] : 0;
                MAT_MAP_TYPE mapC = laneid_div_4 < tail ? bitMatrixMul(mapA, mapB) : 0;

                fragB[0] = laneid_div_4 < tail ? cur_valB[laneid * 2] : 0;
                fragB[1] = laneid_div_4 < tail ? cur_valB[laneid * 2 + 1] : 0;

                half fragC[8] = {0};

                mma_m8n8k4_fp16_rr(fragC, fragA, fragB);

                int offsetC = 0;

                if (mapC)
                {
                    if (laneid_mod_4 == 0)
                    {
                        offsetC = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB);
                        left = offsetC;
                        curBlcMap[offsetC] |= mapC;
                    }
                }
                offsetC = __shfl_sync(0xffffffff, offsetC, 0, 4);

                int offsetC_val = offsetC * BSR_NNZ + laneid_mod_4 * 4;
                if (mapC)
                {
                    int startid = laneid < 16 ? 0 : 4;
                    curBlcVal[offsetC_val] += fragC[startid];
                    curBlcVal[offsetC_val + 1] += fragC[startid + 1];
                    curBlcVal[offsetC_val + 2] += fragC[startid + 2];
                    curBlcVal[offsetC_val + 3] += fragC[startid + 3];
                }
            }
        }
        else
        {
            int offsetA = i * BSR_NNZ;
            for (int j = startB + laneid; j < endB; j += WARP_SIZE)
            {
                MAT_IDX_TYPE cidB = BlcCidB[j];
                MAT_MAP_TYPE mapB = BlcMapB[j];

                MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);
                if (mapC)
                {
                    int offsetB = j * BSR_NNZ;

                    int strideA = __builtin_popcount(mapA);
                    int strideB = __builtin_popcount(mapB);

                    half temp_valC[BSR_NNZ] = {0};

                    for (int ii = 0; ii < BSR_M; ++ii)
                    {
                        for (int jj = 0; jj < BSR_N; ++jj)
                        {
                            half a = half_valA[offsetA + ii * BSR_N + jj];
                            if (mapA & (1 << (ii * BSR_N + jj)))
                            {
                                for (int kk = 0; kk < BSR_N; ++kk)
                                {
                                    half b = half_valB[offsetB + jj * BSR_N + kk];
                                    if (mapB & (1 << (jj * BSR_N + kk)))
                                    {
                                        temp_valC[ii * BSR_N + kk] += a * b;
                                    }
                                }
                            }
                        }
                    }

                    int offsetC = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB);
                    left = offsetC;
                    curBlcMap[offsetC] |= mapC;

                    for (int rc = 0; rc < BSR_NNZ; rc++)
                    {
                        if (getbit(mapC, rc))
                        {
                            curBlcVal[offsetC * BSR_NNZ + rc] += temp_valC[rc];
                        }
                    }
                }
            }
        }
    }
}
__global__ void numeric_spgemm_hybrid_tf32(MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA, float *BlcValA,
                                           MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB, float *BlcValB,
                                           MAT_PTR_TYPE *BlcPtrC, MAT_IDX_TYPE *BlcCidC, MAT_MAP_TYPE *BlcMapC, float *BlcValC,
                                           int m, int n, int k)
{
    int gl_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = gl_tid / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    int rowid = warpid;
    if (rowid >= m)
        return;

    int start = BlcPtrA[rowid];
    int end = BlcPtrA[rowid + 1];

    // MAT_VAL_TYPE fragA, fragB, fragC[2];
    uint32_t fragA[2] = {0};
    uint32_t fragB[1] = {0};

    float fragC[4] = {0};

    int laneid_mod_16 = laneid & 15;

    MAT_IDX_TYPE *curBlcCid = BlcCidC + BlcPtrC[rowid];
    MAT_MAP_TYPE *curBlcMap = BlcMapC + BlcPtrC[rowid];
    float *curBlcVal = BlcValC + (BlcPtrC[rowid] * BSR_NNZ);

    for (int i = start; i < end; i++)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        int left = 0, right = BlcPtrC[rowid + 1] - BlcPtrC[rowid];
        int offsetA = i * BSR_NNZ;

        int startB = BlcPtrB[cidA];
        int endB = BlcPtrB[cidA + 1];

        if (__builtin_popcount(mapA) >= 10)
        {
            // fragA = BlcValA[offsetA + laneid_mod_16];
            asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragA[0]) : "f"(BlcValA[offsetA + laneid_mod_16]));

            MAT_IDX_TYPE cidB[2];
            MAT_MAP_TYPE mapB[2];
            MAT_MAP_TYPE mapC[2];
            int offsetB[2];

            for (int j = startB; j < endB;)
            {
                cidB[0] = BlcCidB[j];
                mapB[0] = BlcMapB[j];

                mapC[0] = bitMatrixMul(mapA, mapB[0]);
                offsetB[0] = j;
                j++;

                if (mapC[0])
                {
                    do
                    {
                        cidB[1] = j < endB ? BlcCidB[j] : 0;
                        mapB[1] = j < endB ? BlcMapB[j] : 0;

                        mapC[1] = bitMatrixMul(mapA, mapB[1]);

                        offsetB[1] = j;
                        j++;
                    } while (mapC[1] == 0 && j < endB);

                    if (offsetB[1] >= endB || mapC[1] == 0)
                    {
                        int offsetB_val = offsetB[0] * BSR_NNZ;
                        int ext_idx = (laneid_mod_16 / 4) + (laneid_mod_16 & 3) * 4;

                        // fragB = laneid < 16 ? BlcValB[offsetB_val + ext_idx] : 0.0;
                        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragB[0]) : "f"((laneid < 16) ? BlcValB[offsetB_val + ext_idx] : 0.0f));

                        fragC[0] = 0.0, fragC[1] = 0.0;

                        mma_m16n8k4_tf32(fragC, fragA, fragB);

                        int offsetC = 0;

                        if (laneid == 0)
                        {
                            offsetC = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB[0]);
                            left = offsetC;
                            curBlcMap[offsetC] |= mapC[0];
                        }
                        offsetC = __shfl_sync(0xffffffff, offsetC, 0);

                        int offsetC_val = offsetC * BSR_NNZ + laneid_mod_16;
                        int laneid_div_4_left = (laneid / 4) * 4;
                        int target_id = laneid_div_4_left + (laneid - laneid_div_4_left) / 2;

                        int laneid_mod_2 = laneid % 2;
                        fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
                        fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id);
                        MAT_VAL_TYPE res = fragC[laneid_mod_2];

                        if (laneid < 16)
                            curBlcVal[offsetC_val] += res;
                    }
                    else
                    {
                        int offsetB_val = laneid < 16 ? offsetB[0] * BSR_NNZ : offsetB[1] * BSR_NNZ;
                        int ext_idx = (laneid_mod_16 / 4) + (laneid_mod_16 & 3) * 4;

                        // fragB = BlcValB[offsetB_val + ext_idx];
                        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragB[0]) : "f"(BlcValB[offsetB_val + ext_idx]));

                        fragC[0] = 0, fragC[1] = 0;

                        mma_m16n8k4_tf32(fragC, fragA, fragB);

                        int offsetC[2] = {0};

                        if (laneid == 0)
                        {
                            offsetC[0] = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB[0]);
                            left = offsetC[0];
                            curBlcMap[offsetC[0]] |= mapC[0];
                        }
                        offsetC[0] = __shfl_sync(0xffffffff, offsetC[0], 0);

                        if (laneid == 0)
                        {
                            offsetC[1] = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB[1]);
                            left = offsetC[1];
                            curBlcMap[offsetC[1]] |= mapC[1];
                        }
                        offsetC[1] = __shfl_sync(0xffffffff, offsetC[1], 0);
                        __syncwarp();

                        int offsetC_val = laneid < 16 ? (offsetC[0] * BSR_NNZ + laneid_mod_16) : (offsetC[1] * BSR_NNZ + laneid_mod_16);
                        int laneid_div_4_left = (laneid / 4) * 4;
                        int target_id = laneid < 16 ? (laneid_div_4_left + (laneid - laneid_div_4_left) / 2) : (laneid_div_4_left + 2 + (laneid - laneid_div_4_left) / 2);

                        int laneid_mod_2 = laneid % 2;

                        fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
                        fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id);
                        MAT_VAL_TYPE res = fragC[laneid_mod_2];

                        curBlcVal[offsetC_val] += res;
                    }
                }
            }
        }
        else
        {
            for (int j = startB + laneid; j < endB; j += WARP_SIZE)
            {
                MAT_IDX_TYPE cidB = BlcCidB[j];
                MAT_MAP_TYPE mapB = BlcMapB[j];

                MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);
                if (mapC)
                {
                    int offsetB = j * BSR_NNZ;

                    int strideA = __builtin_popcount(mapA);
                    int strideB = __builtin_popcount(mapB);

                    float temp_valC[BSR_NNZ] = {0};

                    for (int ii = 0; ii < BSR_M; ++ii)
                    {
                        for (int jj = 0; jj < BSR_N; ++jj)
                        {
                            float a = BlcValA[offsetA + ii * BSR_N + jj];
                            if (mapA & (1 << (ii * BSR_N + jj)))
                            {
                                for (int kk = 0; kk < BSR_N; ++kk)
                                {
                                    float b = BlcValB[offsetB + jj * BSR_N + kk];
                                    if (mapB & (1 << (jj * BSR_N + kk)))
                                    {
                                        temp_valC[ii * BSR_N + kk] += a * b;
                                    }
                                }
                            }
                        }
                    }

                    int offsetC = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB);
                    left = offsetC;
                    curBlcMap[offsetC] |= mapC;

                    for (int rc = 0; rc < BSR_NNZ; rc++)
                    {
                        if (getbit(mapC, rc))
                        {
                            curBlcVal[offsetC * BSR_NNZ + rc] += temp_valC[rc];
                        }
                    }
                }
            }
        }
    }
}

__global__ void numeric_spgemm_hybrid(MAT_PTR_TYPE *BlcPtrA, MAT_IDX_TYPE *BlcCidA, MAT_MAP_TYPE *BlcMapA, MAT_VAL_TYPE *BlcValA,
                                      MAT_PTR_TYPE *BlcPtrB, MAT_IDX_TYPE *BlcCidB, MAT_MAP_TYPE *BlcMapB, MAT_VAL_TYPE *BlcValB,
                                      MAT_PTR_TYPE *BlcPtrC, MAT_IDX_TYPE *BlcCidC, MAT_MAP_TYPE *BlcMapC, MAT_VAL_TYPE *BlcValC,
                                      int m, int n, int k)
{
    int gl_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = gl_tid / WARP_SIZE;
    int laneid = threadIdx.x & (WARP_SIZE - 1);

    int rowid = warpid;
    if (rowid >= m)
        return;

    int start = BlcPtrA[rowid];
    int end = BlcPtrA[rowid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2];

    int laneid_mod_16 = laneid & 15;

    for (int i = start; i < end; i++)
    {
        MAT_IDX_TYPE cidA = BlcCidA[i];
        MAT_MAP_TYPE mapA = BlcMapA[i];

        int left = 0, right = BlcPtrC[rowid + 1] - BlcPtrC[rowid];
        int offsetA = i * BSR_NNZ;

        int startB = BlcPtrB[cidA];
        int endB = BlcPtrB[cidA + 1];

        if (__builtin_popcount(mapA) >= 10)
        {
            fragA = BlcValA[offsetA + laneid_mod_16];

            MAT_IDX_TYPE cidB[2];
            MAT_MAP_TYPE mapB[2];
            MAT_MAP_TYPE mapC[2];
            int offsetB[2];

            for (int j = startB; j < endB;)
            {
                cidB[0] = BlcCidB[j];
                mapB[0] = BlcMapB[j];

                mapC[0] = bitMatrixMul(mapA, mapB[0]);
                offsetB[0] = j;
                j++;

                if (mapC[0])
                {
                    do
                    {
                        cidB[1] = j < endB ? BlcCidB[j] : 0;
                        mapB[1] = j < endB ? BlcMapB[j] : 0;

                        mapC[1] = bitMatrixMul(mapA, mapB[1]);

                        offsetB[1] = j;
                        j++;
                    } while (mapC[1] == 0 && j < endB);

                    if (offsetB[1] >= endB || mapC[1] == 0)
                    {
                        int offsetB_val = offsetB[0] * BSR_NNZ;
                        int ext_idx = (laneid_mod_16 / 4) + (laneid_mod_16 & 3) * 4;

                        fragB = laneid < 16 ? BlcValB[offsetB_val + ext_idx] : 0.0;

                        fragC[0] = 0.0, fragC[1] = 0.0;

                        mma_m8n8k4_SpGEMM(fragC, fragA, fragB);

                        int offsetC = 0;
                        MAT_IDX_TYPE *curBlcCid = BlcCidC + BlcPtrC[rowid];
                        MAT_MAP_TYPE *curBlcMap = BlcMapC + BlcPtrC[rowid];
                        MAT_VAL_TYPE *curBlcVal = BlcValC + (BlcPtrC[rowid] * BSR_NNZ);

                        if (laneid == 0)
                        {
                            offsetC = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB[0]);
                            left = offsetC;
                            curBlcMap[offsetC] |= mapC[0];
                        }
                        offsetC = __shfl_sync(0xffffffff, offsetC, 0);

                        int offsetC_val = offsetC * BSR_NNZ + laneid_mod_16;
                        int laneid_div_4_left = (laneid / 4) * 4;
                        int target_id = laneid_div_4_left + (laneid - laneid_div_4_left) / 2;

                        int laneid_mod_2 = laneid % 2;
                        fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
                        fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id);
                        MAT_VAL_TYPE res = fragC[laneid_mod_2];

                        if (laneid < 16)
                            curBlcVal[offsetC_val] += res;
                    }
                    else
                    {
                        int offsetB_val = laneid < 16 ? offsetB[0] * BSR_NNZ : offsetB[1] * BSR_NNZ;
                        int ext_idx = (laneid_mod_16 / 4) + (laneid_mod_16 & 3) * 4;

                        fragB = BlcValB[offsetB_val + ext_idx];

                        fragC[0] = 0, fragC[1] = 0;

                        mma_m8n8k4_SpGEMM(fragC, fragA, fragB);

                        int offsetC[2] = {0};
                        MAT_IDX_TYPE *curBlcCid = BlcCidC + BlcPtrC[rowid];
                        MAT_MAP_TYPE *curBlcMap = BlcMapC + BlcPtrC[rowid];
                        MAT_VAL_TYPE *curBlcVal = BlcValC + (BlcPtrC[rowid] * BSR_NNZ);

                        if (laneid == 0)
                        {
                            offsetC[0] = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB[0]);
                            left = offsetC[0];
                            curBlcMap[offsetC[0]] |= mapC[0];
                        }
                        offsetC[0] = __shfl_sync(0xffffffff, offsetC[0], 0);

                        if (laneid == 0)
                        {
                            offsetC[1] = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB[1]);
                            left = offsetC[1];
                            curBlcMap[offsetC[1]] |= mapC[1];
                        }
                        offsetC[1] = __shfl_sync(0xffffffff, offsetC[1], 0);
                        __syncwarp();

                        int offsetC_val = laneid < 16 ? (offsetC[0] * BSR_NNZ + laneid_mod_16) : (offsetC[1] * BSR_NNZ + laneid_mod_16);
                        int laneid_div_4_left = (laneid / 4) * 4;
                        int target_id = laneid < 16 ? (laneid_div_4_left + (laneid - laneid_div_4_left) / 2) : (laneid_div_4_left + 2 + (laneid - laneid_div_4_left) / 2);

                        int laneid_mod_2 = laneid % 2;

                        fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
                        fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id);
                        MAT_VAL_TYPE res = fragC[laneid_mod_2];

                        curBlcVal[offsetC_val] += res;
                    }
                }
            }
        }
        else
        {
            for (int j = startB + laneid; j < endB; j += WARP_SIZE)
            {
                MAT_IDX_TYPE cidB = BlcCidB[j];
                MAT_MAP_TYPE mapB = BlcMapB[j];

                MAT_MAP_TYPE mapC = bitMatrixMul(mapA, mapB);
                if (mapC)
                {
                    int offsetB = j * BSR_NNZ;

                    int strideA = __builtin_popcount(mapA);
                    int strideB = __builtin_popcount(mapB);

                    MAT_VAL_TYPE temp_valC[BSR_NNZ] = {0};

                    for (int ii = 0; ii < BSR_M; ++ii)
                    {
                        for (int jj = 0; jj < BSR_N; ++jj)
                        {
                            MAT_VAL_TYPE a = BlcValA[offsetA + ii * BSR_N + jj];
                            if (mapA & (1 << (ii * BSR_N + jj)))
                            {
                                for (int kk = 0; kk < BSR_N; ++kk)
                                {
                                    MAT_VAL_TYPE b = BlcValB[offsetB + jj * BSR_N + kk];
                                    if (mapB & (1 << (jj * BSR_N + kk)))
                                    {
                                        temp_valC[ii * BSR_N + kk] += a * b;
                                    }
                                }
                            }
                        }
                    }

                    MAT_IDX_TYPE *curBlcCid = BlcCidC + BlcPtrC[rowid];
                    MAT_MAP_TYPE *curBlcMap = BlcMapC + BlcPtrC[rowid];
                    MAT_VAL_TYPE *curBlcVal = BlcValC + (BlcPtrC[rowid] * BSR_NNZ);
                    int offsetC = BinarySearch2_SpGEMM(curBlcCid, left, right, cidB);
                    left = offsetC;
                    curBlcMap[offsetC] |= mapC;

                    for (int rc = 0; rc < BSR_NNZ; rc++)
                    {
                        if (getbit(mapC, rc))
                        {
                            curBlcVal[offsetC * BSR_NNZ + rc] += temp_valC[rc];
                        }
                    }
                }
            }
        }
    }
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

__global__ void checkData(int m, int nnz, int *ptr1, int *ptr2, int *idx1, int *idx2, double *val1, double *val2)
{
    for (int i = 0; i < m; i++)
    {
        if (ptr1[i] != ptr2[i])
        {
            printf("ptr %d %d %d\n", i, ptr1[i], ptr2[i]);
        }
    }
    for (int i = 0; i < nnz; i++)
    {
        if (idx1[i] != idx2[i])
        {
            printf("idx %d %d %d\n", i, idx1[i], idx2[i]);
        }
    }

    for (int i = 0; i < nnz; i++)
    {
        if (val1[i] != val2[i])
        {
            printf("val %d %lf %lf\n", i, val1[i], val2[i]);
        }
    }
}
__global__ void bsr_val_fp64_to_32(MAT_VAL_TYPE *fp64_val_csr, float *fp32_val_bsr, int nnz)
{
    int ptr = threadIdx.x + blockDim.x * blockIdx.x;
    if (ptr >= nnz)
        return;
    fp32_val_bsr[ptr] = fp64_val_csr[ptr];
    // printf("%f\t %lf\t %lf \n", fp32_val_bsr[ptr], fp64_val_csr[ptr], fp64_val_csr[ptr] - fp32_val_bsr[ptr]);
}

__global__ void bsr_val_fp32_to_64(MAT_VAL_TYPE *fp64_val_csr, float *fp32_val_bsr, int nnz)
{
    int ptr = threadIdx.x + blockDim.x * blockIdx.x;
    if (ptr >= nnz)
        return;
    fp64_val_csr[ptr] = fp32_val_bsr[ptr];
}

__global__ void bsr_val_fp16_to_64(MAT_VAL_TYPE *fp64_val_csr, uint32_t *fp16_val_bsr, int nnz)
{
    int ptr = threadIdx.x + blockDim.x * blockIdx.x;
    if (ptr >= nnz)
        return;
    half *d_bsr_half = reinterpret_cast<half *>(&fp16_val_bsr[0]);
    fp64_val_csr[ptr] = d_bsr_half[ptr];
}

__global__ void bsr_val_fp64_to_16(MAT_VAL_TYPE *fp64_val_csr, uint32_t *fp16_val_bsr, int nnz)
{
    int ptr = threadIdx.x + blockDim.x * blockIdx.x;
    if (ptr >= nnz)
        return;
    half *d_bsr_half = reinterpret_cast<half *>(&fp16_val_bsr[0]);
    d_bsr_half[ptr] = fp64_val_csr[ptr];
}
void spgemm_amgT_fp64(hypre_CSRMatrix *A,
                      hypre_CSRMatrix *B,
                      hypre_CSRMatrix **C_ptr)
{
    struct timeval t1, t2;
    struct timeval t_start, t_end;
    gettimeofday(&t1, NULL);
    CSR2BSR_GPU(A);
    CSR2BSR_GPU(B);
    gettimeofday(&t2, NULL);
    time_spgemm_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    bsrMAT dmatA, dmatB;
    bsrMAT *dmatC = (bsrMAT *)malloc(sizeof(bsrMAT));
    dmatA = *hypre_BSR(A);
    dmatB = *hypre_BSR(B);
    dmatC->blc_row = dmatA.blc_row;
    dmatC->blc_col = dmatB.blc_col;
    dmatC->row = hypre_CSRMatrixNumRows(A);
    dmatC->col = hypre_CSRMatrixNumCols(B);

    int blc_m = dmatA.blc_row;
    int blc_n = dmatB.blc_col;
    int blc_k = dmatA.blc_col;

    MAT_PTR_TYPE *blcCub;
    cudaMalloc((void **)&blcCub, sizeof(MAT_PTR_TYPE) * blc_m);

    int *bin_offset;
    cudaMalloc((void **)&bin_offset, sizeof(int) * (BIN_NUM + 1));
    cudaMemset(bin_offset, 0, sizeof(int) * (BIN_NUM + 1));
    int *bin_size;
    cudaMalloc((void **)&bin_size, sizeof(int) * BIN_NUM);
    cudaMemset(bin_size, 0, sizeof(int) * BIN_NUM);
    MAT_IDX_TYPE *bin_rowidx;
    cudaMalloc((void **)&bin_rowidx, sizeof(MAT_IDX_TYPE) * blc_m);
    int *max_num;
    cudaMalloc((void **)&max_num, sizeof(int));

    cudaMalloc((void **)&dmatC->blcPtr, sizeof(MAT_PTR_TYPE) * (blc_m + 1));
    cudaMemset(dmatC->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (blc_m + 1));

    int ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
    int BlockNum = (blc_m + ThreadNum - 1) / ThreadNum;
    // preprocess of spgemm
    compute_Cub_bin<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatB.blcPtr, blcCub, blc_m, bin_offset);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + (BIN_NUM + 1), bin_offset, 0);

    set_bin<<<BlockNum, ThreadNum>>>(blc_m, blcCub, bin_rowidx, bin_offset, bin_size, max_num);
    cudaDeviceSynchronize();

    int max_len;
    cudaMemcpy(&max_len, max_num, sizeof(int), cudaMemcpyDeviceToHost);
    int *offset = (int *)malloc(sizeof(int) * (BIN_NUM + 1));
    cudaMemcpy(offset, bin_offset, sizeof(int) * (BIN_NUM + 1), cudaMemcpyDeviceToHost);

    for (int i = BIN_NUM - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
        BlockNum = row_num;

        if (row_num)
            switch (i)
            {
            case 0:
                symbolic_spgemm_step1<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 1:
                symbolic_spgemm_step1<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 2:
                symbolic_spgemm_step1<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 3:
                symbolic_spgemm_step1<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 4:
                symbolic_spgemm_step1<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 5:
                symbolic_spgemm_step1<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 6:
                symbolic_spgemm_step1<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 7:
            {
                int over_num = 0;
                int *d_over_num, *d_over_rid;
                cudaMalloc((void **)&d_over_num, sizeof(int));
                cudaMalloc((void **)&d_over_rid, sizeof(int) * row_num);
                cudaMemset(d_over_num, 0, sizeof(int));
                symbolic_spgemm_step1_large<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                           d_over_num, d_over_rid,
                                                                           dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                           dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                           dmatC->blcPtr, blc_m, blc_n, blc_k);
                cudaDeviceSynchronize();
                cudaMemcpy(&over_num, d_over_num, sizeof(int), cudaMemcpyDeviceToHost);
                if (over_num)
                {
                    int *hash_global;
                    cudaMalloc((void **)&hash_global, sizeof(int) * over_num * max_len);
                    cudaMemset(hash_global, -1, sizeof(int) * over_num * max_len);

                    BlockNum = over_num;
                    symbolic_spgemm_step1_gl<<<BlockNum, ThreadNum>>>(d_over_rid, hash_global, max_len,
                                                                      dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                      dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                      dmatC->blcPtr, blc_m, blc_n, blc_k);
                    cudaDeviceSynchronize();
                    cudaFree(hash_global);
                }
                cudaFree(d_over_num);
                cudaFree(d_over_rid);
            }
            break;
            default:
                break;
            }
    }
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, dmatC->blcPtr, dmatC->blcPtr + (blc_m + 1), dmatC->blcPtr, 0);

    int blc_num;
    cudaMemcpy(&blc_num, dmatC->blcPtr + blc_m, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    dmatC->blc_num = blc_num;

    cudaMalloc((void **)&dmatC->blcIdx, sizeof(MAT_IDX_TYPE) * blc_num);

    for (int i = BIN_NUM - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
        BlockNum = row_num;

        if (row_num != 0)
            switch (i)
            {
            case 0:
                symbolic_spgemm_step2<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 1:
                symbolic_spgemm_step2<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 2:
                symbolic_spgemm_step2<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 3:
                symbolic_spgemm_step2<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 4:
                symbolic_spgemm_step2<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 5:
                symbolic_spgemm_step2<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 6:
                symbolic_spgemm_step2<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 7:
            {
                int over_num = 0;
                int *d_over_num, *d_over_rid;
                cudaMalloc((void **)&d_over_num, sizeof(int));
                cudaMalloc((void **)&d_over_rid, sizeof(int) * row_num);
                cudaMemset(d_over_num, 0, sizeof(int));
                symbolic_spgemm_step2_large<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                           d_over_num, d_over_rid,
                                                                           dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                           dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                           dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                cudaDeviceSynchronize();
                cudaMemcpy(&over_num, d_over_num, sizeof(int), cudaMemcpyDeviceToHost);

                if (over_num)
                {
                    int *hash_global;
                    cudaMalloc((void **)&hash_global, sizeof(int) * over_num * max_len);
                    cudaMemset(hash_global, -1, sizeof(int) * over_num * max_len);

                    BlockNum = over_num;
                    symbolic_spgemm_step2_gl<<<BlockNum, ThreadNum>>>(d_over_rid, hash_global, max_len,
                                                                      dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                      dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                      dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                    cudaDeviceSynchronize();
                    cudaFree(hash_global);
                }
                cudaFree(d_over_num);
                cudaFree(d_over_rid);
            }
            break;
            default:
                break;
            }
    }
    cudaDeviceSynchronize();

    MAT_PTR_TYPE blc_nnz = blc_num * BSR_NNZ;
    dmatC->nnz = blc_nnz;
    cudaMalloc((void **)&dmatC->blcMap, sizeof(MAT_MAP_TYPE) * blc_num);
    cudaMemset(dmatC->blcMap, 0, sizeof(MAT_MAP_TYPE) * blc_num);

    cudaMalloc((void **)&dmatC->blcVal, sizeof(MAT_VAL_TYPE) * blc_nnz);
    cudaMemset(dmatC->blcVal, 0, sizeof(MAT_VAL_TYPE) * blc_nnz);

    BlockNum = (blc_m + WARP_NUM_SPGM - 1) / WARP_NUM_SPGM;

    numeric_spgemm_hybrid<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap, dmatA.blcVal,
                                                   dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap, dmatB.blcVal,
                                                   dmatC->blcPtr, dmatC->blcIdx, dmatC->blcMap, dmatC->blcVal,
                                                   blc_m, blc_n, blc_k);
    cudaDeviceSynchronize();

    // BSR2CSR_GPU

    //  csrMAT csrmat;
    gettimeofday(&t2, NULL);
    double time_spgemm_kernel_time = 0;
    time_spgemm_kernel_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spgemm_kernel_m=%d\n", blc_m);
    printf("spgemm_kernel_k=%d\n", blc_k);
    printf("spgemm_kernel_n=%d\n", blc_n);
    printf("spgemm_kernel_time=%lf\n", time_spgemm_kernel_time);
#endif

    time_spgemm += time_spgemm_kernel_time;

    gettimeofday(&t1, NULL);
    gettimeofday(&t_start, NULL);
    MAT_PTR_TYPE *d_csrptr = NULL;
    int dmatC_nnz = 0;
    d_csrptr = hypre_TAlloc(HYPRE_Int, dmatC->row + 1, HYPRE_MEMORY_DEVICE);
    // cudaMalloc((void **)&d_csrptr, sizeof(MAT_PTR_TYPE) * (dmatC->row + 1));
    cudaMemset(d_csrptr, 0.0, sizeof(MAT_PTR_TYPE) * (dmatC->row + 1));
    BSR_TO_CSR_step1(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col, d_csrptr, dmatC->blcPtr, dmatC->blcMap);
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);
    bsr2csr_step1 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    gettimeofday(&t_start, NULL);
    thrust::exclusive_scan(thrust::device, d_csrptr, d_csrptr + (dmatC->row + 1), d_csrptr, 0);
    cudaMemcpy(&(dmatC_nnz), &d_csrptr[dmatC->row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    bsr2csr_step2 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    gettimeofday(&t_start, NULL);
    MAT_IDX_TYPE *d_csridx = NULL;
    MAT_VAL_TYPE *d_csrval = NULL;
    d_csridx = hypre_TAlloc(HYPRE_Int, dmatC_nnz, HYPRE_MEMORY_DEVICE);
    d_csrval = hypre_TAlloc(HYPRE_Complex, dmatC_nnz, HYPRE_MEMORY_DEVICE);
    // cudaMalloc((void **)&d_csridx, sizeof(MAT_IDX_TYPE) * dmatC_nnz);
    // cudaMalloc((void **)&d_csrval, sizeof(MAT_VAL_TYPE) * dmatC_nnz);

    cudaMemset(d_csrval, 0.0, sizeof(MAT_VAL_TYPE) * dmatC_nnz);

    BSR_TO_CSR_step2(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col,
                     d_csrptr, d_csridx, d_csrval,
                     dmatC->blcPtr, dmatC->blcIdx, dmatC->blcVal, dmatC->blcMap);
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    bsr2csr_step3 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    //  printf("SpGEMM 6\n");
    *C_ptr = hypre_CSRMatrixCreate(dmatC->row, dmatC->col, 0);
    hypre_CSRMatrixMemoryLocation(*C_ptr) = HYPRE_MEMORY_DEVICE;
    hypre_CSRMatrixNumNonzeros(*C_ptr) = dmatC_nnz;
    hypre_CSRMatrixI(*C_ptr) = d_csrptr;
    hypre_CSRMatrixJ(*C_ptr) = d_csridx;
    hypre_CSRMatrixData(*C_ptr) = d_csrval;
    //    printf("SpGEMM 7\n");
    hypre_BSR(*C_ptr) = dmatC;
    hypre_BSRTAG(*C_ptr) = 1;

    gettimeofday(&t2, NULL);
    time_spgemm_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
}
enum AMGT_PRECISION get_calculationPrecison(int level)
{

    if (level <= 0)
    {
        return AMGT_DOUBLE;
    }
    else if (level <= 1)
    {
        return AMGT_FLOAT;
    }
    else
    {
        return AMGT_HALF;
    }
}
void spgemm_amgT_fp16(hypre_CSRMatrix *A,
                      hypre_CSRMatrix *B,
                      hypre_CSRMatrix **C_ptr)
{
    struct timeval t1, t2;
    struct timeval t_start, t_end;
    gettimeofday(&t1, NULL);
    // printf("begin preprocess\n");
    CSR2BSR_GPU(A);
    CSR2BSR_GPU(B);
    // printf("end preprocess\n");
    gettimeofday(&t2, NULL);
    time_spgemm_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    bsrMAT dmatA, dmatB;
    bsrMAT *dmatC = (bsrMAT *)malloc(sizeof(bsrMAT));
    dmatA = *hypre_BSR(A);
    dmatB = *hypre_BSR(B);
    dmatC->blc_row = dmatA.blc_row;
    dmatC->blc_col = dmatB.blc_col;
    dmatC->row = hypre_CSRMatrixNumRows(A);
    dmatC->col = hypre_CSRMatrixNumCols(B);

    int blc_m = dmatA.blc_row;
    int blc_n = dmatB.blc_col;
    int blc_k = dmatA.blc_col;

    MAT_PTR_TYPE *blcCub;
    cudaMalloc((void **)&blcCub, sizeof(MAT_PTR_TYPE) * blc_m);

    int *bin_offset;
    cudaMalloc((void **)&bin_offset, sizeof(int) * (BIN_NUM + 1));
    cudaMemset(bin_offset, 0, sizeof(int) * (BIN_NUM + 1));
    int *bin_size;
    cudaMalloc((void **)&bin_size, sizeof(int) * BIN_NUM);
    cudaMemset(bin_size, 0, sizeof(int) * BIN_NUM);
    MAT_IDX_TYPE *bin_rowidx;
    cudaMalloc((void **)&bin_rowidx, sizeof(MAT_IDX_TYPE) * blc_m);
    int *max_num;
    cudaMalloc((void **)&max_num, sizeof(int));

    cudaMalloc((void **)&dmatC->blcPtr, sizeof(MAT_PTR_TYPE) * (blc_m + 1));
    cudaMemset(dmatC->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (blc_m + 1));

    int ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
    int BlockNum = (blc_m + ThreadNum - 1) / ThreadNum;
    // preprocess of spgemm
    compute_Cub_bin<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatB.blcPtr, blcCub, blc_m, bin_offset);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + (BIN_NUM + 1), bin_offset, 0);

    set_bin<<<BlockNum, ThreadNum>>>(blc_m, blcCub, bin_rowidx, bin_offset, bin_size, max_num);
    cudaDeviceSynchronize();

    int max_len;
    cudaMemcpy(&max_len, max_num, sizeof(int), cudaMemcpyDeviceToHost);
    int *offset = (int *)malloc(sizeof(int) * (BIN_NUM + 1));
    cudaMemcpy(offset, bin_offset, sizeof(int) * (BIN_NUM + 1), cudaMemcpyDeviceToHost);

    for (int i = BIN_NUM - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
        BlockNum = row_num;

        if (row_num)
            switch (i)
            {
            case 0:
                symbolic_spgemm_step1<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 1:
                symbolic_spgemm_step1<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 2:
                symbolic_spgemm_step1<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 3:
                symbolic_spgemm_step1<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 4:
                symbolic_spgemm_step1<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 5:
                symbolic_spgemm_step1<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 6:
                symbolic_spgemm_step1<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 7:
            {
                int over_num = 0;
                int *d_over_num, *d_over_rid;
                cudaMalloc((void **)&d_over_num, sizeof(int));
                cudaMalloc((void **)&d_over_rid, sizeof(int) * row_num);
                cudaMemset(d_over_num, 0, sizeof(int));
                symbolic_spgemm_step1_large<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                           d_over_num, d_over_rid,
                                                                           dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                           dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                           dmatC->blcPtr, blc_m, blc_n, blc_k);
                cudaDeviceSynchronize();
                cudaMemcpy(&over_num, d_over_num, sizeof(int), cudaMemcpyDeviceToHost);
                if (over_num)
                {
                    int *hash_global;
                    cudaMalloc((void **)&hash_global, sizeof(int) * over_num * max_len);
                    cudaMemset(hash_global, -1, sizeof(int) * over_num * max_len);

                    BlockNum = over_num;
                    symbolic_spgemm_step1_gl<<<BlockNum, ThreadNum>>>(d_over_rid, hash_global, max_len,
                                                                      dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                      dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                      dmatC->blcPtr, blc_m, blc_n, blc_k);
                    cudaDeviceSynchronize();
                    cudaFree(hash_global);
                }
                cudaFree(d_over_num);
                cudaFree(d_over_rid);
            }
            break;
            default:
                break;
            }
    }
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, dmatC->blcPtr, dmatC->blcPtr + (blc_m + 1), dmatC->blcPtr, 0);

    int blc_num;
    cudaMemcpy(&blc_num, dmatC->blcPtr + blc_m, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    dmatC->blc_num = blc_num;

    cudaMalloc((void **)&dmatC->blcIdx, sizeof(MAT_IDX_TYPE) * blc_num);

    for (int i = BIN_NUM - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
        BlockNum = row_num;

        if (row_num != 0)
            switch (i)
            {
            case 0:
                symbolic_spgemm_step2<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 1:
                symbolic_spgemm_step2<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 2:
                symbolic_spgemm_step2<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 3:
                symbolic_spgemm_step2<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 4:
                symbolic_spgemm_step2<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 5:
                symbolic_spgemm_step2<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 6:
                symbolic_spgemm_step2<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 7:
            {
                int over_num = 0;
                int *d_over_num, *d_over_rid;
                cudaMalloc((void **)&d_over_num, sizeof(int));
                cudaMalloc((void **)&d_over_rid, sizeof(int) * row_num);
                cudaMemset(d_over_num, 0, sizeof(int));
                symbolic_spgemm_step2_large<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                           d_over_num, d_over_rid,
                                                                           dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                           dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                           dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                cudaDeviceSynchronize();
                cudaMemcpy(&over_num, d_over_num, sizeof(int), cudaMemcpyDeviceToHost);

                if (over_num)
                {
                    int *hash_global;
                    cudaMalloc((void **)&hash_global, sizeof(int) * over_num * max_len);
                    cudaMemset(hash_global, -1, sizeof(int) * over_num * max_len);

                    BlockNum = over_num;
                    symbolic_spgemm_step2_gl<<<BlockNum, ThreadNum>>>(d_over_rid, hash_global, max_len,
                                                                      dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                      dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                      dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                    cudaDeviceSynchronize();
                    cudaFree(hash_global);
                }
                cudaFree(d_over_num);
                cudaFree(d_over_rid);
            }
            break;
            default:
                break;
            }
    }
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);

    double time_spgemm_kernel_time = 0;
    time_spgemm_kernel_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // time_spgemm += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;



    MAT_PTR_TYPE blc_nnz = blc_num * BSR_NNZ;
    dmatC->nnz = blc_nnz;
    cudaMalloc((void **)&dmatC->blcMap, sizeof(MAT_MAP_TYPE) * blc_num);
    cudaMemset(dmatC->blcMap, 0, sizeof(MAT_MAP_TYPE) * blc_num);
    cudaMalloc((void **)&dmatC->blcVal_fp16, sizeof(uint32_t) * ((blc_nnz + 1) / 2));
    cudaMemset(dmatC->blcVal_fp16, 0, sizeof(uint32_t) * ((blc_nnz + 1) / 2));
    cudaMalloc((void **)&dmatC->blcVal, sizeof(MAT_VAL_TYPE) * blc_nnz);
    cudaMemset(dmatC->blcVal, 0, sizeof(MAT_VAL_TYPE) * blc_nnz);

    int ThreadNum_spgemm = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_spgemm_A = (dmatA.nnz + ThreadNum_spgemm - 1) / ThreadNum_spgemm;
    int BlockNum_spgemm_B = (dmatB.nnz + ThreadNum_spgemm - 1) / ThreadNum_spgemm;
    // cudaMalloc((void **)&dmatA.blcVal_fp32, sizeof(float) * dmatA.nnz);
    // cudaMemset(dmatA.blcVal_fp32, 0, sizeof(float) * dmatA.nnz);
    // bsr_val_fp64_to_32<<<ThreadNum_spgemm, BlockNum_spgemm_A>>>(dmatA.blcVal, dmatA.blcVal_fp32, dmatA.nnz);

    // cudaMalloc((void **)&dmatB.blcVal_fp32, sizeof(float) * dmatB.nnz);
    // cudaMemset(dmatB.blcVal_fp32, 0, sizeof(float) * dmatB.nnz);
    // bsr_val_fp64_to_32<<<ThreadNum_spgemm,BlockNum_spgemm_B>>>(dmatB.blcVal,dmatB.blcVal_fp32,dmatB.nnz);

    if (!hypre_MixedPrecisionTag(A))
    {
        hypre_MixedPrecisionTag(A) = 1;
        cudaMalloc((void **)&dmatA.blcVal_fp16, sizeof(uint32_t) * ((dmatA.nnz + 1) / 2));
        cudaMemset(dmatA.blcVal_fp16, 0, sizeof(uint32_t) * ((dmatA.nnz + 1) / 2));
        bsr_val_fp64_to_16<<<BlockNum_spgemm_A, ThreadNum_spgemm>>>(dmatA.blcVal, dmatA.blcVal_fp16, dmatA.nnz);
    }
    // else{
    //     printf("no need A\n");
    // }

    if (!hypre_MixedPrecisionTag(B))
    {
        hypre_MixedPrecisionTag(B) = 1;
        cudaMalloc((void **)&dmatB.blcVal_fp16, sizeof(uint32_t) * ((dmatB.nnz + 1) / 2));
        cudaMemset(dmatB.blcVal_fp16, 0, sizeof(uint32_t) * ((dmatB.nnz + 1) / 2));
        bsr_val_fp64_to_16<<<BlockNum_spgemm_B, ThreadNum_spgemm>>>(dmatB.blcVal, dmatB.blcVal_fp16, dmatB.nnz);
    }
    // else{
    //     printf("no need B\n");
    // }
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);
    // printf("numeric begin 2\n");
    BlockNum = (blc_m + WARP_NUM_SPGM - 1) / WARP_NUM_SPGM;

    numeric_spgemm_hybrid_f16<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap, dmatA.blcVal_fp16,
                                                       dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap, dmatB.blcVal_fp16,
                                                       dmatC->blcPtr, dmatC->blcIdx, dmatC->blcMap, dmatC->blcVal_fp16,
                                                       blc_m, blc_n, blc_k);

    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    // double time_spgemm_kernel_time = 0;
    time_spgemm_kernel_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spgemm_kernel_m=%d\n", blc_m);
    printf("spgemm_kernel_k=%d\n", blc_k);
    printf("spgemm_kernel_n=%d\n", blc_n);
    printf("spgemm_kernel_time=%lf\n", time_spgemm_kernel_time);
#endif

    time_spgemm += time_spgemm_kernel_time;
    // numeric_spgemm_hybrid<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap, dmatA.blcVal,
    //                                                dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap, dmatB.blcVal,
    //                                                dmatC->blcPtr, dmatC->blcIdx, dmatC->blcMap, dmatC->blcVal,
    //                                                blc_m, blc_n, blc_k);

    cudaDeviceSynchronize();
    int BlockNum_spgemm_C = (dmatC->nnz + ThreadNum_spgemm - 1) / ThreadNum_spgemm;
    // HYPRE_Real *sum_err;
    // HYPRE_Real sum_err_cpu;
    // cudaMalloc((void **)&sum_err, sizeof(HYPRE_Real));
    // cudaMemset(sum_err, 0.0, sizeof(HYPRE_Real));
    // check_val_fp64_32<<<BlockNum_spgemm_C, ThreadNum_spgemm>>>(dmatC->blcVal, dmatC->blcVal_fp32, sum_err, dmatC->nnz);
    // cudaDeviceSynchronize();

    // cudaMemcpy(&sum_err_cpu, sum_err, sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);
    // printf("begin fuzhi error %lf\n", sum_err_cpu);

    bsr_val_fp16_to_64<<<BlockNum_spgemm_C, ThreadNum_spgemm>>>(dmatC->blcVal, dmatC->blcVal_fp16, dmatC->nnz);

    cudaDeviceSynchronize();
    // BSR2CSR_GPU

    // cudaMemset(sum_err, 0.0, sizeof(HYPRE_Real));
    // check_val_fp64_32<<<BlockNum_spgemm_C, ThreadNum_spgemm>>>(dmatC->blcVal, dmatC->blcVal_fp32, sum_err, dmatC->nnz);
    // cudaDeviceSynchronize();

    // cudaMemcpy(&sum_err_cpu, sum_err, sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);
    // printf("end fuzhi error %lf\n", sum_err_cpu);

    //  csrMAT csrmat;

    gettimeofday(&t1, NULL);
    gettimeofday(&t_start, NULL);
    MAT_PTR_TYPE *d_csrptr = NULL;
    int dmatC_nnz = 0;
    // printf("SpGEMM 0.1\n");
    // cudaMalloc((void **)&d_csrptr, sizeof(HYPRE_Int) * (dmatC->row + 1));
    d_csrptr = hypre_TAlloc(HYPRE_Int, dmatC->row + 1, HYPRE_MEMORY_DEVICE);
    // cudaMalloc((void **)&d_csrptr, sizeof(MAT_PTR_TYPE) * (dmatC->row + 1));
    // printf("SpGEMM 1.1\n");
    cudaMemset(d_csrptr, 0.0, sizeof(MAT_PTR_TYPE) * (dmatC->row + 1));
    // printf("SpGEMM 1\n");
    BSR_TO_CSR_step1(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col, d_csrptr, dmatC->blcPtr, dmatC->blcMap);
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);
    bsr2csr_step1 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
    // printf("SpGEMM 2\n");
    gettimeofday(&t_start, NULL);
    thrust::exclusive_scan(thrust::device, d_csrptr, d_csrptr + (dmatC->row + 1), d_csrptr, 0);
    cudaMemcpy(&(dmatC_nnz), &d_csrptr[dmatC->row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // printf("SpGEMM 3\n");
    gettimeofday(&t_end, NULL);
    bsr2csr_step2 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
    // printf("SpGEMM 4\n");
    gettimeofday(&t_start, NULL);
    MAT_IDX_TYPE *d_csridx = NULL;
    MAT_VAL_TYPE *d_csrval = NULL;
    d_csridx = hypre_TAlloc(HYPRE_Int, dmatC_nnz, HYPRE_MEMORY_DEVICE);
    d_csrval = hypre_TAlloc(HYPRE_Complex, dmatC_nnz, HYPRE_MEMORY_DEVICE);
    // cudaMalloc((void **)&d_csridx, sizeof(MAT_IDX_TYPE) * dmatC_nnz);
    // cudaMalloc((void **)&d_csrval, sizeof(MAT_VAL_TYPE) * dmatC_nnz);
    // printf("SpGEMM 5\n");
    cudaMemset(d_csrval, 0.0, sizeof(MAT_VAL_TYPE) * dmatC_nnz);

    BSR_TO_CSR_step2(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col,
                     d_csrptr, d_csridx, d_csrval,
                     dmatC->blcPtr, dmatC->blcIdx, dmatC->blcVal, dmatC->blcMap);
    // BSR_TO_CSR_step2_fp32(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col,
    //                       d_csrptr, d_csridx, d_csrval,
    //                       dmatC->blcPtr, dmatC->blcIdx, dmatC->blcVal_fp32, dmatC->blcMap);
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    bsr2csr_step3 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    // printf("SpGEMM 6\n");
    *C_ptr = hypre_CSRMatrixCreate(dmatC->row, dmatC->col, 0);
    hypre_CSRMatrixMemoryLocation(*C_ptr) = HYPRE_MEMORY_DEVICE;
    hypre_CSRMatrixNumNonzeros(*C_ptr) = dmatC_nnz;
    hypre_CSRMatrixI(*C_ptr) = d_csrptr;
    hypre_CSRMatrixJ(*C_ptr) = d_csridx;
    hypre_CSRMatrixData(*C_ptr) = d_csrval;
    // printf("SpGEMM 7\n");
    hypre_BSR(*C_ptr) = dmatC;
    hypre_BSRTAG(*C_ptr) = 1;
    hypre_BSR(A)->blcVal_fp16 = dmatA.blcVal_fp16;
    hypre_BSR(B)->blcVal_fp16 = dmatB.blcVal_fp16;
    // hypre_MixedPrecisionTag(*C_ptr) = 1;

    // printf("SpGEMM 8\n");

    gettimeofday(&t2, NULL);
    time_spgemm_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
}
void spgemm_amgT_fp32(hypre_CSRMatrix *A,
                      hypre_CSRMatrix *B,
                      hypre_CSRMatrix **C_ptr)
{
    struct timeval t1, t2;
    struct timeval t_start, t_end;
    gettimeofday(&t1, NULL);
    // printf("begin preprocess\n");
    CSR2BSR_GPU(A);
    CSR2BSR_GPU(B);
    // printf("end preprocess\n");
    gettimeofday(&t2, NULL);
    time_spgemm_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    bsrMAT dmatA, dmatB;
    bsrMAT *dmatC = (bsrMAT *)malloc(sizeof(bsrMAT));
    dmatA = *hypre_BSR(A);
    dmatB = *hypre_BSR(B);
    dmatC->blc_row = dmatA.blc_row;
    dmatC->blc_col = dmatB.blc_col;
    dmatC->row = hypre_CSRMatrixNumRows(A);
    dmatC->col = hypre_CSRMatrixNumCols(B);

    int blc_m = dmatA.blc_row;
    int blc_n = dmatB.blc_col;
    int blc_k = dmatA.blc_col;

    MAT_PTR_TYPE *blcCub;
    cudaMalloc((void **)&blcCub, sizeof(MAT_PTR_TYPE) * blc_m);

    int *bin_offset;
    cudaMalloc((void **)&bin_offset, sizeof(int) * (BIN_NUM + 1));
    cudaMemset(bin_offset, 0, sizeof(int) * (BIN_NUM + 1));
    int *bin_size;
    cudaMalloc((void **)&bin_size, sizeof(int) * BIN_NUM);
    cudaMemset(bin_size, 0, sizeof(int) * BIN_NUM);
    MAT_IDX_TYPE *bin_rowidx;
    cudaMalloc((void **)&bin_rowidx, sizeof(MAT_IDX_TYPE) * blc_m);
    int *max_num;
    cudaMalloc((void **)&max_num, sizeof(int));

    cudaMalloc((void **)&dmatC->blcPtr, sizeof(MAT_PTR_TYPE) * (blc_m + 1));
    cudaMemset(dmatC->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (blc_m + 1));

    int ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
    int BlockNum = (blc_m + ThreadNum - 1) / ThreadNum;
    // preprocess of spgemm
    compute_Cub_bin<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatB.blcPtr, blcCub, blc_m, bin_offset);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + (BIN_NUM + 1), bin_offset, 0);

    set_bin<<<BlockNum, ThreadNum>>>(blc_m, blcCub, bin_rowidx, bin_offset, bin_size, max_num);
    cudaDeviceSynchronize();

    int max_len;
    cudaMemcpy(&max_len, max_num, sizeof(int), cudaMemcpyDeviceToHost);
    int *offset = (int *)malloc(sizeof(int) * (BIN_NUM + 1));
    cudaMemcpy(offset, bin_offset, sizeof(int) * (BIN_NUM + 1), cudaMemcpyDeviceToHost);

    for (int i = BIN_NUM - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
        BlockNum = row_num;

        if (row_num)
            switch (i)
            {
            case 0:
                symbolic_spgemm_step1<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 1:
                symbolic_spgemm_step1<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 2:
                symbolic_spgemm_step1<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 3:
                symbolic_spgemm_step1<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 4:
                symbolic_spgemm_step1<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 5:
                symbolic_spgemm_step1<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 6:
                symbolic_spgemm_step1<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, blc_m, blc_n, blc_k);
                break;
            case 7:
            {
                int over_num = 0;
                int *d_over_num, *d_over_rid;
                cudaMalloc((void **)&d_over_num, sizeof(int));
                cudaMalloc((void **)&d_over_rid, sizeof(int) * row_num);
                cudaMemset(d_over_num, 0, sizeof(int));
                symbolic_spgemm_step1_large<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                           d_over_num, d_over_rid,
                                                                           dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                           dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                           dmatC->blcPtr, blc_m, blc_n, blc_k);
                cudaDeviceSynchronize();
                cudaMemcpy(&over_num, d_over_num, sizeof(int), cudaMemcpyDeviceToHost);
                if (over_num)
                {
                    int *hash_global;
                    cudaMalloc((void **)&hash_global, sizeof(int) * over_num * max_len);
                    cudaMemset(hash_global, -1, sizeof(int) * over_num * max_len);

                    BlockNum = over_num;
                    symbolic_spgemm_step1_gl<<<BlockNum, ThreadNum>>>(d_over_rid, hash_global, max_len,
                                                                      dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                      dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                      dmatC->blcPtr, blc_m, blc_n, blc_k);
                    cudaDeviceSynchronize();
                    cudaFree(hash_global);
                }
                cudaFree(d_over_num);
                cudaFree(d_over_rid);
            }
            break;
            default:
                break;
            }
    }
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, dmatC->blcPtr, dmatC->blcPtr + (blc_m + 1), dmatC->blcPtr, 0);

    int blc_num;
    cudaMemcpy(&blc_num, dmatC->blcPtr + blc_m, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    dmatC->blc_num = blc_num;

    cudaMalloc((void **)&dmatC->blcIdx, sizeof(MAT_IDX_TYPE) * blc_num);

    for (int i = BIN_NUM - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * WARP_NUM_SPGM;
        BlockNum = row_num;

        if (row_num != 0)
            switch (i)
            {
            case 0:
                symbolic_spgemm_step2<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 1:
                symbolic_spgemm_step2<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 2:
                symbolic_spgemm_step2<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                    dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                    dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                    dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 3:
                symbolic_spgemm_step2<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 4:
                symbolic_spgemm_step2<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 5:
                symbolic_spgemm_step2<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 6:
                symbolic_spgemm_step2<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                     dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                     dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                     dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                break;
            case 7:
            {
                int over_num = 0;
                int *d_over_num, *d_over_rid;
                cudaMalloc((void **)&d_over_num, sizeof(int));
                cudaMalloc((void **)&d_over_rid, sizeof(int) * row_num);
                cudaMemset(d_over_num, 0, sizeof(int));
                symbolic_spgemm_step2_large<8192><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                                           d_over_num, d_over_rid,
                                                                           dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                           dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                           dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                cudaDeviceSynchronize();
                cudaMemcpy(&over_num, d_over_num, sizeof(int), cudaMemcpyDeviceToHost);

                if (over_num)
                {
                    int *hash_global;
                    cudaMalloc((void **)&hash_global, sizeof(int) * over_num * max_len);
                    cudaMemset(hash_global, -1, sizeof(int) * over_num * max_len);

                    BlockNum = over_num;
                    symbolic_spgemm_step2_gl<<<BlockNum, ThreadNum>>>(d_over_rid, hash_global, max_len,
                                                                      dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap,
                                                                      dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap,
                                                                      dmatC->blcPtr, dmatC->blcIdx, blc_m, blc_n, blc_k);
                    cudaDeviceSynchronize();
                    cudaFree(hash_global);
                }
                cudaFree(d_over_num);
                cudaFree(d_over_rid);
            }
            break;
            default:
                break;
            }
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_spgemm_kernel_time = 0;
    time_spgemm_kernel_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
// #ifdef PRINT_KERNEL_PERFORMANCE
//     printf("spgemm_kernel_m=%d\n", blc_m);
//     printf("spgemm_kernel_k=%d\n", blc_k);
//     printf("spgemm_kernel_n=%d\n", bsr_n);
//     printf("spgemm_kernel_time=%lf\n", time_spgemm_kernel_time);
// #endif

    MAT_PTR_TYPE blc_nnz = blc_num * BSR_NNZ;
    dmatC->nnz = blc_nnz;
    cudaMalloc((void **)&dmatC->blcMap, sizeof(MAT_MAP_TYPE) * blc_num);
    cudaMemset(dmatC->blcMap, 0, sizeof(MAT_MAP_TYPE) * blc_num);
    cudaMalloc((void **)&dmatC->blcVal_fp32, sizeof(float) * blc_nnz);
    cudaMemset(dmatC->blcVal_fp32, 0, sizeof(float) * blc_nnz);
    cudaMalloc((void **)&dmatC->blcVal, sizeof(MAT_VAL_TYPE) * blc_nnz);
    cudaMemset(dmatC->blcVal, 0, sizeof(MAT_VAL_TYPE) * blc_nnz);

    int ThreadNum_spgemm = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_spgemm_A = (dmatA.nnz + ThreadNum_spgemm - 1) / ThreadNum_spgemm;
    int BlockNum_spgemm_B = (dmatB.nnz + ThreadNum_spgemm - 1) / ThreadNum_spgemm;
    // cudaMalloc((void **)&dmatA.blcVal_fp32, sizeof(float) * dmatA.nnz);
    // cudaMemset(dmatA.blcVal_fp32, 0, sizeof(float) * dmatA.nnz);
    // bsr_val_fp64_to_32<<<ThreadNum_spgemm, BlockNum_spgemm_A>>>(dmatA.blcVal, dmatA.blcVal_fp32, dmatA.nnz);

    // cudaMalloc((void **)&dmatB.blcVal_fp32, sizeof(float) * dmatB.nnz);
    // cudaMemset(dmatB.blcVal_fp32, 0, sizeof(float) * dmatB.nnz);
    // bsr_val_fp64_to_32<<<ThreadNum_spgemm,BlockNum_spgemm_B>>>(dmatB.blcVal,dmatB.blcVal_fp32,dmatB.nnz);

    if (!hypre_MixedPrecisionTag(A))
    {
        hypre_MixedPrecisionTag(A) = 1;
        cudaMalloc((void **)&dmatA.blcVal_fp32, sizeof(float) * dmatA.nnz);
        cudaMemset(dmatA.blcVal_fp32, 0, sizeof(float) * dmatA.nnz);
        bsr_val_fp64_to_32<<<BlockNum_spgemm_A, ThreadNum_spgemm>>>(dmatA.blcVal, dmatA.blcVal_fp32, dmatA.nnz);
    }
    // else{
    //     printf("no need A\n");
    // }

    if (!hypre_MixedPrecisionTag(B))
    {
        hypre_MixedPrecisionTag(B) = 1;
        cudaMalloc((void **)&dmatB.blcVal_fp32, sizeof(float) * dmatB.nnz);
        cudaMemset(dmatB.blcVal_fp32, 0, sizeof(float) * dmatB.nnz);
        bsr_val_fp64_to_32<<<BlockNum_spgemm_B, ThreadNum_spgemm>>>(dmatB.blcVal, dmatB.blcVal_fp32, dmatB.nnz);
    }
    // else{
    //     printf("no need B\n");
    // }
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    // printf("numeric begin 2\n");
    BlockNum = (blc_m + WARP_NUM_SPGM - 1) / WARP_NUM_SPGM;

    numeric_spgemm_hybrid_tf32<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap, dmatA.blcVal_fp32,
                                                        dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap, dmatB.blcVal_fp32,
                                                        dmatC->blcPtr, dmatC->blcIdx, dmatC->blcMap, dmatC->blcVal_fp32,
                                                        blc_m, blc_n, blc_k);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time_spgemm_kernel_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spgemm_kernel_m=%d\n", blc_m);
    printf("spgemm_kernel_k=%d\n", blc_k);
    printf("spgemm_kernel_n=%d\n", blc_n);
    printf("spgemm_kernel_time=%lf\n", time_spgemm_kernel_time);
#endif

    time_spgemm += time_spgemm_kernel_time;
    // numeric_spgemm_hybrid<<<BlockNum, ThreadNum>>>(dmatA.blcPtr, dmatA.blcIdx, dmatA.blcMap, dmatA.blcVal,
    //                                                dmatB.blcPtr, dmatB.blcIdx, dmatB.blcMap, dmatB.blcVal,
    //                                                dmatC->blcPtr, dmatC->blcIdx, dmatC->blcMap, dmatC->blcVal,
    //                                                blc_m, blc_n, blc_k);

    cudaDeviceSynchronize();
    int BlockNum_spgemm_C = (dmatC->nnz + ThreadNum_spgemm - 1) / ThreadNum_spgemm;
    // HYPRE_Real *sum_err;
    // HYPRE_Real sum_err_cpu;
    // cudaMalloc((void **)&sum_err, sizeof(HYPRE_Real));
    // cudaMemset(sum_err, 0.0, sizeof(HYPRE_Real));
    // check_val_fp64_32<<<BlockNum_spgemm_C, ThreadNum_spgemm>>>(dmatC->blcVal, dmatC->blcVal_fp32, sum_err, dmatC->nnz);
    // cudaDeviceSynchronize();

    // cudaMemcpy(&sum_err_cpu, sum_err, sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);
    // printf("begin fuzhi error %lf\n", sum_err_cpu);

    bsr_val_fp32_to_64<<<BlockNum_spgemm_C, ThreadNum_spgemm>>>(dmatC->blcVal, dmatC->blcVal_fp32, dmatC->nnz);

    cudaDeviceSynchronize();
    // BSR2CSR_GPU

    // cudaMemset(sum_err, 0.0, sizeof(HYPRE_Real));
    // check_val_fp64_32<<<BlockNum_spgemm_C, ThreadNum_spgemm>>>(dmatC->blcVal, dmatC->blcVal_fp32, sum_err, dmatC->nnz);
    // cudaDeviceSynchronize();

    // cudaMemcpy(&sum_err_cpu, sum_err, sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);
    // printf("end fuzhi error %lf\n", sum_err_cpu);

    //  csrMAT csrmat;

    gettimeofday(&t1, NULL);
    gettimeofday(&t_start, NULL);
    MAT_PTR_TYPE *d_csrptr = NULL;
    int dmatC_nnz = 0;
    // printf("SpGEMM 0.1\n");
    // cudaMalloc((void **)&d_csrptr, sizeof(HYPRE_Int) * (dmatC->row + 1));
    d_csrptr = hypre_TAlloc(HYPRE_Int, dmatC->row + 1, HYPRE_MEMORY_DEVICE);
    // cudaMalloc((void **)&d_csrptr, sizeof(MAT_PTR_TYPE) * (dmatC->row + 1));
    // printf("SpGEMM 1.1\n");
    cudaMemset(d_csrptr, 0.0, sizeof(MAT_PTR_TYPE) * (dmatC->row + 1));
    // printf("SpGEMM 1\n");
    BSR_TO_CSR_step1(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col, d_csrptr, dmatC->blcPtr, dmatC->blcMap);
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);
    bsr2csr_step1 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
    // printf("SpGEMM 2\n");
    gettimeofday(&t_start, NULL);
    thrust::exclusive_scan(thrust::device, d_csrptr, d_csrptr + (dmatC->row + 1), d_csrptr, 0);
    cudaMemcpy(&(dmatC_nnz), &d_csrptr[dmatC->row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // printf("SpGEMM 3\n");
    gettimeofday(&t_end, NULL);
    bsr2csr_step2 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
    // printf("SpGEMM 4\n");
    gettimeofday(&t_start, NULL);
    MAT_IDX_TYPE *d_csridx = NULL;
    MAT_VAL_TYPE *d_csrval = NULL;
    d_csridx = hypre_TAlloc(HYPRE_Int, dmatC_nnz, HYPRE_MEMORY_DEVICE);
    d_csrval = hypre_TAlloc(HYPRE_Complex, dmatC_nnz, HYPRE_MEMORY_DEVICE);
    // cudaMalloc((void **)&d_csridx, sizeof(MAT_IDX_TYPE) * dmatC_nnz);
    // cudaMalloc((void **)&d_csrval, sizeof(MAT_VAL_TYPE) * dmatC_nnz);
    // printf("SpGEMM 5\n");
    cudaMemset(d_csrval, 0.0, sizeof(MAT_VAL_TYPE) * dmatC_nnz);

    BSR_TO_CSR_step2(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col,
                     d_csrptr, d_csridx, d_csrval,
                     dmatC->blcPtr, dmatC->blcIdx, dmatC->blcVal, dmatC->blcMap);
    // BSR_TO_CSR_step2_fp32(dmatC->row, dmatC->col, dmatC->blc_row, dmatC->blc_col,
    //                       d_csrptr, d_csridx, d_csrval,
    //                       dmatC->blcPtr, dmatC->blcIdx, dmatC->blcVal_fp32, dmatC->blcMap);
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    bsr2csr_step3 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    // printf("SpGEMM 6\n");
    *C_ptr = hypre_CSRMatrixCreate(dmatC->row, dmatC->col, 0);
    hypre_CSRMatrixMemoryLocation(*C_ptr) = HYPRE_MEMORY_DEVICE;
    hypre_CSRMatrixNumNonzeros(*C_ptr) = dmatC_nnz;
    hypre_CSRMatrixI(*C_ptr) = d_csrptr;
    hypre_CSRMatrixJ(*C_ptr) = d_csridx;
    hypre_CSRMatrixData(*C_ptr) = d_csrval;
    // printf("SpGEMM 7\n");
    hypre_BSR(*C_ptr) = dmatC;
    hypre_BSRTAG(*C_ptr) = 1;
    hypre_BSR(A)->blcVal_fp32 = dmatA.blcVal_fp32;
    hypre_BSR(B)->blcVal_fp32 = dmatB.blcVal_fp32;
    // hypre_MixedPrecisionTag(*C_ptr) = 1;

    // printf("SpGEMM 8\n");

    gettimeofday(&t2, NULL);
    time_spgemm_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
}
HYPRE_Int
hypreDevice_CSRSpGemm(hypre_CSRMatrix  *A,
                      hypre_CSRMatrix  *B,
                      hypre_CSRMatrix **C_ptr)
{
    struct timeval t1, t2;
    struct timeval t_start, t_end;

    spgemm_times++;
    // printf("spgemm device\n");
#ifdef Hypre_AMGT

#ifdef MIXED_PRESION
    AMGT_PRECISION precision = get_calculationPrecison(hypre_Level(A));
    // printf("level:%d\n", hypre_Level(A));
    // printf("begin spgemm\n");
    // spgemm_amgT_fp64(A, B, C_ptr);
    // printf("end spgemm\n");
    // printf("spgemm device %d\n", precision);
    if (precision == AMGT_DOUBLE)
    {
        // printf("DOUBLE\n");
        spgemm_amgT_fp64(A, B, C_ptr);
    }
    else if (precision == AMGT_FLOAT)
    {
        // printf("FLOAT\n");
        spgemm_amgT_fp32(A, B, C_ptr);
        // printf("FLOAT end\n");
    }
    else
    {
        // printf("HALF\n");
        spgemm_amgT_fp16(A, B, C_ptr);
        // printf("HALF end\n");
    }
#else
    spgemm_amgT_fp64(A, B, C_ptr);
#endif

#else
    gettimeofday1(&t1, NULL);
    HYPRE_Complex *d_a = hypre_CSRMatrixData(A);
    HYPRE_Int *d_ia = hypre_CSRMatrixI(A);
    HYPRE_Int *d_ja = hypre_CSRMatrixJ(A);
    HYPRE_Int m = hypre_CSRMatrixNumRows(A);
    HYPRE_Int k = hypre_CSRMatrixNumCols(A);
    HYPRE_Int nnza = hypre_CSRMatrixNumNonzeros(A);
    HYPRE_Complex *d_b = hypre_CSRMatrixData(B);
    HYPRE_Int *d_ib = hypre_CSRMatrixI(B);
    HYPRE_Int *d_jb = hypre_CSRMatrixJ(B);
    HYPRE_Int n = hypre_CSRMatrixNumCols(B);
    HYPRE_Int nnzb = hypre_CSRMatrixNumNonzeros(B);
    HYPRE_Complex *d_c;
    HYPRE_Int *d_ic;
    HYPRE_Int *d_jc;
    HYPRE_Int nnzC;
    hypre_CSRMatrix *C;
    // printf("A_m=%d A_n=%d\n",m,k);
    // printf("B_m=%d B_n=%d\n",k,n);

    *C_ptr = C = hypre_CSRMatrixCreate(m, n, 0);
    hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

    /* trivial case */
    if (nnza == 0 || nnzb == 0)
    {
        hypre_CSRMatrixI(C) = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);

        return hypre_error_flag;
    }

#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_SPGEMM] -= hypre_MPI_Wtime();
#endif

#ifdef HYPRE_SPGEMM_TIMING
    hypre_ForceSyncComputeStream(hypre_handle());
    HYPRE_Real ta = hypre_MPI_Wtime();
#endif

    /* use CUSPARSE or rocSPARSE*/
    if (hypre_HandleSpgemmUseVendor(hypre_handle()))
    {
#if defined(HYPRE_USING_CUSPARSE)
        hypreDevice_CSRSpGemmCusparse(m, k, n,
                                      hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                      hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                      hypre_CSRMatrixGPUMatDescr(C), &nnzC, &d_ic, &d_jc, &d_c);
#elif defined(HYPRE_USING_ROCSPARSE)
        hypreDevice_CSRSpGemmRocsparse(m, k, n,
                                       hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                       hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                       hypre_CSRMatrixGPUMatDescr(C), hypre_CSRMatrixGPUMatInfo(C), &nnzC, &d_ic, &d_jc, &d_c);
#elif defined(HYPRE_USING_ONEMKLSPARSE)
        hypreDevice_CSRSpGemmOnemklsparse(m, k, n,
                                          hypre_CSRMatrixGPUMatHandle(A), nnza, d_ia, d_ja, d_a,
                                          hypre_CSRMatrixGPUMatHandle(B), nnzb, d_ib, d_jb, d_b,
                                          hypre_CSRMatrixGPUMatHandle(C), &nnzC, &d_ic, &d_jc, &d_c);
#else
        hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                          "Attempting to use device sparse matrix library for SpGEMM without having compiled support for it!\n");
#endif
    }
    else
    {
        d_a = hypre_CSRMatrixPatternOnly(A) ? NULL : d_a;
        d_b = hypre_CSRMatrixPatternOnly(B) ? NULL : d_b;

        HYPRE_Int *d_rc = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
        const HYPRE_Int alg = hypre_HandleSpgemmAlgorithm(hypre_handle());

        if (hypre_HandleSpgemmNumBin(hypre_handle()) == 0)
        {
            hypreDevice_CSRSpGemmBinnedGetBlockNumDim();
        }

        if (alg == 1)
        {
            printf("333\n");
            hypreDevice_CSRSpGemmRownnz(m, k, n, nnza, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);

            hypreDevice_CSRSpGemmNumerWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, &d_ic, &d_jc, &d_c, &nnzC);
        }
        else /* if (alg == 3) */
        {
            printf("444\n");
            const HYPRE_Int row_est_mtd = hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle());

            hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, row_est_mtd);

            HYPRE_Int rownnz_exact;

            hypreDevice_CSRSpGemmRownnzUpperbound(m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc, &rownnz_exact);

            hypreDevice_CSRSpGemmNumerWithRownnzUpperbound(m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact, &d_ic, &d_jc, &d_c, &nnzC);
        }

        hypre_TFree(d_rc, HYPRE_MEMORY_DEVICE);
    }

#ifdef HYPRE_SPGEMM_TIMING
    hypre_ForceSyncComputeStream(hypre_handle());
    HYPRE_Real tb = hypre_MPI_Wtime() - ta;
    HYPRE_SPGEMM_PRINT("SpGemm time %f\n", tb);
    // printf("SpGemm time %f\n", tb);
#endif

    hypre_CSRMatrixNumNonzeros(C) = nnzC;
    hypre_CSRMatrixI(C) = d_ic;
    hypre_CSRMatrixJ(C) = d_jc;
    hypre_CSRMatrixData(C) = d_c;
    //    hypre_CSRMatrixI(C)           = d_csrptr;
    //    hypre_CSRMatrixJ(C)           = d_csridx;
    //    hypre_CSRMatrixData(C)        = d_csrval;

    //    MAT_PTR_TYPE *RowPtr_1 = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (m + 1));
    //    MAT_IDX_TYPE *ColIdx_1 = (MAT_IDX_TYPE *)malloc(sizeof(MAT_IDX_TYPE) * nnzC);
    //    MAT_VAL_TYPE *Value_1 = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * nnzC);

    //    cudaMemcpy(RowPtr_1, d_csrptr, sizeof(MAT_PTR_TYPE) * (m + 1), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(ColIdx_1, d_csridx, sizeof(MAT_PTR_TYPE) * (nnzC), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(Value_1, d_csrval, sizeof(MAT_VAL_TYPE) * (nnzC), cudaMemcpyDeviceToHost);

    //    cudaMemcpy(d_ic, RowPtr_1, sizeof(MAT_PTR_TYPE) * (m + 1), cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_jc, ColIdx_1, sizeof(MAT_PTR_TYPE) * (nnzC), cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_c, Value_1, sizeof(MAT_VAL_TYPE) * (nnzC), cudaMemcpyHostToDevice);

    // exit(0);
    //    hypre_CSRMatrixI(C)           = d_csrptr;
    //    hypre_CSRMatrixJ(C)           = d_csridx;
    //    hypre_CSRMatrixData(C)        = d_csrval;

    // hypre_CSRMatrixI(*C_ptr)           = d_csrptr;
    // hypre_CSRMatrixJ(*C_ptr)           = d_csridx;
    // hypre_CSRMatrixData(*C_ptr)        = d_csrval;
    //    printf("Check answer.......\n");

    //    MAT_PTR_TYPE *RowPtr_1 = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (row + 1));
    //    cudaMemcpy(RowPtr_1, d_ic, sizeof(MAT_PTR_TYPE) * (row + 1), cudaMemcpyDeviceToHost);

    //    MAT_IDX_TYPE *ColIdx_1 = (MAT_IDX_TYPE *)malloc(sizeof(MAT_IDX_TYPE) * A_nnz);
    //    cudaMemcpy(ColIdx_1, d_jc, sizeof(MAT_PTR_TYPE) * (A_nnz), cudaMemcpyDeviceToHost);

    //    MAT_VAL_TYPE *Value_1 = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * A_nnz);
    //    cudaMemcpy(Value_1, d_c, sizeof(MAT_VAL_TYPE) * (A_nnz), cudaMemcpyDeviceToHost);

    //    for(int i=0;i<row+1;i++){
    //         if(RowPtr_1[i]!=RowPtr[i]){
    //             printf("ptr %d\t%d\t%d\n",i,RowPtr[i],RowPtr_1[i]);
    //         }
    //    }
    //    for(int i=0;i<A_nnz;i++){
    //         if(ColIdx_1[i]!=ColIdx[i]){
    //             printf("idx %d\t%d\t%d\n",i,ColIdx[i],ColIdx_1[i]);
    //         }
    //    }
    //    for(int i=0;i<A_nnz;i++){
    //         if(Value_1[i]!=Value[i]){
    //             printf("val %lf\t%lf\t%lf\n",i,Value[i],Value_1[i]);
    //         }
    //    }

    //    char file_name[] = "C_a.mtx";

    //    FILE *fp = fopen(file_name, "w+");
    //    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    //    fprintf(fp, "%ld ", row);
    //    fprintf(fp, "%ld ", column);
    //    fprintf(fp, "%ld\n", A_nnz);

    //    for (int i = 0; i < row; i++)
    //    {
    //       for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
    //       {
    //          fprintf(fp, "%d %d %lf %d %lf\n", i + 1, ColIdx[j] + 1, Value[j],ColIdx_1[j] + 1, Value_1[j]);
    //       }
    //    }

    //    fclose(fp);

    //    char file_name_1[] = "C_a_1.mtx";

    //    fp = fopen(file_name_1, "w+");
    //    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    //    fprintf(fp, "%ld ", row);
    //    fprintf(fp, "%ld ", column);
    //    fprintf(fp, "%ld\n", A_nnz);

    //    for (int i = 0; i < row; i++)
    //    {
    //       for (int j = RowPtr_1[i]; j < RowPtr_1[i + 1]; j++)
    //       {
    //          fprintf(fp, "%d %ld %f\n", i + 1, ColIdx_1[j] + 1, Value_1[j]);
    //       }
    //    }

    //    fclose(fp);

    //    exit(0);
    //    printf("nnz %d\t%d\n",dmatC_nnz,nnzC);

    //    checkData<<<1,1>>>(m,nnzC,d_ic,d_csrptr,d_jc,d_csridx,d_c,d_csrval);
    // cudaDeviceSynchronize();
    //  exit(0);
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_SPGEMM] += hypre_MPI_Wtime();
#endif
    gettimeofday1(&t2, NULL);
    time_spgemm += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // printf("time_spgemm=%.5lf",time_spgemm);
#endif
    return hypre_error_flag;
}

#endif /* defined(HYPRE_USING_GPU) */

