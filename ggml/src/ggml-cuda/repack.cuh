#pragma once

#include "common.cuh"

struct block_q8_0_Z64 {
    static constexpr int I = 64;

    half    d[I][2];
    int8_t qs[I][2*QK8_0];
};
static_assert(sizeof(block_q8_0_Z64) == block_q8_0_Z64::I * 2 * sizeof(block_q8_0), "wrong q8_0 block size/padding");

static __global__ void repack_q8_0(const void * __restrict__ vx, void * __restrict__ vy) {
    const block_q8_0 * x = (const block_q8_0 *) vx;
    block_q8_0_Z64   * y = (block_q8_0_Z64   *) vy;

    x += blockIdx.y*block_q8_0_Z64::I*gridDim.x*2 + blockIdx.x*2;
    y += blockIdx.y                  *gridDim.x   + blockIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < block_q8_0_Z64::I; i0 += 32) {
        const int i = i0 + threadIdx.x;

#pragma unroll
        for (int k = 0; k < 2; ++k) {
            y->d[i][k] = x[i*gridDim.x*2 + k].d;
        }
    }

#pragma unroll
    for (int i = 0; i < block_q8_0_Z64::I; ++i) {
#pragma unroll
        for (int k0 = 0; k0 < 2*QK8_0; k0 += 32) {
            y->qs[i][k0 + threadIdx.x] = x[i*gridDim.x*2 + k0/32].qs[threadIdx.x];
        }
    }
}
