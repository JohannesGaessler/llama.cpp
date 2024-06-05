#include "common.cuh"

struct mma_int_A_M16K8 {
    static constexpr int M  = 16;
    static constexpr int K  = 8;
    static constexpr int ne = 4;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_m(const int i) {
        return (i%2)*8 + threadIdx.x/4;
    }

    static __device__ __forceinline__ int get_k(const int i) {
        return (i/2)*4 + threadIdx.x%4;
    }
};

struct mma_int_B_N8K8 {
    static constexpr int N  = 8;
    static constexpr int K  = 8;
    static constexpr int ne = 2;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_n(const int /* i */) {
        return threadIdx.x/4;
    }

    static __device__ __forceinline__ int get_k(const int i) {
        return i*4 + threadIdx.x%4;
    }
};

struct mma_int_C_M16N8 {
    static constexpr int M  = 16;
    static constexpr int N  = 8;
    static constexpr int ne = 4;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_m(const int i) {
        return (i/2)*8 + threadIdx.x/4;
    }

    static __device__ __forceinline__ int get_n(const int i) {
        return 2*(threadIdx.x%4) + i%2;
    }

    __device__ __forceinline__ void mma_K8(const mma_int_A_M16K8 & mma_A, const mma_int_B_N8K8 & mma_B) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_A.x[2]), "r"(mma_A.x[3]), "r"(mma_B.x[0]), "r"(mma_B.x[1]));
    }
};
