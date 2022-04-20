#include <stdio.h>
#include <assert.h>

typedef unsigned char uchar;       // 8
typedef unsigned short ushort;     // 16
typedef unsigned long long ullong; // 64

#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X)+BITWIDTH-1)>>LOG_BITWIDTH)
#define FEIL(X) ((((X)+BITWIDTH-1)>>LOG_BITWIDTH)<<LOG_BITWIDTH)

// A faster way to obtain lane id in a warp
#define GET_LANEID              \
    unsigned laneid;            \
    asm("mov.u32 %0, %%laneid;" \
        : "=r"(laneid));

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store64(const void *addr, T a, T b)
{
    *((float2 *)addr) = make_float2(*(float *)(&a), *(float *)(&b));
}
//For higher memory access efficiency

template <typename T>
__device__ __inline__ void store128(const void *addr, T a, T b, T c, T d)
{
    *((float4 *)addr) = make_float4(*(float *)(&a), *(float *)(&b), *(float *)(&c), *(float *)(&d));
}

#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X)+BITWIDTH-1)>>LOG_BITWIDTH)
#define FEIL(X) ((((X)+BITWIDTH-1)>>LOG_BITWIDTH)<<LOG_BITWIDTH)

#define BITWIDTH64 64
#define LOG_BITWIDTH64 6
#define CEIL64(X) (((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)
#define FEIL64(X) ((((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)<<LOG_BITWIDTH64)


//======================================================================================
// bit-packing
//======================================================================================
// for dense unpack
/** @brief Unpack 32-bit row-major unsigned activation matrix into floating-point.
 *
 *  Unpack compact 32-bit unsigned layer output activation matrix into floating-point for 
 *  validation purpose.
 *
 *  @return Void.
 */

// for dense pack
template <typename T>
__global__ void ToBit32RowDenseBin(const T *__restrict__ A, unsigned *B,
                                const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval = 0;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = ((by * 32 + i < A_width) && (bx * 32 + laneid < A_height)) ? A[(bx * 32 + laneid) * A_width + by * 32 + i] : (T)-1;
        Bval = (Bval << 1) + (f0 > 0); // already binarized B
    }
    if (bx * gridDim.y * 32 + laneid * gridDim.y + by < A_height * gridDim.y)
        B[bx * gridDim.y * 32 + laneid * gridDim.y + by] = Bval;
}

template <typename T>
__global__ void ToBit32RowDenseSign(const T *__restrict__ A, unsigned *B, 
                                    const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval = 0;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = ((by * 32 + i < A_width) && (bx * 32 + laneid < A_height)) ? A[(bx * 32 + laneid) * A_width + by * 32 + i] : (T)-1;
        Bval = (Bval << 1) + (f0 >= 0);
    }
    if (bx * gridDim.y * 32 + laneid * gridDim.y + by < A_height * gridDim.y)
        B[bx * gridDim.y * 32 + laneid * gridDim.y + by] = Bval;
}

template <typename T>
__global__ void ToBit32ColUd(const T *__restrict__ A, unsigned *B,
                             const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = ((by * 32 + laneid < A_width) && (bx * 32 + i < A_height)) ? A[(bx * 32 + i) * A_width + by * 32 + laneid] : (T)-1;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 >= 0));
        if (laneid == i)
            Bval = r0;
    }
    if (laneid < A_height * A_width)
        B[by * gridDim.x * 32 + bx * 32 + laneid] = Bval;
}

//======================================================================================
// From column-major 32-bit-array to row-major normal array. No padding.
//======================================================================================
template <typename T>
__global__ void Bit32ColTo(const unsigned *__restrict__ A, T *B,
                           const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Aval = A[by * A_height + bx * 32 + laneid];
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        unsigned r0 = __shfl_sync(0xFFFFFFFF, Aval, i); //from lane-i
        B[(32 * bx + i) * A_width * 32 + by * 32 + laneid] = (T)((r0 >> (31 - laneid)) & 0x1);
    }
}

// col-major packing bit 4
template <typename T>
__global__ void ToBit4Col(const T *__restrict__ A, uchar *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/64)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
    T f0;

#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        f0 = by * 16 * 64 + i * 16 * 2 + laneid < nblocks * 16 ? A[by * 16 * 64 + i * 16 * 2 + laneid] : 0; // <-- laneid will get consecutive 32 (2-blocks)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));                                    //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }

    // layout block0 at high-16
    B[by * 4 * 64 + laneid * 4 * 2] = (Bval & 0xF0000000) >> 28;
    B[by * 4 * 64 + laneid * 4 * 2 + 1] = (Bval & 0x0F000000) >> 24;
    B[by * 4 * 64 + laneid * 4 * 2 + 2] = (Bval & 0x00F00000) >> 20;
    B[by * 4 * 64 + laneid * 4 * 2 + 3] = (Bval & 0x000F0000) >> 16;

    // layout block1 at low-16
    B[by * 4 * 64 + laneid * 4 * 2 + 4] = (Bval & 0x0000F000) >> 12;
    B[by * 4 * 64 + laneid * 4 * 2 + 5] = (Bval & 0x00000F00) >> 8;
    B[by * 4 * 64 + laneid * 4 * 2 + 6] = (Bval & 0x000000F0) >> 4;
    B[by * 4 * 64 + laneid * 4 * 2 + 7] = (Bval & 0x0000000F);
}

// row-major packing bit 4
template <typename T>
__global__ void ToBit4Row(const T *__restrict__ A, uchar *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 4))
    {
        unsigned Bval = 0;
        T f0;

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            if (i % 8 < 4)
                f0 = (T)(0); // high-4 bit remain 0
            else
                f0 = A[bx * 4 * 4 + (i - 4 * ((i / 8) + 1))];

            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx * 4] = (Bval & 0xFF000000) >> 24;
        B[bx * 4 + 1] = (Bval & 0x00FF0000) >> 16;
        B[bx * 4 + 2] = (Bval & 0x0000FF00) >> 8;
        B[bx * 4 + 3] = Bval & 0x000000FF;
    }
}

// col-major packing bit 8
// process 4 8x8x4 at the same time
template <typename T>
__global__ void ToBit8Col(const T *__restrict__ A, uchar *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/16)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = by * 8 * 8 * 4 * 4 + i * 32 + laneid < nblocks * 8 * 8 ? A[by * 8 * 8 * 4 * 4 + i * 32 + laneid] : 0; // <-- laneid will get consecutive 32 (half-block)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));                                             //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }

    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4] = (Bval & 0xFF000000) >> 24;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 1] = (Bval & 0x00FF0000) >> 16;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 2] = (Bval & 0x0000FF00) >> 8;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 3] = Bval & 0x000000FF;
}

// row-major packing bit 8
template <typename T>
__global__ void ToBit8Row(const T *__restrict__ A, uchar *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 4))
    {
        unsigned Bval = 0;

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 8 * 4 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx * 4] = (Bval & 0xFF000000) >> 24;
        B[bx * 4 + 1] = (Bval & 0x00FF0000) >> 16;
        B[bx * 4 + 2] = (Bval & 0x0000FF00) >> 8;
        B[bx * 4 + 3] = Bval & 0x000000FF;
    }
}

// col-major packing bit 16
template <typename T>
__global__ void ToBit16Col(const T *__restrict__ A, ushort *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/4)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = by * 16 * 16 * 4 + i * 16 * 2 + laneid < nblocks * 16 * 16 ? A[by * 16 * 16 * 4 + i * 16 * 2 + laneid] : 0;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0)); //__brev(__ballot(f0>0));

        if (laneid == i)
            Bval = r0;
    }

    B[by * 16 * 4 + laneid * 2] = (Bval & 0xFFFF0000) >> 16;
    B[by * 16 * 4 + laneid * 2 + 1] = (Bval & 0x0000FFFF);
}
// 4 16x16 at the same time

// row-major packing bit 16
template <typename T>
__global__ void ToBit16Row(const T *__restrict__ A, ushort *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 2))
    {
        unsigned Bval = 0;
#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 32 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }

        B[bx * 2] = (Bval & 0xFFFF0000) >> 16;
        B[bx * 2 + 1] = (Bval & 0x0000FFFF);
    }
}

// weight should be col-major packing, layout is 32 * (32*numofblocks)
// input should be row-major packing, layout is whatever it is originally

// col-major packing bit 32
template <typename T>
__global__ void ToBit32Col(const T *__restrict__ A, unsigned *B, const int A_height, const int A_width) // blocksize, nblocks * blocksize
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // nblocks
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = A[by * 32 * 32 + i * 32 + laneid];
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0)); //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }
    B[by * 32 + laneid] = Bval;
}

// row-major packing bit 32
template <typename T>
__global__ void ToBit32Row(const T *__restrict__ A, unsigned *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows)
    {
        unsigned Bval = 0;
#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 32 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx] = Bval;
    }
}

// col-major packing bit 64
template <typename T>
__global__ void ToBit64Col(const T *__restrict__ A, ullong *B, const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; //nblocks
    const unsigned bx = blockIdx.x; // 2 <- set this
    ullong Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = A[by * 64 * 64 + bx * 64 * 32 + i * 64 + laneid];
        T f1 = A[by * 64 * 64 + bx * 64 * 32 + i * 64 + 32 + laneid];
        unsigned r0 = __ballot_sync(0xFFFFFFFF, f0 > 0);
        unsigned r1 = __ballot_sync(0xFFFFFFFF, f1 > 0);

        //        unsigned r0 = __ballot(f0>0);
        //        unsigned r1 = __ballot(f1>0);

        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};"
                     : "=l"(l0)
                     : "r"(r0), "r"(r1)); //lo,hi
        if (laneid == i)
            Bval = __brevll(l0);
    }
    B[by * 64 + bx * 32 + laneid] = Bval;
}

// row-major packing bit 64
template <typename T>
__global__ void ToBit64Row(const T *__restrict__ A, ullong *B, const int A_height, const int A_width, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows)
    {
        GET_LANEID;

        ullong Bval = 0;
#pragma unroll
        for (int i = 0; i < 64; i++)
        {
            T f0 = A[bx * 64 + i];
            Bval = (Bval << 1) | (f0 > 0);
        }
        B[bx] = Bval;
    }
}

//======================================================================================
// spmm kernel
//======================================================================================
// 1.50
__global__ void spmm32_full_1(const unsigned *__restrict__ A, const float *__restrict__ B, float *C,
                              const int *__restrict__ rowptr, const int *__restrict__ colind,
                              const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;
    __shared__ float shared_B[32 * 32 * 7]; // hand-tune

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const unsigned *Asub = &(A[row_start * 32]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 1024 * nBcols]);
        float sum[7] = {0}; // hand-tune

#pragma unroll
        for (int i = 0; i < load; i++)
        {
            unsigned r0 = Asub[i * 32 + laneid];

            for (int j = 0; j < nBcols; j++)
            {
                shared_B[warpid * 32 + laneid * nBcols + j] = Bsub[colindsub[i] * 32 * nBcols + laneid * nBcols + j];
            }
            __syncthreads();

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (shared_B[warpid * 32 * nBcols + k * nBcols + j]);
                }
            }
        }

        // store
        for (int j = 0; j < nBcols; j++)
        {
            Csub[warpid * 32 * nBcols + laneid * nBcols + j] += sum[j];
        }
    }
}

// 32 warp-per-threadblock, no shared // 4.13 // 1.80
__global__ void spmm32_full_2(const unsigned *__restrict__ A, const float *__restrict__ B, float *C,
                              const int *__restrict__ rowptr, const int *__restrict__ colind,
                              const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned warpid = (threadIdx.x >> 5);
    GET_LANEID;
    int row = bx * 32 + warpid;

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const unsigned *Asub = &(A[row_start * 32]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 1024 * nBcols]);
        float sum[7] = {0}; // hand-tune

#pragma unroll
        for (int i = 0; i < load; i++)
        {
            unsigned r0 = Asub[i * 32 + laneid];

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i] * 32 + k * nBcols + j]);
                }
            }
        }

        // store
        for (int j = 0; j < nBcols; j++)
        {
            Csub[warpid * 32 + laneid * nBcols + j] += sum[j];
        }
    }
}

// 1 warp-per-threadblock, no shared // 1.91 // 0.84 // fault
__global__ void spmm32_full(const unsigned *__restrict__ A, const float *__restrict__ B, float *C,
                            const int *__restrict__ rowptr, const int *__restrict__ colind,
                            const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx;

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const unsigned *Asub = &(A[row_start * 32]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 32 * nBcols]);
        float sum[64] = {0}; // hand-tune

#pragma unroll
        for (int i = 0; i < load; i++)
        {
            unsigned r0 = Asub[i * 32 + laneid];

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i] * 32 * nBcols + j * nBcols + k]);
                }
            }
        }

        // store
        for (int j = 0; j < nBcols; j++)
        {
            Csub[bx * 32 + laneid * nBcols + j] += sum[j];
        }
    }
}

// 1 warp-per-threadblock, shared // 0.86
__global__ void spmm32_full_4(const unsigned *__restrict__ A, const float *__restrict__ B, float *C,
                              const int *__restrict__ rowptr, const int *__restrict__ colind,
                              const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx;
    __shared__ float shared_B[32 * 7]; // hand-tune

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const unsigned *Asub = &(A[row_start * 32]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 32 * nBcols]);
        float sum[7] = {0}; // hand-tune

#pragma unroll
        for (int i = 0; i < load; i++)
        {
            unsigned r0 = Asub[i * 32 + laneid];

            for (int j = 0; j < nBcols; j++)
            {
                shared_B[laneid * nBcols + j] = Bsub[colindsub[i] * 32 * nBcols + laneid * nBcols + j];
            }
            __syncthreads();

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (shared_B[k * nBcols + j]);
                }
            }
        }

        // store
        for (int j = 0; j < nBcols; j++)
        {
            Csub[bx * 32 + laneid * nBcols + j] += sum[j];
        }
    }
}

//======================================================================================
// spmm16
//======================================================================================
__global__ void spmm16_full(const ushort *__restrict__ A, const float *__restrict__ B, float *C,
                            const int *__restrict__ rowptr, const int *__restrict__ colind,
                            const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx;

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const ushort *Asub = &(A[row_start * 16]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 16 * nBcols]);
        float sum[64] = {0}; // hand-tune

#pragma unroll
        int i;
        for (i = 0; i < ((load + 4 - 1) / 4) * 4 - 4; i += 4)
        {
            unsigned a0 = Asub[i * 16 + (laneid / 16) * 32 + (laneid % 16)];
            unsigned a1 = Asub[i * 16 + (laneid / 16) * 32 + 16 + (laneid % 16)];
            unsigned r0 = a0 << 16 | a1;

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i + laneid / 8] * 16 * nBcols + j * nBcols + k % 16]);
                }
            }
        }

        // less than 4
        {
            unsigned a0 = i * 16 + (laneid / 16) * 32 + (laneid % 16) < load * 16 ? Asub[i * 16 + (laneid / 16) * 32 + (laneid % 16)] : 0;
            unsigned a1 = i * 16 + (laneid / 16) * 32 + 16 + (laneid % 16) < load * 16 ? Asub[i * 16 + (laneid / 16) * 32 + 16 + (laneid % 16)] : 0;
            unsigned r0 = a0 << 16 | a1;

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i + laneid / 8] * 16 * nBcols + j * nBcols + k % 16]);
                }
            }
        }

        // store
        for (int j = 0; j < nBcols; j++)
        {
            atomicAdd(Csub + bx * 16 + laneid % 16, sum[j]);
        }
    }
}

//======================================================================================
// spmm8
//======================================================================================
__global__ void spmm8_full(const uchar *__restrict__ A, const float *__restrict__ B, float *C,
                           const int *__restrict__ rowptr, const int *__restrict__ colind,
                           const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx;

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 8]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 8 * nBcols]);
        float sum[64] = {0}; // hand-tune

#pragma unroll
        int i;
        for (i = 0; i < ((load + 16 - 1) / 16) * 16 - 16; i += 16)
        {
            unsigned a0 = Asub[i * 8 + (laneid / 8) * 32 + (laneid % 8)];
            unsigned a1 = Asub[i * 8 + (laneid / 8) * 32 + 8 + (laneid % 8)];
            unsigned a2 = Asub[i * 8 + (laneid / 8) * 32 + 16 + (laneid % 8)];
            unsigned a3 = Asub[i * 8 + (laneid / 8) * 32 + 24 + (laneid % 8)];
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i + laneid / 2] * 8 * nBcols + j * nBcols + k % 8]);
                }
            }
        }

        // less than 16
        {
            unsigned a0 = i * 8 + (laneid / 8) * 32 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + (laneid % 8)] : 0;
            unsigned a1 = i * 8 + (laneid / 8) * 32 + 8 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + 8 + (laneid % 8)] : 0;
            unsigned a2 = i * 8 + (laneid / 8) * 32 + 16 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + 16 + (laneid % 8)] : 0;
            unsigned a3 = i * 8 + (laneid / 8) * 32 + 24 + (laneid % 8) < load * 8 ? Asub[i * 8 + (laneid / 8) * 32 + 24 + (laneid % 8)] : 0;
            unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 32; k++)
                {
                    if ((r0 >> (31 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i + laneid / 2] * 8 * nBcols + j * nBcols + k % 8]);
                }
            }
        }

        // store
        for (int j = 0; j < nBcols; j++)
        {
            atomicAdd(Csub + bx * 8 + laneid % 8, sum[j]);
        }
    }
}

//======================================================================================
// spmm4
//======================================================================================
__global__ void spmm4_full(const uchar *__restrict__ A, const float *__restrict__ B, float *C,
                           const int *__restrict__ rowptr, const int *__restrict__ colind,
                           const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx;

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[bx * 4 * nBcols]);
        float sum[64] = {0}; // hand-tune

#pragma unroll
        int i;
        for (i = 0; i < ((load + 32 - 1) / 32) * 32 - 32; i += 32)
        {
            ushort a0 = Asub[i * 4 + (laneid / 4) * 16 + (laneid % 4)];
            ushort a1 = Asub[i * 4 + (laneid / 4) * 16 + 4 + (laneid % 4)];
            ushort a2 = Asub[i * 4 + (laneid / 4) * 16 + 8 + (laneid % 4)];
            ushort a3 = Asub[i * 4 + (laneid / 4) * 16 + 12 + (laneid % 4)];
            ushort r0 = a0 << 12 | a1 << 8 | a2 << 4 | a3;

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 16; k++)
                {
                    if ((r0 >> (15 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i + laneid / 4] * 8 * nBcols + j * nBcols + k % 4]);
                }
            }
        }

        // less than 16
        {
            ushort a0 = i * 4 + (laneid / 4) * 16 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + (laneid % 4)] : 0;
            ushort a1 = i * 4 + (laneid / 4) * 16 + 4 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + 4 + (laneid % 4)] : 0;
            ushort a2 = i * 4 + (laneid / 4) * 16 + 8 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + 8 + (laneid % 4)] : 0;
            ushort a3 = i * 4 + (laneid / 4) * 16 + 12 + (laneid % 4) < load * 4 ? Asub[i * 4 + (laneid / 4) * 16 + 12 + (laneid % 4)] : 0;
            ushort r0 = a0 << 12 | a1 << 8 | a2 << 4 | a3;

            for (int j = 0; j < nBcols; j++)
            {
                for (int k = 0; k < 16; k++)
                {
                    if ((r0 >> (15 - k)) & 0x1)
                        sum[j] += (Bsub[colindsub[i + laneid / 4] * 8 * nBcols + j * nBcols + k % 4]);
                }
            }
        }

        // store
        for (int j = 0; j < nBcols; j++)
        {
            atomicAdd(Csub + bx * 4 + laneid % 4, sum[j]);
        }
    }
}

// A in B2SR-4, B in 4-bin
__global__ void spmm4_bin(const uchar *__restrict__ A, const uchar *__restrict__ B, uchar *C,
                          const int *__restrict__ rowptr, const int *__restrict__ colind,
                          const int nblockrows, const int nBcols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx * 8 + (laneid >> 2); // /4

    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const uchar *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        uchar *Csub = &(C[bx * 8 * nBcols]);
        register unsigned Cm[64] = {0}; // hand-tune

#pragma unroll
        for (int j = 0; j < nBcols; j += 1)
        {
            for (int i = 0; i < load; i += 1)
            {
                unsigned r0 = Asub[i * 4 + laneid % 4];
                unsigned r1 = Bsub[(colindsub[i])];
                Cm[j] += (__popc(r0 & r1) - __popc(r0 & ~r1));
            }
        }

        // store
#pragma unroll
        for (int j = 0; j < nBcols; j += 1)
        {
            unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[j] >= 0);
            uchar temp = (uchar)(((__brev(r2) >> (28 - ((laneid >> 2) * 4))) & 0xF) & 0x0F);
            Csub[(laneid >> 2) * nBcols + j] |= temp;
        }
    }
}
//======================================================================================
// typical spmm
//======================================================================================

// ---------------------------------------------------------------------------
// A in b2sr-4, B in BSTC-32. Binary ops. One outunit.
// ---------------------------------------------------------------------------
// [cora] b2sr: 4.915/csr-float: 66.720/speedup: 13.574
// [pubmed] b2sr: 15.354/csr-float: 147.072/speedup: 9.579
// [citeseer] b2sr: 4.826/csr-float: 54.272/speedup: 11.247
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_1(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C,
                            const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4]);

        __shared__ unsigned B_shared[32]; // 32 * outunit (1)
        register unsigned Ctemp = 0; 

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload B
            B_shared[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            __syncthreads();

            // compute 
            for(int j=0; j<4; j++) 
            {
                Ctemp |= (((Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0F) >> (3-j)) & 0x1) ? B_shared[(laneid/4)*4+j] : 0;
            }
        }
        // store
        atomicOr(Csub+laneid%4, Ctemp);
    }
}

// ---------------------------------------------------------------------------
// A in b2sr-4, B in BSTC-32. Binary ops. Two outunits.
// ---------------------------------------------------------------------------
// [cora] b2sr: 5.830/csr-float: 99.680/speedup: 17.097
// [pubmed] b2sr: 19.616/csr-float: 2218.784/speedup: 113.111
// [citeseer] b2sr: 5.536/csr-float: 84.000/speedup: 15.173
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_2(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C,
                            const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4 * 2]);

        __shared__ unsigned B_shared[64]; // 32 * outunit (2)
        register unsigned Ctemp[2] = {0};

        for (int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload B
            B_shared[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*2+(laneid%4)*2] : 0;
            B_shared[32+laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*2+(laneid%4)*2+1] : 0;
            __syncthreads();

            // compute 
            for(int j=0; j<4; j++) 
            {
                if (((Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0F) >> (3-j)) & 0x1)
                {
                    Ctemp[0] |= B_shared[(laneid/4)*4+j];
                    Ctemp[1] |= B_shared[32+(laneid/4)*4+j];
                }
            }
        }

        // store
        atomicOr(Csub+(laneid%4)*2, Ctemp[0]);
        atomicOr(Csub+(laneid%4)*2+1, Ctemp[1]);
    }
}

//======================================================================================
// spmm with +/- 1 operation
//======================================================================================

// ---------------------------------------------------------------------------
// if-else eval. Parallize the computation by B's column
// ---------------------------------------------------------------------------
// [cora] b2sr: 97.408/csr-float: 65.952/speedup: 0.677
// [pubmed] b2sr: 219.040/csr-float: 115.392/speedup: 0.527
// [citeseer] b2sr: 71.232/csr-float: 49.120/speedup: 0.690
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_1_001(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C,
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned B_shared[32]; // 32 * outunit (1)
        __shared__ int Ctemp[4*32];
        store128<int>(Ctemp+laneid*4, 0, 0, 0, 0); // init shared memory

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            __syncthreads();

            // compute
            for (int k=0; k<4; k++)
            {
                for(int j=0; j<32; j++)
                {
                    if ((((A_shared[j] & 0x0F) >> (3-k)) & 0x1))
                    {
                        if ((B_shared[(j/4)*4+k] >> (31-laneid)) & 0x1) Ctemp[(j%4)*32+laneid]  += 1;
                        else Ctemp[(j%4)*32+laneid] -= 1;
                    }
                }
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[32+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[64+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// if-else eval. Parallize the computation by A's bitvec (with atomic write to C)
// ---------------------------------------------------------------------------
// [cora] b2sr: 85.088/csr-float: 65.376/speedup: 0.768
// [pubmed] b2sr: 192.704/csr-float: 120.960/speedup: 0.628
// [citeseer] b2sr: 60.480/csr-float: 51.424/speedup: 0.850
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_1_002(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                   const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned B_shared[32]; // 32 * outunit (1)
        __shared__ int Ctemp[4*32];
        store128<int>(Ctemp+laneid*4, 0, 0, 0, 0); // init shared memory

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            __syncthreads();

            // compute
            for (int k=0; k<4; k++)
            {
                for(int j=0; j<32; j++)
                {
                    if ((((A_shared[j] & 0x0F) >> (3-k)) & 0x1))
                    {
                        if ((B_shared[(j/4)*4+k] >> (31-laneid)) & 0x1) Ctemp[(j%4)*32+laneid]  += 1;
                        else Ctemp[(j%4)*32+laneid] -= 1;
                    }
                }
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[32+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[64+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). Temp b's bitvec store in register
// ---------------------------------------------------------------------------
// [cora] b2sr: 53.088/csr-float: 71.904/speedup: 1.354
// [pubmed] b2sr: 376.890/csr-float: 148.608/speedup: 0.394
// [citeseer] b2sr: 46.342/csr-float: 56.448/speedup: 1.218
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_1_003(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared[32]; // 32 * outunit (1)
        __shared__ int Ctemp[4*32];
        store128<int>(Ctemp+laneid*4, 0, 0, 0, 0); // init shared memory

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (laneid/4 == 0)
            {
                a[laneid] = (A_shared[0+laneid] & 0x0F) << 28 | (A_shared[4+laneid] & 0x0F) << 24 | (A_shared[8+laneid] & 0x0F) << 20 | (A_shared[12+laneid] & 0x0F) << 16 |
                            (A_shared[16+laneid] & 0x0F) << 12 | (A_shared[20+laneid] & 0x0F) << 8 | (A_shared[24+laneid] & 0x0F) << 4 | (A_shared[28+laneid] & 0x0F);
            }

            // get vertical b
            register unsigned b[32] = {0}; // assume whole
            for(int j=0; j<32; j++)
            {
                b[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[laneid] >> (31-j)) & 0x1)?1:0));
            }

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp[i*32+laneid] += (__popc(a[i] & b[laneid]) - __popc(a[i] & (~b[laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[32+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[64+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). Temp b's bitvec store in shared memory 
// ---------------------------------------------------------------------------
// [cora] b2sr: 30.547/csr-float: 70.336/speedup: 2.303
// [pubmed] b2sr: 75.309/csr-float: 155.680/speedup: 2.067
// [citeseer] b2sr: 22.003/csr-float: 50.944/speedup: 2.315
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_1_004(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared[32]; // 32 * outunit (1)
        __shared__ unsigned b[32];
        __shared__ int Ctemp[4*32];
        store128<int>(Ctemp+laneid*4, 0, 0, 0, 0); // init shared memory

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            b[laneid] = 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (laneid/4 == 0)
            {
                a[laneid] = (A_shared[0+laneid] & 0x0F) << 28 | (A_shared[4+laneid] & 0x0F) << 24 | (A_shared[8+laneid] & 0x0F) << 20 | (A_shared[12+laneid] & 0x0F) << 16 |
                            (A_shared[16+laneid] & 0x0F) << 12 | (A_shared[20+laneid] & 0x0F) << 8 | (A_shared[24+laneid] & 0x0F) << 4 | (A_shared[28+laneid] & 0x0F);
            }

            // transpose B_shared
            for(int j=0; j<32; j++)
            {
                b[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[laneid] >> (31-j)) & 0x1)?1:0));
            }

            if(row == 0 && laneid == 0)
            {

                for (unsigned k = 1 << 31; k > 0; k = k / 2)
                {
                    (b[0] & k) ? printf("1") : printf("0");
                }

                printf("\n");
            }

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp[i*32+laneid] += (__popc(a[i] & b[laneid]) - __popc(a[i] & (~b[laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[32+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[64+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// A*(2H-1). Temp b's bitvec store in shared memory. Add threadsync before store 
// ---------------------------------------------------------------------------
// [cora] b2sr: 30.688/csr-float: 70.560/speedup: 2.299
// [pubmed] b2sr: 75.942/csr-float: 153.120/speedup: 2.016
// [citeseer] b2sr: 22.086/csr-float: 57.088/speedup: 2.585
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_1(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared[32]; // 32 * outunit (1)
        __shared__ unsigned b[32];
        __shared__ int Ctemp[4*32];
        store128<int>(Ctemp+laneid*4, 0, 0, 0, 0); // init shared memory

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            b[laneid] = 0;
            __syncthreads();
            
            // layout A as 8 uchar into 1 unsigned
            if (laneid/4 == 0)
            {
                a[laneid] = (A_shared[0+laneid] & 0x0F) << 28 | (A_shared[4+laneid] & 0x0F) << 24 | (A_shared[8+laneid] & 0x0F) << 20 | (A_shared[12+laneid] & 0x0F) << 16 |
                            (A_shared[16+laneid] & 0x0F) << 12 | (A_shared[20+laneid] & 0x0F) << 8 | (A_shared[24+laneid] & 0x0F) << 4 | (A_shared[28+laneid] & 0x0F);
            }

            // get vertical b
            for(int j=0; j<32; j++)
            {
                b[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[laneid] >> (31-j)) & 0x1)?1:0));
            }

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp[i*32+laneid] += (2 * (__popc(a[i] & b[laneid])) - __popc(a[i]));
            }
        }

        // store
        __syncthreads();
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[32+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[64+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). Temp b's bitvec store in shared memory 
// ---------------------------------------------------------------------------
// [cora] b2sr: 49.331/csr-float: 101.152/speedup: 2.050
// [pubmed] b2sr: 123.686/csr-float: 2212.736/speedup: 17.890
// [citeseer] b2sr: 35.021/csr-float: 84.352/speedup: 2.409
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_2(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4 * 2]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared1[32];
        __shared__ unsigned B_shared2[32];
        __shared__ unsigned b1[32];
        __shared__ unsigned b2[32];
        __shared__ int Ctemp1[4*32];
        __shared__ int Ctemp2[4*32];
        store128<int>(Ctemp1+laneid*4, 0, 0, 0, 0); // init shared memory
        store128<int>(Ctemp2+laneid*4, 0, 0, 0, 0);

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared1[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*2+(laneid%4)*2] : 0;
            B_shared2[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*2+(laneid%4)*2+1] : 0;
            b1[laneid] = 0;
            b2[laneid] = 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (laneid/4 == 0)
            {
                a[laneid] = (A_shared[0+laneid] & 0x0F) << 28 | (A_shared[4+laneid] & 0x0F) << 24 | (A_shared[8+laneid] & 0x0F) << 20 | (A_shared[12+laneid] & 0x0F) << 16 |
                            (A_shared[16+laneid] & 0x0F) << 12 | (A_shared[20+laneid] & 0x0F) << 8 | (A_shared[24+laneid] & 0x0F) << 4 | (A_shared[28+laneid] & 0x0F);
            }

            // get vertical b
            for(int j=0; j<32; j++)
            {
                b1[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1[laneid] >> (31-j)) & 0x1)?1:0));
                b2[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2[laneid] >> (31-j)) & 0x1)?1:0));
            }

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp1[i*32+laneid] += (__popc(a[i] & b1[laneid]) - __popc(a[i] & (~b1[laneid])));
                Ctemp2[i*32+laneid] += (__popc(a[i] & b2[laneid]) - __popc(a[i] & (~b2[laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[32+laneid] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[64+laneid] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[96+laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[32+laneid] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[64+laneid] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
        store128<unsigned>(Csub+4, r4, r5, r6, r7);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). Temp b's bitvec store in shared memory 
// ---------------------------------------------------------------------------
// [reddit] b2sr: 146006.274/csr-float: 6820119.629/speedup: 46.711
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_4_32(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                  const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4 * 4]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared1[32];
        __shared__ unsigned B_shared2[32];
        __shared__ unsigned B_shared3[32];
        __shared__ unsigned B_shared4[32];
        __shared__ unsigned b1[32];
        __shared__ unsigned b2[32];
        __shared__ unsigned b3[32];
        __shared__ unsigned b4[32];
        __shared__ int Ctemp1[4*32];
        __shared__ int Ctemp2[4*32];
        __shared__ int Ctemp3[4*32];
        __shared__ int Ctemp4[4*32];
        store128<int>(Ctemp1+laneid*4, 0, 0, 0, 0); // init shared memory
        store128<int>(Ctemp2+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp3+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp4+laneid*4, 0, 0, 0, 0);

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared1[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*4+(laneid%4)*4] : 0;
            B_shared2[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*4+(laneid%4)*4+1] : 0;
            B_shared3[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*4+(laneid%4)*4+2] : 0;
            B_shared4[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*4+(laneid%4)*4+3] : 0;
            b1[laneid] = 0;
            b2[laneid] = 0;
            b3[laneid] = 0;
            b4[laneid] = 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (laneid/4 == 0)
            {
                a[laneid] = (A_shared[0+laneid] & 0x0F) << 28 | (A_shared[4+laneid] & 0x0F) << 24 | (A_shared[8+laneid] & 0x0F) << 20 | (A_shared[12+laneid] & 0x0F) << 16 |
                            (A_shared[16+laneid] & 0x0F) << 12 | (A_shared[20+laneid] & 0x0F) << 8 | (A_shared[24+laneid] & 0x0F) << 4 | (A_shared[28+laneid] & 0x0F);
            }

            // get vertical b
            for(int j=0; j<32; j++)
            {
                b1[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1[laneid] >> (31-j)) & 0x1)?1:0));
                b2[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2[laneid] >> (31-j)) & 0x1)?1:0));
                b3[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared3[laneid] >> (31-j)) & 0x1)?1:0));
                b4[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared4[laneid] >> (31-j)) & 0x1)?1:0));
            }

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp1[i*32+laneid] += (__popc(a[i] & b1[laneid]) - __popc(a[i] & (~b1[laneid])));
                Ctemp2[i*32+laneid] += (__popc(a[i] & b2[laneid]) - __popc(a[i] & (~b2[laneid])));
                Ctemp3[i*32+laneid] += (__popc(a[i] & b3[laneid]) - __popc(a[i] & (~b3[laneid])));
                Ctemp4[i*32+laneid] += (__popc(a[i] & b4[laneid]) - __popc(a[i] & (~b4[laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[laneid] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[32+laneid] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[64+laneid] >= 0));
        unsigned r12 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[96+laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[laneid] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[32+laneid] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[64+laneid] >= 0));
        unsigned r13 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[96+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[laneid] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[32+laneid] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[64+laneid] >= 0));
        unsigned r14 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[96+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[laneid] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[32+laneid] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[64+laneid] >= 0));
        unsigned r15 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
        store128<unsigned>(Csub+4, r4, r5, r6, r7);
        store128<unsigned>(Csub+8, r8, r9, r10, r11);
        store128<unsigned>(Csub+12, r12, r13, r14, r15);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). Temp b's bitvec store in shared memory 
// ---------------------------------------------------------------------------
// [flickr] b2sr: 8259.943 / csr-float: 161415.710 / speedup: 19.542
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_8_32(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4 * 8]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared1[32];
        __shared__ unsigned B_shared2[32];
        __shared__ unsigned B_shared3[32];
        __shared__ unsigned B_shared4[32];
        __shared__ unsigned B_shared5[32];
        __shared__ unsigned B_shared6[32];
        __shared__ unsigned B_shared7[32];
        __shared__ unsigned B_shared8[32];
        __shared__ unsigned b1[32];
        __shared__ unsigned b2[32];
        __shared__ unsigned b3[32];
        __shared__ unsigned b4[32];
        __shared__ unsigned b5[32];
        __shared__ unsigned b6[32];
        __shared__ unsigned b7[32];
        __shared__ unsigned b8[32];
        __shared__ int Ctemp1[4*32];
        __shared__ int Ctemp2[4*32];
        __shared__ int Ctemp3[4*32];
        __shared__ int Ctemp4[4*32];
        __shared__ int Ctemp5[4*32];
        __shared__ int Ctemp6[4*32];
        __shared__ int Ctemp7[4*32];
        __shared__ int Ctemp8[4*32];
        store128<int>(Ctemp1+laneid*4, 0, 0, 0, 0); // init shared memory
        store128<int>(Ctemp2+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp3+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp4+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp5+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp6+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp7+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp8+laneid*4, 0, 0, 0, 0);

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared1[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8] : 0;
            B_shared2[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+1] : 0;
            B_shared3[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+2] : 0;
            B_shared4[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+3] : 0;
            B_shared5[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+4] : 0;
            B_shared6[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+5] : 0;
            B_shared7[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+6] : 0;
            B_shared8[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+7] : 0;
            b1[laneid] = 0;
            b2[laneid] = 0;
            b3[laneid] = 0;
            b4[laneid] = 0;
            b5[laneid] = 0;
            b6[laneid] = 0;
            b7[laneid] = 0;
            b8[laneid] = 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (laneid/4 == 0)
            {
                a[laneid] = (A_shared[0+laneid] & 0x0F) << 28 | (A_shared[4+laneid] & 0x0F) << 24 | (A_shared[8+laneid] & 0x0F) << 20 | (A_shared[12+laneid] & 0x0F) << 16 |
                            (A_shared[16+laneid] & 0x0F) << 12 | (A_shared[20+laneid] & 0x0F) << 8 | (A_shared[24+laneid] & 0x0F) << 4 | (A_shared[28+laneid] & 0x0F);
            }

            // get vertical b
            for(int j=0; j<32; j++)
            {
                b1[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1[laneid] >> (31-j)) & 0x1)?1:0));
                b2[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2[laneid] >> (31-j)) & 0x1)?1:0));
                b3[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared3[laneid] >> (31-j)) & 0x1)?1:0));
                b4[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared4[laneid] >> (31-j)) & 0x1)?1:0));
                b5[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared5[laneid] >> (31-j)) & 0x1)?1:0));
                b6[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared6[laneid] >> (31-j)) & 0x1)?1:0));
                b7[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared7[laneid] >> (31-j)) & 0x1)?1:0));
                b8[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared8[laneid] >> (31-j)) & 0x1)?1:0));
            }

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp1[i*32+laneid] += (__popc(a[i] & b1[laneid]) - __popc(a[i] & (~b1[laneid])));
                Ctemp2[i*32+laneid] += (__popc(a[i] & b2[laneid]) - __popc(a[i] & (~b2[laneid])));
                Ctemp3[i*32+laneid] += (__popc(a[i] & b3[laneid]) - __popc(a[i] & (~b3[laneid])));
                Ctemp4[i*32+laneid] += (__popc(a[i] & b4[laneid]) - __popc(a[i] & (~b4[laneid])));
                Ctemp5[i*32+laneid] += (__popc(a[i] & b5[laneid]) - __popc(a[i] & (~b5[laneid])));
                Ctemp6[i*32+laneid] += (__popc(a[i] & b6[laneid]) - __popc(a[i] & (~b6[laneid])));
                Ctemp7[i*32+laneid] += (__popc(a[i] & b7[laneid]) - __popc(a[i] & (~b7[laneid])));
                Ctemp8[i*32+laneid] += (__popc(a[i] & b8[laneid]) - __popc(a[i] & (~b8[laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[laneid] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[32+laneid] >= 0));
        unsigned r16 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[64+laneid] >= 0));
        unsigned r24 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[96+laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[laneid] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[32+laneid] >= 0));
        unsigned r17 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[64+laneid] >= 0));
        unsigned r25 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[96+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[laneid] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[32+laneid] >= 0));
        unsigned r18 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[64+laneid] >= 0));
        unsigned r26 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[96+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[laneid] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[32+laneid] >= 0));
        unsigned r19 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[64+laneid] >= 0));
        unsigned r27 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[96+laneid] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp5[laneid] >= 0));
        unsigned r12 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp5[32+laneid] >= 0));
        unsigned r20 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp5[64+laneid] >= 0));
        unsigned r28 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp5[96+laneid] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp6[laneid] >= 0));
        unsigned r13 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp6[32+laneid] >= 0));
        unsigned r21 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp6[64+laneid] >= 0));
        unsigned r29 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp6[96+laneid] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp7[laneid] >= 0));
        unsigned r14 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp7[32+laneid] >= 0));
        unsigned r22 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp7[64+laneid] >= 0));
        unsigned r30 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp7[96+laneid] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp8[laneid] >= 0));
        unsigned r15 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp8[32+laneid] >= 0));
        unsigned r23 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp8[64+laneid] >= 0));
        unsigned r31 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp8[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
        store128<unsigned>(Csub+4, r4, r5, r6, r7);
        store128<unsigned>(Csub+8, r8, r9, r10, r11);
        store128<unsigned>(Csub+12, r12, r13, r14, r15);
        store128<unsigned>(Csub+16, r16, r17, r18, r19);
        store128<unsigned>(Csub+20, r20, r21, r22, r23);
        store128<unsigned>(Csub+24, r24, r25, r26, r27);
        store128<unsigned>(Csub+28, r28, r29, r30, r31);
    }
}



// ---------------------------------------------------------------------------
// further consider shared mem size (96 KB per sm), thread block (32 per sm)
// move some shared mem storage to register file
// max register per thread/thread block
// consider the situation when sm is full and not full
// ---------------------------------------------------------------------------
// [cora] b2sr: 49.331/csr-float: 101.152/speedup: 2.050
// [pubmed] b2sr: 123.686/csr-float: 2212.736/speedup: 17.890
// [citeseer] b2sr: 35.021/csr-float: 84.352/speedup: 2.409
// ---------------------------------------------------------------------------

// not implemented yet

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 64 thrds (2 warps) per tb
// ---------------------------------------------------------------------------
// [reddit] b2sr: 132736.792/ csr-float: 6823527.832/ speedup: 51.406
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_4_64(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx; // only 1 tilerow at a time
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4 * 4]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared[4*32];
        __shared__ unsigned b[4*32];
        __shared__ int Ctemp[4*128];
        store128<int>(Ctemp+(warpid*2)*128+laneid*4, 0, 0, 0, 0); // init shared memory
        store128<int>(Ctemp+(warpid*2+1)*128+laneid*4, 0, 0, 0, 0);

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared[(warpid*2)*32+laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*4+(laneid%4)*4+(warpid*2)] : 0;
            B_shared[(warpid*2+1)*32+laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*4+(laneid%4)*4+(warpid*2+1)] : 0;
            b[(warpid*2)*32+laneid] = 0;
            b[(warpid*2+1)*32+laneid] = 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (threadIdx.x/4 == 0)
            {
                a[threadIdx.x] = (A_shared[0+threadIdx.x] & 0x0F) << 28 | (A_shared[4+threadIdx.x] & 0x0F) << 24 | (A_shared[8+threadIdx.x] & 0x0F) << 20 | (A_shared[12+threadIdx.x] & 0x0F) << 16 |
                                 (A_shared[16+threadIdx.x] & 0x0F) << 12 | (A_shared[20+threadIdx.x] & 0x0F) << 8 | (A_shared[24+threadIdx.x] & 0x0F) << 4 | (A_shared[28+threadIdx.x] & 0x0F);
            }

            // get vertical b
            for(int j=0; j<32; j++)
            {
                b[(warpid*2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[(warpid*2)*32+laneid] >> (31-j)) & 0x1)?1:0));
                b[(warpid*2+1)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[(warpid*2+1)*32+laneid] >> (31-j)) & 0x1)?1:0));
            }
            __syncthreads();

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp[(warpid*2)*128+i*32+laneid] += (__popc(a[i] & b[(warpid*2)*32+laneid]) - __popc(a[i] & (~b[(warpid*2)*32+laneid])));
                Ctemp[(warpid*2+1)*128+i*32+laneid] += (__popc(a[i] & b[(warpid*2+1)*32+laneid]) - __popc(a[i] & (~b[(warpid*2+1)*32+laneid])));
            }
        }

        // store
        __syncthreads();
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[laneid] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[32+laneid] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[64+laneid] >= 0));
        unsigned r12 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[96+laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[128+laneid] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[128+32+laneid] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[128+64+laneid] >= 0));
        unsigned r13 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[128+96+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+laneid] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+32+laneid] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+64+laneid] >= 0));
        unsigned r14 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+96+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+laneid] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+32+laneid] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+64+laneid] >= 0));
        unsigned r15 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
        store128<unsigned>(Csub+4, r4, r5, r6, r7);
        store128<unsigned>(Csub+8, r8, r9, r10, r11);
        store128<unsigned>(Csub+12, r12, r13, r14, r15);
    }
}


// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 64 thrds (2 warps) per tb
// ---------------------------------------------------------------------------
// [flickr] b2sr: 4068.243/ csr-float: 161717.819/speedup: 39.751
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_8_64(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                               const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx; // only 1 tilerow at a time
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4 * 8]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared[8*32]; // 8 slot, 32 each
        __shared__ unsigned b[8*32]; // 8 slot, 32 each
        __shared__ int Ctemp[8*128]; // 8 slot, 4*32 each
        store128<int>(Ctemp+(warpid*4)*128+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp+(warpid*4+1)*128+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp+(warpid*4+2)*128+laneid*4, 0, 0, 0, 0);
        store128<int>(Ctemp+(warpid*4+3)*128+laneid*4, 0, 0, 0, 0);

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0; 
            B_shared[(warpid*4)*32+laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+warpid*4] : 0;
            B_shared[(warpid*4+1)*32+laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+warpid*4+1] : 0;
            B_shared[(warpid*4+2)*32+laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+warpid*4+2] : 0;
            B_shared[(warpid*4+3)*32+laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*8+(laneid%4)*8+warpid*4+3] : 0;
            b[(warpid*4)*32+laneid] = 0;
            b[(warpid*4+1)*32+laneid] = 0;
            b[(warpid*4+2)*32+laneid] = 0;
            b[(warpid*4+3)*32+laneid] = 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (threadIdx.x/4 == 0)
            {
                a[threadIdx.x] = (A_shared[0+threadIdx.x] & 0x0F) << 28 | (A_shared[4+threadIdx.x] & 0x0F) << 24 | (A_shared[8+threadIdx.x] & 0x0F) << 20 | (A_shared[12+threadIdx.x] & 0x0F) << 16 |
                            (A_shared[16+threadIdx.x] & 0x0F) << 12 | (A_shared[20+threadIdx.x] & 0x0F) << 8 | (A_shared[24+threadIdx.x] & 0x0F) << 4 | (A_shared[28+threadIdx.x] & 0x0F);
            }

            // get vertical b
            for(int j=0; j<32; j++)
            {
                b[(warpid*4)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[(warpid*4)*32+laneid] >> (31-j)) & 0x1)?1:0));
                b[(warpid*4+1)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[(warpid*4+1)*32+laneid] >> (31-j)) & 0x1)?1:0));
                b[(warpid*4+2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[(warpid*4+2)*32+laneid] >> (31-j)) & 0x1)?1:0));
                b[(warpid*4+3)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared[(warpid*4+3)*32+laneid] >> (31-j)) & 0x1)?1:0));
            }
            // __syncthreads();

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp[(warpid*4)*128+i*32+laneid] += (__popc(a[i] & b[(warpid*4)*32+laneid]) - __popc(a[i] & (~b[(warpid*4)*32+laneid])));
                Ctemp[(warpid*4+1)*128+i*32+laneid] += (__popc(a[i] & b[(warpid*4+1)*32+laneid]) - __popc(a[i] & (~b[(warpid*4+1)*32+laneid])));
                Ctemp[(warpid*4+2)*128+i*32+laneid] += (__popc(a[i] & b[(warpid*4+2)*32+laneid]) - __popc(a[i] & (~b[(warpid*4+2)*32+laneid])));
                Ctemp[(warpid*4+3)*128+i*32+laneid] += (__popc(a[i] & b[(warpid*4+3)*32+laneid]) - __popc(a[i] & (~b[(warpid*4+3)*32+laneid])));
            }
        }
        __syncthreads();
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[laneid] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[32+laneid] >= 0));
        unsigned r16 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[64+laneid] >= 0));
        unsigned r24 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[96+laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1*128+laneid] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1*128+32+laneid] >= 0));
        unsigned r17 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1*128+64+laneid] >= 0));
        unsigned r25 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1*128+96+laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+laneid] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+32+laneid] >= 0));
        unsigned r18 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+64+laneid] >= 0));
        unsigned r26 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2*128+96+laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+laneid] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+32+laneid] >= 0));
        unsigned r19 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+64+laneid] >= 0));
        unsigned r27 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3*128+96+laneid] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[4*128+laneid] >= 0));
        unsigned r12 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[4*128+32+laneid] >= 0));
        unsigned r20 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[4*128+64+laneid] >= 0));
        unsigned r28 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[4*128+96+laneid] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[5*128+laneid] >= 0));
        unsigned r13 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[5*128+32+laneid] >= 0));
        unsigned r21 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[5*128+64+laneid] >= 0));
        unsigned r29 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[5*128+96+laneid] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[6*128+laneid] >= 0));
        unsigned r14 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[6*128+32+laneid] >= 0));
        unsigned r22 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[6*128+64+laneid] >= 0));
        unsigned r30 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[6*128+96+laneid] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[7*128+laneid] >= 0));
        unsigned r15 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[7*128+32+laneid] >= 0));
        unsigned r23 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[7*128+64+laneid] >= 0));
        unsigned r31 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[7*128+96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
        store128<unsigned>(Csub+4, r4, r5, r6, r7);
        store128<unsigned>(Csub+8, r8, r9, r10, r11);
        store128<unsigned>(Csub+12, r12, r13, r14, r15);
        store128<unsigned>(Csub+16, r16, r17, r18, r19);
        store128<unsigned>(Csub+20, r20, r21, r22, r23);
        store128<unsigned>(Csub+24, r24, r25, r26, r27);
        store128<unsigned>(Csub+28, r28, r29, r30, r31);
    }
}

//======================================================================================
// spmm with +/- 1 operation, two-unit, column-major layout
//======================================================================================

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). Temp b's bitvec store in shared memory 
// ---------------------------------------------------------------------------
// [cora] b2sr: 49.331/csr-float: 101.152/speedup: 2.050
// [pubmed] b2sr: 128.614 (123.686)/csr-float: 2212.736/speedup: <17.890
// [citeseer] b2sr: 34.816/csr-float: 84.352/speedup: 2.409
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_2_colmajor(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                        const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                        const int B_height)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    int row = bx; // only 1 tilerow at a time
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[bx * 4]); // <-- revise here
        unsigned *Csub2 = &(C[bx * 4 + B_height]);

        __shared__ uchar A_shared[32];
        __shared__ unsigned a[4];
        store128<unsigned>(a, 0, 0, 0, 0);
        __shared__ unsigned B_shared1[32];
        __shared__ unsigned B_shared2[32];
        __shared__ unsigned b1[32];
        __shared__ unsigned b2[32];
        __shared__ int Ctemp1[4*32];
        __shared__ int Ctemp2[4*32];
        store128<int>(Ctemp1+laneid*4, 0, 0, 0, 0); // init shared memory
        store128<int>(Ctemp2+laneid*4, 0, 0, 0, 0);

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared[laneid] = (i+laneid/4 < load) ? Asub[(i+(laneid/4))*4+(laneid%4)] : 0;
            B_shared1[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*2+(laneid%4)*2] : 0;
            B_shared2[laneid] = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4*2+(laneid%4)*2+B_height] : 0; // <-- revise here
            b1[laneid] = 0;
            b2[laneid] = 0;
            __syncthreads();

            // layout A as 8 uchar into 1 unsigned
            if (laneid/4 == 0)
            {
                a[laneid] = (A_shared[0+laneid] & 0x0F) << 28 | (A_shared[4+laneid] & 0x0F) << 24 | (A_shared[8+laneid] & 0x0F) << 20 | (A_shared[12+laneid] & 0x0F) << 16 |
                            (A_shared[16+laneid] & 0x0F) << 12 | (A_shared[20+laneid] & 0x0F) << 8 | (A_shared[24+laneid] & 0x0F) << 4 | (A_shared[28+laneid] & 0x0F);
            }

            // get vertical b
            for(int j=0; j<32; j++)
            {
                b1[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1[laneid] >> (31-j)) & 0x1)?1:0));
                b2[j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2[laneid] >> (31-j)) & 0x1)?1:0));
            }

            // compute
            for (int i=0; i<4; i++)
            {
                Ctemp1[i*32+laneid] += (__popc(a[i] & b1[laneid]) - __popc(a[i] & (~b1[laneid])));
                Ctemp2[i*32+laneid] += (__popc(a[i] & b2[laneid]) - __popc(a[i] & (~b2[laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[laneid] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[32+laneid] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[64+laneid] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[96+laneid] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[laneid] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[32+laneid] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[64+laneid] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[96+laneid] >= 0));
        store128<unsigned>(Csub, r0, r2, r4, r6); // <-- revise here
        Csub2[0] = r1;
        Csub2[1] = r3;
        Csub2[2] = r5;
        Csub2[3] = r7;
        // store128<unsigned>(Csub2, r1, r3, r5, r7); // <-- revise here
        // it seems like when the store addr must be 4b-aligned
    }
}

//======================================================================================
// spmm with +/- 1 operation, 1024 trds (kernel remapping)
//======================================================================================
// ---------------------------------------------------------------------------
// [1-unit] popc(r0&r1)-popc(r0&~r1). Temp b transpose with ballot & stored in shared mem
// ---------------------------------------------------------------------------
// [cora] b2sr: 10.854 /csr-float: 70.336/speedup: 6.48
// [pubmed] b2sr: 40.346 /csr-float: 155.680/speedup: 3.85
// [citeseer] b2sr: 8.397 /csr-float: 50.944/speedup: 6.07
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_1_1024(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[row * 4]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared = 0;
        __shared__ unsigned b[32*32];
        register int Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            b[warpid*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b[warpid*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// [1-unit] popc(r0&r1)-popc(r0&~r1). Temp b's transpose do not store in shared mem 
// ---------------------------------------------------------------------------
// [cora] b2sr: 16.179 /csr-float: 70.336/speedup: 4.35
// [pubmed] b2sr: 54.886 /csr-float: 155.680/speedup: 2.84
// [citeseer] b2sr: 11.878 /csr-float: 50.944/speedup: 4.28
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_1_1024_002(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[row * 4]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared = 0; // 32 * outunit (1)
        register unsigned b;
        register int Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            b = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // transpose B_shared
            for(int j=0; j<32; j++)
            {
                b = (b << 1) | (((__shfl_sync(0xFFFFFFFF, B_shared, j) >> (31-laneid)) & 0x1));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b)));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}


// ---------------------------------------------------------------------------
// [2-unit] popc(r0&r1)-popc(r0&~r1). Temp b transpose with ballot & stored in shared mem
// ---------------------------------------------------------------------------
// [cora] b2sr: b2sr: 93.184 / csr-float: 95.424 / speedup: 1.024
// [pubmed] b2sr: 1072.128 / csr-float: 2120.992 / speedup: 1.978
// [citeseer] b2sr: 88.064 / csr-float: 71.392 / speedup: 0.811
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_2_1024(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                    const int B_height)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[row * 4]);
        unsigned *Csub2 = &(C[row * 4 + B_height]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        __shared__ unsigned b1[32*32];
        __shared__ unsigned b2[32*32];
        register int Ctemp1[4] = {0};
        register int Ctemp2[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+B_height] : 0;
            b1[warpid*32+laneid] = 0;
            b2[warpid*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[warpid*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[warpid*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[3] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[0] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[1] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[2] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[3] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
        store128<unsigned>(Csub2, r4, r5, r6, r7);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 32 thrds (1 warp) per tb
// ---------------------------------------------------------------------------
// [reddit] b2sr: 157544.580 (was 132736.792) / csr-float: 6820322.754 / speedup: 43.291
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_4_1024(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                    const int B_height)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start*4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub1 = &(C[row*4]);
        unsigned *Csub2 = &(C[row*4+B_height]);
        unsigned *Csub3 = &(C[row*4+2*B_height]);
        unsigned *Csub4 = &(C[row*4+3*B_height]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        register unsigned B_shared3 = 0;
        register unsigned B_shared4 = 0;
        __shared__ unsigned b1[32*32];
        __shared__ unsigned b2[32*32];
        __shared__ unsigned b3[32*32];
        __shared__ unsigned b4[32*32]; // total 16KB/tb
        register int Ctemp1[4] = {0};
        register int Ctemp2[4] = {0};
        register int Ctemp3[4] = {0};
        register int Ctemp4[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+B_height] : 0;
            B_shared3 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+2*B_height] : 0;
            B_shared4 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+3*B_height] : 0;
            b1[warpid*32+laneid] = 0;
            b2[warpid*32+laneid] = 0;
            b3[warpid*32+laneid] = 0;
            b4[warpid*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
                b3[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared3 >> (31-j)) & 0x1)));
                b4[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared4 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[warpid*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[warpid*32+laneid])));
                Ctemp3[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b3[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b3[warpid*32+laneid])));
                Ctemp4[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b4[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b4[warpid*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[3] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[0] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[1] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[2] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[3] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[0] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[1] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[2] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[3] >= 0));
        unsigned r12 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[0] >= 0));
        unsigned r13 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[1] >= 0));
        unsigned r14 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[2] >= 0));
        unsigned r15 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[3] >= 0));

        store128<unsigned>(Csub1, r0, r1, r2, r3);
        store128<unsigned>(Csub2, r4, r5, r6, r7);
        store128<unsigned>(Csub3, r8, r9, r10, r11);
        store128<unsigned>(Csub4, r12, r13, r14, r15);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 64 thrds (2 warp) per tb
// ---------------------------------------------------------------------------
// [reddit] b2sr: 148748.901 / csr-float: 6817758.789 / speedup: 45.834
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_4_1024_64(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_height)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 16 + warpid/2;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start*4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub1 = &(C[row*4 + ((warpid%2)*2+0)*B_height]);
        unsigned *Csub2 = &(C[row*4 + ((warpid%2)*2+1)*B_height]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        __shared__ unsigned b1[16*64];
        __shared__ unsigned b2[16*64]; // total 8KB/tb
        register int Ctemp1[4] = {0};
        register int Ctemp2[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%2)*2+0)*B_height] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%2)*2+1)*B_height] : 0;
            b1[(warpid/2)*64+(warpid%2)*32+laneid] = 0;
            b2[(warpid/2)*64+(warpid%2)*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[(warpid/2)*64+(warpid%2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[(warpid/2)*64+(warpid%2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[(warpid/2)*64+(warpid%2)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[(warpid/2)*64+(warpid%2)*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[(warpid/2)*64+(warpid%2)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[(warpid/2)*64+(warpid%2)*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[3] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[0] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[1] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[2] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[3] >= 0));

        store128<unsigned>(Csub1, r0, r1, r2, r3);
        store128<unsigned>(Csub2, r4, r5, r6, r7);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 64 thrds (2 warp) per tb
// cannot be 32 thrds becuz # of register > 32 per thrd is not possible
// ---------------------------------------------------------------------------
// [flickr] b2sr: 2470.912 / csr-float: 162366.272 / speedup: 65.711
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_8_1024(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                    const int B_height)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 16 + warpid/2;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start*4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub1 = &(C[row*4 + ((warpid%2)*4+0)*B_height]);
        unsigned *Csub2 = &(C[row*4 + ((warpid%2)*4+1)*B_height]);
        unsigned *Csub3 = &(C[row*4 + ((warpid%2)*4+2)*B_height]);
        unsigned *Csub4 = &(C[row*4 + ((warpid%2)*4+3)*B_height]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        register unsigned B_shared3 = 0;
        register unsigned B_shared4 = 0;
        __shared__ unsigned b1[16*64];
        __shared__ unsigned b2[16*64];
        __shared__ unsigned b3[16*64];
        __shared__ unsigned b4[16*64]; // total 16KB/tb
        register int Ctemp1[4] = {0};
        register int Ctemp2[4] = {0};
        register int Ctemp3[4] = {0};
        register int Ctemp4[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%2)*4+0)*B_height] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%2)*4+1)*B_height] : 0;
            B_shared3 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%2)*4+2)*B_height] : 0;
            B_shared4 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%2)*4+3)*B_height] : 0;
            b1[(warpid/2)*64+(warpid%2)*32+laneid] = 0;
            b2[(warpid/2)*64+(warpid%2)*32+laneid] = 0;
            b3[(warpid/2)*64+(warpid%2)*32+laneid] = 0;
            b4[(warpid/2)*64+(warpid%2)*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);


            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[(warpid/2)*64+(warpid%2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[(warpid/2)*64+(warpid%2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
                b3[(warpid/2)*64+(warpid%2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared3 >> (31-j)) & 0x1)));
                b4[(warpid/2)*64+(warpid%2)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared4 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[(warpid/2)*64+(warpid%2)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[(warpid/2)*64+(warpid%2)*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[(warpid/2)*64+(warpid%2)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[(warpid/2)*64+(warpid%2)*32+laneid])));
                Ctemp3[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b3[(warpid/2)*64+(warpid%2)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b3[(warpid/2)*64+(warpid%2)*32+laneid])));
                Ctemp4[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b4[(warpid/2)*64+(warpid%2)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b4[(warpid/2)*64+(warpid%2)*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[3] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[0] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[1] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[2] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[3] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[0] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[1] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[2] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[3] >= 0));
        unsigned r12 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[0] >= 0));
        unsigned r13 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[1] >= 0));
        unsigned r14 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[2] >= 0));
        unsigned r15 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[3] >= 0));

        store128<unsigned>(Csub1, r0, r1, r2, r3);
        store128<unsigned>(Csub2, r4, r5, r6, r7);
        store128<unsigned>(Csub3, r8, r9, r10, r11);
        store128<unsigned>(Csub4, r12, r13, r14, r15);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 128 thrds (4 warp) per tb
// ---------------------------------------------------------------------------
// [flickr] b2sr: 2305.024 / csr-float: 162299.393 / speedup: 70.411
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_8_1024_128(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                    const int B_height)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 8 + warpid/4;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start*4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub1 = &(C[row*4 + ((warpid%4)*2+0)*B_height]);
        unsigned *Csub2 = &(C[row*4 + ((warpid%4)*2+1)*B_height]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        __shared__ unsigned b1[8*128];
        __shared__ unsigned b2[8*128]; // total 8KB/tb
        register int Ctemp1[4] = {0};
        register int Ctemp2[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%4)*2+0)*B_height] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%4)*2+1)*B_height] : 0;
            b1[(warpid/4)*128+(warpid%4)*32+laneid] = 0;
            b2[(warpid/4)*128+(warpid%4)*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);


            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[(warpid/4)*128+(warpid%4)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[(warpid/4)*128+(warpid%4)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[(warpid/4)*128+(warpid%4)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[(warpid/4)*128+(warpid%4)*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[(warpid/4)*128+(warpid%4)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[(warpid/4)*128+(warpid%4)*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[3] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[0] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[1] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[2] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[3] >= 0));

        store128<unsigned>(Csub1, r0, r1, r2, r3);
        store128<unsigned>(Csub2, r4, r5, r6, r7);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 128 thrds (4 warp) per tb
// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_16_1024(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                     const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                     const int B_height)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 8 + warpid/4;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start*4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub1 = &(C[row*4 + ((warpid%4)*4+0)*B_height]);
        unsigned *Csub2 = &(C[row*4 + ((warpid%4)*4+1)*B_height]);
        unsigned *Csub3 = &(C[row*4 + ((warpid%4)*4+2)*B_height]);
        unsigned *Csub4 = &(C[row*4 + ((warpid%4)*4+3)*B_height]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        register unsigned B_shared3 = 0;
        register unsigned B_shared4 = 0;
        __shared__ unsigned b1[8*128];
        __shared__ unsigned b2[8*128]; 
        __shared__ unsigned b3[8*128];
        __shared__ unsigned b4[8*128];
        register int Ctemp1[4] = {0};
        register int Ctemp2[4] = {0};
        register int Ctemp3[4] = {0};
        register int Ctemp4[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%4)*4+0)*B_height] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%4)*4+1)*B_height] : 0;
            B_shared3 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%4)*4+2)*B_height] : 0;
            B_shared4 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%4)*4+3)*B_height] : 0;
            b1[(warpid/4)*128+(warpid%4)*32+laneid] = 0;
            b2[(warpid/4)*128+(warpid%4)*32+laneid] = 0;
            b3[(warpid/4)*128+(warpid%4)*32+laneid] = 0;
            b4[(warpid/4)*128+(warpid%4)*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);


            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[(warpid/4)*128+(warpid%4)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[(warpid/4)*128+(warpid%4)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
                b3[(warpid/4)*128+(warpid%4)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared3 >> (31-j)) & 0x1)));
                b4[(warpid/4)*128+(warpid%4)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared4 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[(warpid/4)*128+(warpid%4)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[(warpid/4)*128+(warpid%4)*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[(warpid/4)*128+(warpid%4)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[(warpid/4)*128+(warpid%4)*32+laneid])));
                Ctemp3[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[(warpid/4)*128+(warpid%4)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[(warpid/4)*128+(warpid%4)*32+laneid])));
                Ctemp4[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[(warpid/4)*128+(warpid%4)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[(warpid/4)*128+(warpid%4)*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[3] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[0] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[1] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[2] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[3] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[0] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[1] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[2] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[3] >= 0));
        unsigned r12 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[0] >= 0));
        unsigned r13 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[1] >= 0));
        unsigned r14 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[2] >= 0));
        unsigned r15 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp4[3] >= 0));

        store128<unsigned>(Csub1, r0, r1, r2, r3);
        store128<unsigned>(Csub2, r4, r5, r6, r7);
        store128<unsigned>(Csub3, r8, r9, r10, r11);
        store128<unsigned>(Csub4, r12, r13, r14, r15);
    }
}

// ---------------------------------------------------------------------------
// popc(r0&r1)-popc(r0&~r1). 256 thrds (8 warp) per unit, for 768 hidden size
// ---------------------------------------------------------------------------
// [pcqm4m] b2sr:  / csr-float:  / speedup: 
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_op_24(const uchar *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, 
                                 const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                 const int B_height) // 600
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 4 + warpid/8;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start*4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub1 = &(C[row*4 + ((warpid%8)*3+0)*B_height]);
        unsigned *Csub2 = &(C[row*4 + ((warpid%8)*3+1)*B_height]);
        unsigned *Csub3 = &(C[row*4 + ((warpid%8)*3+2)*B_height]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        register unsigned B_shared3 = 0;
        __shared__ unsigned b1[4*256];
        __shared__ unsigned b2[4*256];
        __shared__ unsigned b3[4*256];
        register int Ctemp1[4] = {0};
        register int Ctemp2[4] = {0};
        register int Ctemp3[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%8)*3+0)*B_height] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%8)*3+1)*B_height] : 0;
            B_shared3 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+((warpid%8)*3+2)*B_height] : 0;
            b1[(warpid/8)*256+(warpid%8)*32+laneid] = 0; 
            b2[(warpid/8)*256+(warpid%8)*32+laneid] = 0; //512
            b3[(warpid/8)*256+(warpid%8)*32+laneid] = 0; //768

            // layout A as 8 uchar into 1 unsigned
            // only the first 4 lane are in use
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);


            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[(warpid/8)*256+(warpid%8)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[(warpid/8)*256+(warpid%8)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
                b3[(warpid/8)*256+(warpid%8)*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared3 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[(warpid/8)*256+(warpid%8)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[(warpid/8)*256+(warpid%8)*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[(warpid/8)*256+(warpid%8)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[(warpid/8)*256+(warpid%8)*32+laneid])));
                Ctemp3[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b3[(warpid/8)*256+(warpid%8)*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b3[(warpid/8)*256+(warpid%8)*32+laneid])));
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp1[3] >= 0));
        unsigned r4 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[0] >= 0));
        unsigned r5 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[1] >= 0));
        unsigned r6 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[2] >= 0));
        unsigned r7 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp2[3] >= 0));
        unsigned r8 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[0] >= 0));
        unsigned r9 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[1] >= 0));
        unsigned r10 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[2] >= 0));
        unsigned r11 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp3[3] >= 0));

        store128<unsigned>(Csub1, r0, r1, r2, r3);
        store128<unsigned>(Csub2, r4, r5, r6, r7);
        store128<unsigned>(Csub3, r8, r9, r10, r11);
    }
}

//======================================================================================
// spmm-full 1024 trds
//======================================================================================
// ---------------------------------------------------------------------------
// [outunit 1] no norm, output full (for 2nd layer), for n_classes <=8
// shared mem used 32KB per tb (reach max)
// ---------------------------------------------------------------------------
// [cora] b2sr: 14.746 / csr-float: 46.784 / speedup: 3.173
// [pubmed] b2sr: 48.096 / csr-float: 53.280 / speedup: 1.108
// [citeseer] b2sr: 11.674 / csr-float: 39.712 / speedup: 3.402
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_1_1024_lessthan8(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                                const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                                const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        __shared__ float B_shared[32*32*8];
        register float Ctemp = 0;

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<B_cols; j++)
            {
                unsigned ind = (colindsub[i+(laneid/4)]*4+(laneid%4))*B_cols+j;
                B_shared[warpid*32*8+laneid*8+j] = (i+laneid/4 < load) ? Bsub[ind] : 0;
            }

            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            // every 4 lane work on one B_cols dimension
            // so B_cols cannot exceed 8
            for(int k=0; k<32; k++)
            {
                float fval = B_shared[warpid*32*8+k*8+(laneid/4)]; 
                Ctemp += ((a >> (31-k)) & 0x1)?fval:0;
            }
        }

        // store
        if((laneid/4)<B_cols) Csub[(laneid%4)*B_cols+(laneid/4)] = Ctemp;
    }
}

// ---------------------------------------------------------------------------
// [outunit 1] no norm, output full (for 2nd layer), for n_classes <=8
// ---------------------------------------------------------------------------
// [cora] b2sr: 501.146 csr-float: 52.416 speedup: 0.105
// [pubmed] b2sr: 4466.419 csr-float: 54.464 speedup: 0.012
// [citeseer] b2sr: 475.264 csr-float: 36.160 speedup: 0.076
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_1_1024_002(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float B_shared[8] = {0};
        register float Ctemp = 0;

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<8; j++)
            {
                unsigned ind = (colindsub[i+(laneid/4)]*4+(laneid%4))*B_cols+j;
                B_shared[j] = (i+laneid/4 < load) ? Bsub[ind] : 0;
            }

            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            // every 4 lane work on one B_cols dimension
            // so B_cols cannot exceed 8
            // collect btemp first
            register float fvals[8*32] = {0};
            for(int j=0; j<8; j++)
            {
                for(int k=0; k<32; k++)
                {
                    fvals[j*32+k] = __shfl_sync(0xFFFFFFFF, B_shared[j], k);
                }
            }

            for(int k=0; k<32; k++)
            {
                Ctemp += ((a >> (31-k)) & 0x1)?fvals[(laneid/4)*32+k]:0;
            }
        }

        // store
        if((laneid/4)<B_cols) Csub[(laneid%4)*B_cols+(laneid/4)] = Ctemp;
    }
}

// ---------------------------------------------------------------------------
// [outunit 1] no norm, output full (for 2nd layer), 1 warps
// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_1_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid)*32+laneid < B_cols)
        {
            Csub[(warpid)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid)*32+laneid] = Ctemp[3];
        }
    }
}

// ---------------------------------------------------------------------------
// [outunit 2] no norm, output full (for 2nd layer), 2 warps
// ---------------------------------------------------------------------------
// [reddit] b2sr: 449085.840 / csr-float: 2097836.426 / speedup: 4.671
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_2_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 16 + warpid/2;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid%2)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid%2)*32+laneid < B_cols)
        {
            Csub[(warpid%2)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid%2)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid%2)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid%2)*32+laneid] = Ctemp[3];
        }
    }
}

// ---------------------------------------------------------------------------
// [outunit 4] no norm, output full (for 2nd layer), 4 warps
// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_4_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 8 + warpid/4;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid%4)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid%4)*32+laneid < B_cols)
        {
            Csub[(warpid%4)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid%4)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid%4)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid%4)*32+laneid] = Ctemp[3];
        }
    }
}

// ---------------------------------------------------------------------------
// [outunit 8] no norm, output full (for 2nd layer), 8 warps
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_8_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 4 + warpid/8;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid%8)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid%8)*32+laneid < B_cols)
        {
            Csub[(warpid%8)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid%8)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid%8)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid%8)*32+laneid] = Ctemp[3];
        }
    }
}

//======================================================================================
// spmm with +/- 1 operation, Bin-Fout, B is BSTC col-major, F is row-major
//======================================================================================
// ---------------------------------------------------------------------------
// [outunit 1] no norm, output full (for 2nd layer), 1 warp
// ---------------------------------------------------------------------------
// [cora] b2sr: 11.622 / csr-float: 47.840 / speedup: 4.116
// [pubmed] b2sr: 42.573 / csr-float: 93.984 / speedup: 2.208
// [citeseer] b2sr: 8.774 / csr-float: 35.616 / speedup: 4.059
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_full_1_1024(const uchar *__restrict__ A, const unsigned *__restrict__ B, float *C, 
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                      const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared = 0;
        __shared__ unsigned b[32*32];
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            b[warpid*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b[warpid*32+laneid])));
            }
        }

        // store
        if (laneid < B_cols)
        {
            Csub[laneid] = Ctemp[0];
            Csub[B_cols+laneid] = Ctemp[1];
            Csub[B_cols*2+laneid] = Ctemp[2];
            Csub[B_cols*3+laneid] = Ctemp[3];
        }
    }
}

// ---------------------------------------------------------------------------
// [outunit 2] no norm, output full (for 2nd layer), 1 warp
// ---------------------------------------------------------------------------
// [reddit] b2sr: 59083.777 / csr-float: 2096802.246 / speedup: 35.489
// ---------------------------------------------------------------------------
__global__ void spmm4_bin_full_2_1024(const uchar *__restrict__ A, const unsigned *__restrict__ B, float *C, 
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                      const int nrows, const int B_cols) /*nrows = FEIL(nrows)*/
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const unsigned *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register unsigned B_shared1 = 0;
        register unsigned B_shared2 = 0;
        __shared__ unsigned b1[32*32];
        __shared__ unsigned b2[32*32];
        register float Ctemp1[4] = {0};
        register float Ctemp2[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            B_shared1 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)] : 0;
            B_shared2 = (i+laneid/4 < load) ? Bsub[colindsub[i+(laneid/4)]*4+(laneid%4)+nrows] : 0;
            b1[warpid*32+laneid] = 0;
            b2[warpid*32+laneid] = 0;

            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // transpose B_shared
            for (int j=0; j<32; j++)
            {
                b1[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared1 >> (31-j)) & 0x1)));
                b2[warpid*32+j] = __brev(__ballot_sync(0xFFFFFFFF, ((B_shared2 >> (31-j)) & 0x1)));
            }

            // compute
            for (int j=0; j<4; j++)
            {
                Ctemp1[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b1[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b1[warpid*32+laneid])));
                Ctemp2[j] += (__popc((__shfl_sync(0xFFFFFFFF, a, j)) & b2[warpid*32+laneid]) - __popc((__shfl_sync(0xFFFFFFFF, a, j)) & (~b2[warpid*32+laneid])));
            }
        }

        // store
        // the first 32
        Csub[laneid] = Ctemp1[0];
        Csub[B_cols+laneid] = Ctemp1[1];
        Csub[B_cols*2+laneid] = Ctemp1[2];
        Csub[B_cols*3+laneid] = Ctemp1[3];
        // the remainings
        if (32+laneid < B_cols)
        {
            Csub[32+laneid] = Ctemp2[0];
            Csub[B_cols+32+laneid] = Ctemp2[1];
            Csub[B_cols*2+32+laneid] = Ctemp2[2];
            Csub[B_cols*3+32+laneid] = Ctemp2[3];
        }
    }
}

//======================================================================================
// spmm with +/- 1 operation, two-unit, Fin-Bout, F is row-major, B is BSTC col-major
//======================================================================================
// ---------------------------------------------------------------------------
// [outunit 2] no norm, output full (for 1st layer), use 2 warps
// ---------------------------------------------------------------------------
// [cora] b2sr: 35.891 / csr-float: 101.728 / speedup: 2.834
// [pubmed] b2sr: 311.910 / csr-float: 2315.808 / speedup: 7.425
// [citeseer] b2sr: 31.539 / csr-float: 84.512 / speedup: 2.680
// ---------------------------------------------------------------------------
__global__ void spmm4_full_bin_2_1024(const uchar *__restrict__ A, const float *__restrict__ B, unsigned *C, 
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                      const int nrows) /*nrows = FEIL(nrows)*/
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 16 + warpid/2;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[row*4+(warpid%2)*nrows]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0};
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*64+(warpid%2)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// [outunit 4] no norm, output full (for 1st layer), use 4 warps
// ---------------------------------------------------------------------------
// [reddit] b2sr: 904614.453 / csr-float: 6819791.992 / speedup: 7.539
// ---------------------------------------------------------------------------
__global__ void spmm4_full_bin_4_1024(const uchar *__restrict__ A, const float *__restrict__ B, unsigned *C, 
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                      const int nrows) /*nrows = FEIL(nrows)*/
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 8 + warpid/4;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[row*4+(warpid%4)*nrows]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0};
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*128+(warpid%4)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// [outunit 8] no norm, output full (for 1st layer), use 8 warps
// ---------------------------------------------------------------------------
// [flickr] b2sr: 14556.300 / csr-float: 161960.678 / speedup: 11.127
// ---------------------------------------------------------------------------
__global__ void spmm4_full_bin_8_1024(const uchar *__restrict__ A, const float *__restrict__ B, unsigned *C, 
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                      const int nrows) /*nrows = FEIL(nrows)*/
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 4 + warpid/8;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[row*4+(warpid%8)*nrows]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0};
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*256+(warpid%8)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// [outunit 16] no norm, output full (for 1st layer), use 16 warps
// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------
__global__ void spmm4_full_bin_16_1024(const uchar *__restrict__ A, const float *__restrict__ B, unsigned *C, 
                                      const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                      const int nrows) /*nrows = FEIL(nrows)*/
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 2 + warpid/16;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        unsigned *Csub = &(C[row*4+(warpid%16)*nrows]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0};
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*512+(warpid%16)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[0] >= 0));
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[1] >= 0));
        unsigned r2 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[2] >= 0));
        unsigned r3 = __brev(__ballot_sync(0xFFFFFFFF, Ctemp[3] >= 0));
        store128<unsigned>(Csub, r0, r1, r2, r3);
    }
}