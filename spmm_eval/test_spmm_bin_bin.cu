#include <iostream>
#include <sys/time.h>
#include <bitset>

#define TEST_TIMES 5
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <vector>
#include "backend/readMtx.hpp"
#include "backend/csr2bsr_batch_bsrbmv_new.cu"

// csr metadata
int nrows, ncols, nnz;

// csr host
int *h_csrRowPtr, *h_csrColInd;
float *h_csrVal;

// csr device
int *csrRowPtr, *csrColInd;
float *csrVal;

// csc host
int *h_cscRowInd, *h_cscColPtr;

// csc device
int *cscRowInd, *cscColPtr;
float *cscVal;

// b2sr metadata
int mb, nb, nblockrows;
int nblocks;

#if TILEDIM == 4
int tiledim = 4;
#elif TILEDIM == 8
int tiledim = 8;
#elif TILEDIM == 16
int tiledim = 16;
#elif TILEDIM == 32
int tiledim = 32;
#endif

// b2sr
int *bsrRowPtr, *bsrColInd;

// b2sr val
#if TILEDIM == 32
unsigned *tA;
#elif TILEDIM == 16
ushort *tA;
#else
uchar *tA;
#endif

// result mat
unsigned *fC;

// Bmat host
float *B;
int nBrows, nBcols;
int outunit;

// Bmat device
float *fB;
unsigned *tB;

// cusparse handles
cusparseMatDescr_t csr_descr = 0;
cusparseMatDescr_t bsr_descr = 0;
cudaStream_t streamId = 0;
cusparseHandle_t handle = 0;

/// ======================
void readMtxCSR(const char *filename)
{
    // graphblast mmio interface
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;
    char *dat_name;
    readMtx(filename, &row_indices, &col_indices, &values,
            &nrows, &ncols, &nnz, 0, false, &dat_name); // directed, mtxinfo

    h_csrRowPtr = (int *)malloc(sizeof(int) * (nrows + 1));
    h_csrColInd = (int *)malloc(sizeof(int) * nnz);
    h_csrVal = (float *)malloc(sizeof(float) * nnz);
    coo2csr(h_csrRowPtr, h_csrColInd, h_csrVal,
            row_indices, col_indices, values, nrows, ncols);

    // copy csr to device
    cudaMalloc(&csrRowPtr, sizeof(int) * (nrows + 1));
    cudaMalloc(&csrColInd, sizeof(int) * nnz);
    cudaMalloc(&csrVal, sizeof(float) * nnz);
    cudaMemcpy(csrRowPtr, h_csrRowPtr, sizeof(int) * (nrows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColInd, h_csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(csrVal, h_csrVal, sizeof(float) * nnz, cudaMemcpyHostToDevice);

    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    cudaMemset(csrVal, 1.0, nnz * sizeof(float));
}

void CSR2B2SR()
{
    // transform from csr to bsr using cuSPARSE API
    mb = (nrows + tiledim - 1) / tiledim;
    nb = (ncols + tiledim - 1) / tiledim;
    nblockrows = mb;

    // cuSPARSE API metadata setup
    cusparseCreateMatDescr(&csr_descr);
    cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&bsr_descr);
    cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreate(&handle);
    cusparseSetStream(handle, streamId);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    cudaMalloc((void **)&bsrRowPtr, sizeof(int) * (nblockrows + 1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, tiledim, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void **)&bsrColInd, sizeof(int) * nblocks);

    // malloc packed matrix & pack
#if TILEDIM == 4
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(uchar));
    csr2bsr_batch_4(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                    bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#elif TILEDIM == 8
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(uchar));
    csr2bsr_batch_8(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                    bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#elif TILEDIM == 16
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(ushort));
    csr2bsr_batch_16(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#elif TILEDIM == 32
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(unsigned));
    csr2bsr_batch_32(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
#endif
}

void packBasSign32()
{
    // copy to device
    cudaMalloc(&fB, nrows * nBcols * sizeof(float));
    cudaMemcpy(fB, B, nrows * nBcols * sizeof(float), cudaMemcpyHostToDevice); // the rest are paddings

    // pack B
    outunit = CEIL(nBcols);
    cudaMalloc(&tB, FEIL(nrows) * CEIL(nBcols) * sizeof(unsigned));
    cudaMemset(tB, 0, FEIL(nrows) * CEIL(nBcols) * sizeof(unsigned));
    ToBit32ColUd<float><<<dim3(CEIL(nrows), CEIL(nBcols)), 32>>>(fB, tB, nrows, nBcols);
}

double evalB2SRSpmmBin32()
{
    // init C (result storage)
    cudaMalloc(&fC, FEIL(nrows) * CEIL(nBcols) * sizeof(unsigned));
    cudaMemset(fC, 0, FEIL(nrows) * CEIL(nBcols) * sizeof(unsigned));
    int gridDim = (int)ceil(cbrt((double)nblockrows/32)); // should /32 when use 1 warp
    dim3 grid(gridDim, gridDim, gridDim);

    int gridDim2 = (int)ceil(cbrt((double)nblockrows/16)); // should /16 when use 2 warps
    dim3 grid2(gridDim2, gridDim2, gridDim2);

    int gridDim3 = (int)ceil(cbrt((double)nblockrows/8)); // should /8 when use 4 warps
    dim3 grid3(gridDim3, gridDim3, gridDim3);

    int gridDim4 = (int)ceil(cbrt((double)nblockrows/4)); // should /4 when use 8 warps
    dim3 grid4(gridDim4, gridDim4, gridDim4);


    // ------
    GpuTimer b2sr_timer;
    b2sr_timer.Start();

    for (int i = 0; i < TEST_TIMES; i++)
    {
#if TILEDIM == 4

#if OUTUNIT == 1    
    spmm4_bin_op_1_1024<<<grid, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows);
#elif OUTUNIT == 2
    spmm4_bin_op_2_1024<<<grid, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, FEIL(nrows));
#elif OUTUNIT == 4
    spmm4_bin_op_4_1024<<<grid2, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, FEIL(nrows));
#elif OUTUNIT == 8
    spmm4_bin_op_8_1024_128<<<grid3, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, FEIL(nrows));
#elif OUTUNIT == 16
    spmm4_bin_op_16_1024<<<grid3, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, FEIL(nrows));
#else
    spmm4_bin_op_24<<<grid4, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, FEIL(nrows));
#endif

#elif TILEDIM == 8
        // spmm8_full<<<grid, 32>>>(tA, fB, fC, bsrRowPtr, bsrColInd, nblockrows, nBcols);
#elif TILEDIM == 16
        // spmm16_full<<<grid, 32>>>(tA, fB, fC, bsrRowPtr, bsrColInd, nblockrows, nBcols);
#elif TILEDIM == 32
        // spmm32_full<<<grid, 32>>>(tA, fB, fC, bsrRowPtr, bsrColInd, nblockrows, nBcols);
#endif
    }

    b2sr_timer.Stop();
    double b2sr_time = b2sr_timer.ElapsedMillis() / double(TEST_TIMES);
    // ------

    return b2sr_time;
}

/// ======================
void freeCSR()
{
    // free csr mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);
}

void freeB2SR()
{
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free storage
    cudaFree(tA);

    // free vec mem
    free(B);
    cudaFree(tB);
    cudaFree(fB);
    cudaFree(fC);

    // free indexing sys
    cudaFree(bsrRowPtr);
    cudaFree(bsrColInd);
}

/// ======================
int main(int argc, char *argv[])
{
    char *Amtxfile = argv[1]; // e.g. "G43.mtx"
    nBcols = atoi(argv[2]);

    cudaSetDevice(0);
    readMtxCSR(Amtxfile);
    CSR2B2SR();
    nBrows = nrows; 

    B = (float *)malloc(sizeof(float) * nBrows * nBcols);
    srand(time(0));
    for (int i = 0; i < nBrows * nBcols; i++)
    {
        float x = (float)rand() / RAND_MAX;
        B[i] = (x > 0.5) ? 1 : -1;
    }

    packBasSign32();

    double spmmtime_b2sr = evalB2SRSpmmBin32();
    printf("%.6f,", spmmtime_b2sr); // ms

    freeCSR();
    freeB2SR();
}