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
#include "backend/csr2bsr_batch_bsrbmv.cu"

// bin-full

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

/// ======================
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

// result mat
float* hC;

// Bmat host
int nBrows, nBcols;
int outunit;

// Bmat device
float *B;


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

double evalCSRSpmmFloatCuSPARSE() // cusparse spmm
{
    // // covert B from row-major to col-major
    // B_col_major = (float *)malloc(sizeof(float) * nBrows * nBcols);
    // int cnt = 0;
    // for (int i=0; i<nBcols; i++) 
    // {
    //     for (int j=0; j<nBrows; j++)
    //     {
    //         B_col_major[cnt++] = B[j*nBcols+i];
    //     }
    // }
    // Host problem definition
    int A_num_rows = nrows;
    int A_num_cols = ncols;
    int A_nnz = nnz;
    int B_num_rows = nBrows;
    int B_num_cols = nBcols;
    int ldb = B_num_rows;
    int ldc = A_num_rows;
    int B_size = ldb * B_num_cols;
    int C_size = ldc * B_num_cols;
    int *hA_csrOffsets = h_csrRowPtr;
    int *hA_columns = h_csrColInd;
    float *hA_values = h_csrVal;
    float *hB = B;
    hC = (float *)malloc(sizeof(float) * C_size);
    for (int i = 0; i < C_size; i++)
    {
        hC[i] = 0.0f;
    } 

#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&dC, C_size * sizeof(float)))

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dC, hC, C_size * sizeof(float),
                          cudaMemcpyHostToDevice))
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL))
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    // execute SpMM
    GpuTimer csr_timer;
    csr_timer.Start();
    for (int i = 0; i < TEST_TIMES; i++)
    {
        cusparseSpMM(handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
        // CHECK_CUSPARSE(cusparseSpMM(handle,
        //                     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        //                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
    }
    csr_timer.Stop();
    double cusparsecsrspmmfloat_time = csr_timer.ElapsedMillis() / double(TEST_TIMES);

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    // CHECK_CUDA(cudaMemcpy(hC, dC, C_size * sizeof(float),
    //                       cudaMemcpyDeviceToHost))
    // // hC from col-major to row-major
    // hC_row_major = (float *)malloc(sizeof(float) * C_size);
    // cnt = 0;
    // for (int i=0; i<nBrows; i++) 
    // {
    //     for (int j=0; j<nBcols; j++)
    //     {
    //         hC_row_major[cnt++] = hC[j*nBrows+i];
    //     }
    // }

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))

    return cusparsecsrspmmfloat_time;
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

/// ======================
int main(int argc, char *argv[])
{
    char *Amtxfile = argv[1]; // e.g. "G43.mtx"
    nBcols = atoi(argv[2]);

    cudaSetDevice(0);
    readMtxCSR(Amtxfile);
    nBrows = nrows; 

    B = (float *)malloc(sizeof(float) * nBrows * nBcols);
    srand(time(0));
    for (int i = 0; i < nBrows * nBcols; i++)
    {
        float x = (float)rand() / RAND_MAX;
        B[i] = (x > 0.5) ? 1 : -1;
    }

    double spmmtime_baseline = evalCSRSpmmFloatCuSPARSE();
    printf("%.6f,", spmmtime_baseline); // ms

    freeCSR();
    free(B);
}
