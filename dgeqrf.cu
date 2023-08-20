#include "TensorBLAS.h"

int m, n, nb;

int parseArguments(int argc,char *argv[])
{
    if(argc < 4)
    {
        printf("Needs m, n and nb as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    return 0;
}

void panelQR(cublasHandle_t handle, long m, long n, double *A, long lda, double *R, long ldr, double *work)
{
    if(n<=NMIN)
    {
        hou_caqr_panel<256,32>(ctxt, m, n, A, lda, R, ldr, work);
        dim3 gridDim((m+31)/32,(n+31)/32);
        dim3 blockDim(32,32);
        setEye<<<gridDim,blockDim>>>(m,n,W,ldw);
        sSubstract(ctxt.cublas_handle,m,n,A,lda,W,ldw);
        deviceCopy<<<gridDim,blockDim>>>( m, n, A, lda, W, ldw );

        reconstructY(ctxt,m,n,A,U,lda);

        float sone = 1.0;

        cublasStrsm(ctxt.cublas_handle,
            CUBLAS_SIDE_RIGHT,  CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,  CUBLAS_DIAG_UNIT,
            m, n,
            &sone,
            A, lda,
            W, ldw
        );
        return;
    }
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    double *dA;
    cudaMalloc(&dA, sizeof(double)*m*n);
    int lwork;
    cusolverDnHandle_t cusolver_handle ;
    cusolverDnCreate(&cusolver_handle);
    cusolverDnDgeqrf_bufferSize(cusolver_handle,
                      m, n, dA, m, &lwork );
    printf("lwork = %d\n", lwork);
    startTimer();
    double *work;
    cudaMalloc(&work, sizeof(double)*lwork);
    double *tau;
    cudaMalloc(&tau, sizeof(double)*m);
    int *info;
    cudaMalloc(&info, sizeof(int));
    int restm = m;
    int restn = n;
    int lda = m;
    float ms = stopTimer();
    printf("malloc takes %fms\n", ms);
    startTimer();
    for(int i = 0; i<n;i+=nb)
    {
        
        restn -= nb;
        int inb = min(n - i, nb);
        // cusolverDnDgeqrf_bufferSize(cusolver_handle,
        //               restm, inb, dA+i+i*lda, lda, &lwork );
        printf("i = %d, inb = %d, lwork = %d\n",i, inb, lwork);
        auto status = cusolverDnDgeqrf(cusolver_handle, restm, inb, dA+i+i*lda, lda, tau, work, lwork, info);
        // if(i+nb<n)
        // {
        //     status = cusolverDnDormqr(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        //                         restm, restn, nb, dA+i*lda, lda,
        //                         tau, dA+i+(i+nb)*lda, lda, work, lwork, info);
        // }
       
        restm-=nb;
        
    }
     ms = stopTimer();
    printf("cuSOVLER DGEQRF size %dx%d takes %f ms, tflops is %f\n",m, n, ms, 2.0*n*n*( m -1.0/3.0*n )/(ms*1e9));
    //printf("cuSOVLER DGEQRF size %dx%d takes %f ms, tflops is %f\n",m, n, ms, 2.0*n*n*( m -1.0/3.0*n )/(ms*1e9));
}