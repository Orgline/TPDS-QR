#include "TensorBLAS.h"

int m = 0, n = 0, type = 0;
float ms = 0;

int parseArguments(int argc,char *argv[])
{
    if(argc < 4)
    {
        printf("Needs m, n and type as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    type = atoi(argv[3]);
    return 0;
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    if(type == 0)
    {
        //double precision
        double *dA;
        cudaMalloc(&dA, sizeof(double)*m*n);
        int lwork;
        cusolverDnHandle_t cusolver_handle ;
        cusolverDnCreate(&cusolver_handle);
        cusolverDnDgeqrf_bufferSize(cusolver_handle,
                      m, n, dA, m, &lwork );
        printf("lwork = %d\n", lwork);
        double *work;
        cudaMalloc(&work, sizeof(double)*lwork);
        double *tau;
        cudaMalloc(&tau, sizeof(double)*m);
        int *info;
        cudaMalloc(&info, sizeof(int));
        startTimer();
        cusolverStatus_t status = cusolverDnDgeqrf(cusolver_handle,
           m, n, dA, m, tau, work, lwork, info);
        ms = stopTimer();
        printf("cuSOVLER DGEQRF size %dx%d takes %lf ms, tflops is %lf, status = %d\n",m, n, ms, 2.0*n*n*( m -1.0/3.0*n )/(ms*1e9), status);
    }
    else if(type == 1)
    {
        float *dA;
        cudaMalloc(&dA, sizeof(float)*m*n);
        int lwork;
        cusolverDnHandle_t cusolver_handle ;
        cusolverDnCreate(&cusolver_handle);
        cusolverDnSgeqrf_bufferSize(cusolver_handle,
                      m, n, dA, m, &lwork );
        printf("lwork = %d\n", lwork);
        float *work;
        cudaMalloc(&work, sizeof(float)*lwork);
        float *tau;
        cudaMalloc(&tau, sizeof(float)*m);
        int *info;
        cudaMalloc(&info, sizeof(int));
        startTimer();
        cusolverStatus_t status = cusolverDnSgeqrf(cusolver_handle,
           m, n, dA, m, tau, work, lwork, info);
        ms = stopTimer();
        printf("cuSOVLER SGEQRF size %dx%d takes %lf ms, tflops is %lf, status = %d\n",m, n, ms, 2.0*n*n*( m -1.0/3.0*n )/(ms*1e9), status);
    }
    else if(type == 2)
    {
        double *dA;
        cudaMalloc(&dA, sizeof(double)*m*n);
        int lwork;
        const int blks = m/n;
        cusolverDnHandle_t cusolver_handle[blks];
        cudaStream_t stream[blks];
        int i = 0;
        for(i = 0; i < m/n; i++)
        {
            cusolverDnCreate(&cusolver_handle[i]);
            cudaStreamCreate(&stream[i]);
            cusolverDnSetStream(cusolver_handle[i], stream[i]);
        }
        
        cusolverDnDgeqrf_bufferSize(cusolver_handle[0],
                      n, n, dA, m, &lwork );
        printf("lwork = %d\n", lwork);
        double *work[m/n];
        for(i = 0; i < m/n; i++)
        {
            cudaMalloc(&work[i], sizeof(double)*lwork);
        }
        double *tau[m/n];
        for(i = 0; i < m/n; i++)
        {
            cudaMalloc(&tau[i], sizeof(double)*n);
        }
        int *info[m/n];
        for(i = 0; i < m/n; i++)
        {
            cudaMalloc(&info[i], sizeof(int));
        }
        startTimer();
        for(i = 0; i < m/n; i++)
        {
            cusolverStatus_t status = cusolverDnDgeqrf(cusolver_handle[i],
                n, n, &dA[i*n], m, tau[i], work[i], lwork, info[i]);
        }
        ms = stopTimer();
        printf("cuSOVLER DGEQRF size %dx%d takes %lf ms, tflops is %lf\n",m, n, ms, 2.0*n*n*( m -1.0/3.0*n )/(ms*1e9));
    }
}