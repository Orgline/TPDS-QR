#include "TensorBLAS.h"

long int m, n, nb;

float ms = 0;
bool check=false;

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

__inline__ __device__ double warpAllReduceSum(double val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// launch parameters: M*N=256*32, blockdim: 32*16
template<long int M, long int N>
__global__ void hou_kernel3(long int m, long int n, double *AA, long int lda, double *RR, long int ldr )
{

    long int mm = m - blockIdx.x*M; // TB local number of rows
    mm = (mm < M) ? mm : M;

    if (mm <= 0) return;

    const long int mnmin = (mm<n) ? mm : n;

    double *A = &AA[blockIdx.x*M];
    double *R = &RR[blockIdx.x*N];
    __shared__ double As[M*N], Rs[N];
    const long int ldas = M/*, ldrs = N*/;

//    float acc0, acc1, acc2, acc3, acc4,acc5, acc6, acc7;
    double acc[8];
    const long int i=threadIdx.x, j=threadIdx.y;

//#define R07(OP) {OP(0);OP(1);OP(2);OP(3);OP(4);OP(5);OP(6);OP(7);}
//#define M1(it) if(threadIdx.x+it*32<mm) As[threadIdx.x+it*32+threadIdx.y*ldas] = A[threadIdx.x+it*32+threadIdx.y*lda]

//#pragma unroll 2
    for (int k=0; k<8; k++) {
        // FIXME: What if n < 32?
        if(i+k*32<mm) As[i+k*32+j*ldas] = A[i+k*32+j*lda];
        if(i+k*32<mm) As[i+k*32+(j+16)*ldas] = A[i+k*32+(j+16)*lda];
    }

    __syncthreads();

    for (int k=0; k<mnmin; k++) {
        // reference: house_gen.m and house_qr from Cleve Moler blog.
        double nu = 0;

        if(threadIdx.y==k%16) { // threadIdx.y is the warpId; each warp takes two columns
#pragma unroll
            for(int it=0; it<8; it++) {
                (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k) ?
                (acc[it] = As[threadIdx.x + it * 32 + k * ldas] *
                           As[threadIdx.x + it * 32 + k * ldas]) :
                 acc[it] = 0;
            }
            nu = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);

            double normxsqr = (warpAllReduceSum(nu));
            double normx = sqrt(normxsqr);

            double scale = 1.0/normx;

#pragma unroll
            for(int it=0; it<8; it++) {
                if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)
                    As[threadIdx.x+it*32+k*ldas] *= scale;
            }
            __syncwarp();
            if(threadIdx.x==k) {
                double u1 = As[k+k*ldas];

                As[k+k*ldas] += (u1>=0) ? 1 : -1;
                Rs[k] = (u1>=0)? -normx :normx;
            }
            __syncwarp();
            scale = 1.0/sqrt(abs(As[k+k*ldas]));
#pragma unroll
            for(int it=0; it<8; it++) {
                if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)
                    As[threadIdx.x+it*32+k*ldas] *= scale;
            }

            __syncwarp();
        }
        __syncthreads();
        if(threadIdx.y>k) {
            double uxl = 0;
#pragma unroll
            for(int it=0; it<8; it++) {
                (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)?
                acc[it]= As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+k*ldas]:
                        acc[it] = 0;
            }
            uxl = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            double ux = warpAllReduceSum(uxl);

#pragma unroll
            for(int it=0; it<8; it++) {
                if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)
                    As[threadIdx.x+it*32+threadIdx.y*ldas] -= ux * As[threadIdx.x+it*32+k*ldas];
            }
        }
        if(16+threadIdx.y>k) {
            double uxl = 0;
#pragma unroll
            for(int it=0; it<8; it++) {
                (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)?
                        acc[it]= As[threadIdx.x+it*32+(16+threadIdx.y)*ldas] * As[threadIdx.x+it*32+k*ldas]:
                        acc[it] = 0;
            }
            uxl = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            double ux = warpAllReduceSum(uxl);

#pragma unroll
            for(int it=0; it<8; it++) {
                if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)
                    As[threadIdx.x+it*32+(16+threadIdx.y)*ldas] -= ux * As[threadIdx.x+it*32+k*ldas];
            }
        }
    }

    __syncthreads();

    // write to R
#pragma unroll
    for (int it=0; it<2; it++) {
        int j = it*16+threadIdx.y;
        int i = threadIdx.x;
        if (i == j)
            R[i + i * ldr] = Rs[i];
        else if (i < j) {
            R[i + j * ldr] = As[i + j * ldas];
            As[i + j * ldas] = 0;

        } else if (i < n) {
            R[i + j * ldr] = 0;
        }
    }



    // compute explict Q from Householder reflectors
    double Q[8*2];
#pragma unroll
    for (int k=0; k<8; k++) {
        Q[k] = 0;
        Q[k+8] = 0;
    }
    if(i==j) Q[0] = 1.0;
    if(i==j+16) Q[8] = 1.0;

    for (int k=mnmin-1; k>=0; k--) {
        double acc[8];
        if(threadIdx.y>=k) {
            double accum = 0;
            for (int l=0; l<8; l++)
                accum += As[i+l*32+k*ldas] * Q[l];
            double vq = warpAllReduceSum(accum);

            for (int l=0; l<8; l++)
                if (i+32*l<mm) Q[l] -= vq*( As[i+32*l + k*ldas] );

        }
        if(threadIdx.y+16>=k) {
            double accum = 0;
            for (int l=0; l<8; l++)
                accum += As[i+l*32+k*ldas] * Q[l+8];

            double vq = warpAllReduceSum(accum);
            for (int l=0; l<8; l++)
                if (i+32*l<mm) Q[l+8] -= vq*( As[i+32*l + k*ldas] );
        }
    }


#pragma unroll
    for (int k=0; k<8; k++) {
        if (i+k*32<mm) A[i+k*32 + j*lda] = Q[k];
        if (i+k*32<mm) A[i+k*32 + (j+16)*lda] = Q[k+8];
    }


}

template<long int M, long int N>
void hou_caqr_panel(cublasHandle_t handle, long int m, long int n, double *A, long int lda, double *R, long int ldr, double *work)
{
    dim3 blockdim(32, 16);
    if ( m <= M ) {
        hou_kernel3<M, N><<<1,blockdim>>>(m, n, A, lda, R, ldr);
        return;
    }
    if ( (m-m/M*M)%N != 0) {
        printf("Error: m must be i*%d + j*%d\n", M, N);
    }
    long int NB = (m+M-1)/M;
    long int ldwork = NB*N;
    long int mm = NB*N;
    hou_kernel3<M,N><<<NB,blockdim>>>(m, n, A, lda, work, ldwork);

    hou_caqr_panel<M,N>(handle, mm, n, work, ldwork, R, ldr,  work+ldwork*n );
    double done = 1.0, dzero = 0.0;
    auto status = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, N,
                              &done, A, lda, M,
                              work, ldwork, N,
                              &dzero, A,lda, M,
                              m/M);
    
    mm = m%M;
    if (mm>0) {
        auto status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    mm, N, N, &done, &A[m/M*M], lda, &work[m/M*N], ldwork,
                    &dzero, &A[m/M*M], lda);
    }

}

void generateUniformMatrix(double *dA,long int m,long int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen,dA,long(m*n));
}

double dnorm(long int m, long int n, double* dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    double sn;
    int incx = 1;
    cublasDnrm2(handle, m*n, dA, incx, &sn);
    cublasDestroy(handle);
    return sn;
}

void checkResult(long int m,long int n, double* A, long int lda, double *Q, long int ldq, double *R, int ldr)
{
    double normA = dnorm(m,n,A);
    double alpha = 1.0;
    double beta = -1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // printMatrixDeviceBlock("oriA.csv",m,n,A,lda);
    // printMatrixDeviceBlock("QQ.csv",m,n,Q,ldq);
    // printMatrixDeviceBlock("RR.csv",n,n,R,ldr);
    startTimer();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, n, &alpha, Q, ldq, R, ldr,
                    &beta, A, lda);
    
    float ms = stopTimer();
    printf("DGEMM m*n*k %d*%d*%d takes %.0f (ms), exec rate %.0f TFLOPS\n",
            m, n, n, ms, 2.0*m*n*n/(ms*1e9));
    // printMatrixDeviceBlock("res.csv",m,n,A,lda);
    double normRes = dnorm(m,n,A);
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n",normRes/normA);
}

__global__
void setEye(int m, int n, double *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i<m && j<n) 
    {
        if(i==j)
            a[i+j*lda] = 1.0;
        else
            a[i+j*lda] = 0.0;
	}
}

__global__
void deviceCopy( long m, long n, double *dB, long ldb, double *dA, long lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		dB[i+j*ldb] = dA[i+j*lda];
	}
}

void dorgqr(int m, int n, double* W, int ldw, double* Y, int ldy, double* work)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    double sone = 1.0;
    double snegone = -1.0;

    dim3 grid1( (m+1)/32, (n+1)/32 );
	dim3 block1( 32, 32 );
    setEye<<<grid1,block1>>>(m, n, work, m);
    // cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,n,n,
    //     &snegone,W,CUDA_R_32F, ldw, Y, CUDA_R_32F, ldy,
    //     &sone, work, CUDA_R_32F, m, CUDA_R_32F,
    //     CUBLAS_GEMM_DEFAULT
    // );
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, n, &snegone, W, ldw, Y, ldy,
                    &sone, work, m);
    cublasDestroy(handle);
    deviceCopy<<<grid1,block1>>>( m, n, W, ldw, work, m);
}

__global__
void minusEye( long int m, long int n, double *a, long int lda, double *w, long ldw)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
		{
            a[i+j*lda] = 1.0 - a[i+j*lda];
            w[i+j*ldw] = a[i+j*lda];
        }
		else
		{
            a[i+j*lda] = 0.0 - a[i+j*lda];
            w[i+j*ldw] = a[i+j*lda];
        }
	}
}

// get U from LU factorization
__global__
void getU(int m, int n, double *a, int lda, double *u, int ldu)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i<m && j<n) 
    {
        if (i>j)
            u[i+j*ldu]  = 0;
        else 
            u[i+j*ldu] = a[i+j*lda];
	}
}

// get L from LU factorization
__global__
void getL(int m, int n, double *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i<m && j<n) 
    {
        if (i<j)
            a[i+j*lda] = 0;
        else if (i==j)
            a[i+j*lda] = 1;
	}
}

void reconstructY(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, long m, long n, double* dA, long lda, double *U, double *work, int *info)
{
    cusolverDnDgetrf(cusolver_handle, m, n, dA, lda,
                     work, NULL, info);

    // dim3 gridDim((n+31)/32,(n+31)/32);
    // dim3 blockDim(32,32);
    // getU<<<gridDim,blockDim>>>(n,n,dA,lda,U,n);
    //getL<<<gridDim, blockDim>>>(n,n,dA,lda);

    // double done = 1.0;
    // cublasDtrsm(cublas_handle,
    //     CUBLAS_SIDE_RIGHT,  CUBLAS_FILL_MODE_UPPER,
    //     CUBLAS_OP_N,  CUBLAS_DIAG_NON_UNIT,
    //     m-n, n,
    //     &done,
    //     U, n,
    //     dA+n, lda
    // );
}

__global__
void setZero(long m, long n, double *I, long ldi)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) 
    {
        I[i+j*ldi] = 0.0;
    }
}

float kernel_ms = 0;
float y_ms = 0;
float dtrsm_ms = 0;
float gemm_ms = 0;


void panelQR(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, long m, long n, double *A, long lda, double *W, long ldw, double *R, long ldr, double *work, int *info)
{
    if(n<=32)
    {
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) 
        // {
        //     printf("CUDA error: %s\n", cudaGetErrorString(err));
        //     return;
        // }
        startTimer();
        
        hou_caqr_panel<128,32>(cublas_handle, m, n, A, lda, R, ldr, work);
        kernel_ms += stopTimer();

        dim3 gridDim((m+31)/32,(n+31)/32);
        dim3 blockDim(32,32);
        
        // setEye<<<gridDim,blockDim>>>(m,n,W,ldw);
        // sSubstract(handle,m,n,A,lda,W,ldw);
        
        minusEye<<<gridDim, blockDim>>>(m, n, A, lda, W, ldw);
        //deviceCopy<<<gridDim,blockDim>>>( m, n, A, lda, W, ldw );
        
        startTimer();
        reconstructY(cusolver_handle, cublas_handle, m, n, A, lda, work, work+n*n, info);
        y_ms += stopTimer();
        getL<<<gridDim, blockDim>>>(n,n,A,lda);
        
        double done = 1.0;
        startTimer();
        cublasDtrsm(cublas_handle,
            CUBLAS_SIDE_RIGHT,  CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,  CUBLAS_DIAG_UNIT,
            m, n,
            &done,
            A, lda,
            W, ldw
        );
        dtrsm_ms+=stopTimer();
        return;
    }
    panelQR(cusolver_handle, cublas_handle, m, n/2, A, lda, W, ldw, R, ldr, work, info);

    double done = 1.0, dzero = 0.0,dnegone = -1.0;
    startTimer();
    cublasDgemm(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n/2,n/2,m,
            &done,
            W, ldw,
            A+lda/2*n,lda,
            &dzero,
            work,n/2
        );
    cublasDgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m,n/2,n/2,
            &dnegone,
            A, lda,
            work,n/2,
            &done,
            A+lda/2*n,lda
        );
    gemm_ms+=stopTimer();
    dim3 gridDim1((n/2+31)/32,(n/2+31)/32);
    dim3 blockDim1(32,32);
    deviceCopy<<<gridDim1,blockDim1>>>(n/2, n/2, R+ldr/2*n, ldr, A+lda/2*n, lda);
    setZero<<<gridDim1,blockDim1>>>(n/2,n/2,A+lda/2*n,lda);
    panelQR(cusolver_handle, cublas_handle ,m-n/2, n/2, A+lda/2*n+n/2, lda, W+ldw/2*n+n/2, ldw, R+n/2*ldr+n/2, ldr,work, info);
    // printf("hrere\n");
    // printMatrixDeviceBlock("R--.csv",n/2,n/2,R+n/2*ldr+n/2,n);

    

    
    startTimer();
    cublasDgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n/2,n/2,m,
                &done,
                A, lda,
                W+ldw/2*n,ldw,
                &dzero,
                work,n/2
    );
    cublasDgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m,n/2,n/2,
                &dnegone,
                W, ldw,
                work,n/2,
                &done,
                W+ldw/2*n,ldw
            );
    gemm_ms+=stopTimer();
    return;
}

__global__
void copyAndClear( long int m, long int n, double *da, int lda, double *db, int ldb )
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
        db[i+j*ldb] = da[i+j*lda];
        da[i+j*lda] = 0.0;
	}
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;

    double *A;
    cudaMalloc(&A, sizeof(double)*m*n);
    double *work;
    cudaMalloc(&work, sizeof(double)*m*n);
    double *R;
    cudaMalloc(&R, sizeof(double)*n*n);
    double *W;
    cudaMalloc(&W, sizeof(double)*m*n);
    double *oriA;

    int *info;
    cudaMalloc (&info, sizeof(int));

    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
    cusolverDnCreate(&cusolver_handle);
    cublasCreate(&cublas_handle);

    generateUniformMatrix(A,m,n);
    if(check)
    {
        cudaMalloc(&oriA, sizeof(double)*m*n);
        cudaMemcpy(oriA, A, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
    }
    //printMatrixDeviceBlock("A.csv",m,n,A,m);
    double done = 1.0, dzero = 0.0,dnegone = -1.0;
    //startTimer();
    //hou_caqr_panel<128,32>(handle, m, n, A, m, R, n, work);
    
    for(int i = 0; i < n; i += nb)
    {
        panelQR(cusolver_handle, cublas_handle, m-i, nb, A+i*m+i, m,  W+i*m+i, m, R+i*n+i, n, work, info);
        
        if(n-i >nb)
        {
            startTimer();
            cublasDgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                nb,n - i - nb, m - i,
                &done,
                W+i*m+i, m,
                A+(i+nb)*m+i, m,
                &dzero,
                work, nb
            );

            cublasDgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m - i,n - i - nb, nb,
                &dnegone,
                A+i*m+i, m,
                work,nb,
                &done,
                A+(i+nb)*m+i,m
            );
            gemm_ms+=stopTimer();
            dim3 grid( (nb+1)/32, (n-i-nb+1)/32 );
            dim3 block( 32, 32 );
            copyAndClear<<<grid, block>>>(nb, n - i - nb, A+(i+nb)*m+i, m, R+(i+nb)*n+i, n); 
        }
        
        if(i!=0)
        {
            cublasDgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                i, nb, m,
                &done,
                A, m,
                W+i*m, m,
                &dzero,
                work, i
            );

            cublasDgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, nb, i,
                &dnegone,
                W, m,
                work,i,
                &done,
                W+i*m,m
            );
        }
    }

    ms = kernel_ms + y_ms + dtrsm_ms+gemm_ms;
    printf("kernel: %fms, construct_y: %fms, dtrsm_ms: %fms, gemm_ms: %fms\n",kernel_ms, y_ms, dtrsm_ms, gemm_ms);
    printf("tc_dgeqrf size %dx%d takes %lf ms, tflops is %lf\n",m, n, ms, 2.0*n*n*( m -1.0/3.0*n )/(ms*1e9));
    // printMatrixDeviceBlock("Y.csv",m,n,A,m);
    // printMatrixDeviceBlock("W.csv",m,n,W,m);
    // printMatrixDeviceBlock("R.csv",n,n,R,n);
    if(check)
    {
        //printMatrixDeviceBlock("RRRR.csv",n,n,R,n);
        dorgqr(m, n, W, m, A, m, work);
        //printMatrixDeviceBlock("Q.csv",m,n,W,m);
        //printMatrixDeviceBlock("RRR.csv",n,n,R,n);
        checkResult(m, n, oriA, m, W, m, R, n);
    }
    
}