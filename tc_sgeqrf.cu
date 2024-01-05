#include "TensorBLAS.h"

long int m, n, nb;
float ms = 0;
bool check = true;
float kernel_ms = 0;
float y_ms = 0;
float dtrsm_ms = 0;
float gemm_ms = 0;

// 确认输入参数，不涉及数据类型
int parseArguments(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Needs m, n and nb as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    return 0;
}

// 归并wrap内所有线程的值，值是float的
__inline__ __device__ float warpAllReduceSum(float val)
{
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// 生成单精度的随机矩阵
void generateUniformMatrix(float *dA, long int m, long int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, dA, long(m * n));
}

// 根据论文应该传入M=256行？
// 后续调用的传入: M*N=128*32, blockdim: 32*16
// 此处M和N变量也可以直接作为参数传入给函数，在本程序里效果一样,写成模板可以额外在编译时指定块大小
// 以线程为单位，M：线程块的行数；N：线程块的列数
// 一个线程块处理一个局部矩阵
// 输入时：AA中存放的是待进行QR分解的矩阵，是m&n大小的大矩阵
// 输出时：AA存放的是Q矩阵，RR中存放的是R矩阵
// m,n:总体矩阵的维度
template <long int M, long int N>
__global__ void hou_kernel3(long int m, long int n, float *AA, long int lda, float *RR, long int ldr)
{

    // 计算每个线程块的行数 mm：即当前线程在整个矩阵中需要处理的行数
    // 当前线程所在线程块*M
    long int mm = m - blockIdx.x * M; // 求需要处理的总行数（即去掉已经处理的前blockIdx.x*M行，当前所在的线程块x维度是blockIdx.x，每个线程块的行数为M）
    mm = (mm < M) ? mm : M;           // 实际上是为了计算最后多出的不能被M整除的行数，在此之前就是每个线程处理M行

    if (mm <= 0)
        return;

    const long int mnmin = (mm < n) ? mm : n; 

    float *A = &AA[blockIdx.x * M];   
    float *R = &RR[blockIdx.x * N];    
    __shared__ float As[M * N], Rs[N]; // 定义共享内存，As 用于存储线程块内的局部矩阵的一部分，而 Rs 用于存储一些中间计算的结果
    const long int ldas = M;           // 局部矩阵 As 的列数ldrs = N

    //    float acc0, acc1, acc2, acc3, acc4,acc5, acc6, acc7;
    float acc[8];
    const long int i = threadIdx.x, j = threadIdx.y; // i和j是当前线程在线程块内的索引号

    // #define R07(OP) {OP(0);OP(1);OP(2);OP(3);OP(4);OP(5);OP(6);OP(7);}
    // #define M1(it) if(threadIdx.x+it*32<mm) As[threadIdx.x+it*32+threadIdx.y*ldas] = A[threadIdx.x+it*32+threadIdx.y*lda]

    // #pragma unroll 2
    // 本循环用于将需要处理的分块后的小矩阵先存入共享内存中，降低后续读写成本
    // 此处是在核函数内，一个线程只处理16个元素，但是并行之后可以所有都处理掉
    // 一个线程块有128*32个线程，32*16*16=256*32个矩阵元素
    for (int k = 0; k < 8; k++)
    {
        // 32的来源：blockdim: 32*16，也就是i最大为31，j最大为15
        if (i + k * 32 < mm)
            // i:threadIdx.x，当前线程在块内的索引，第x行
            // k * 32：循环8次，当前是第k次循环
            // j * ldas：j是块内第j列，j * ldas代表按照列优先来编号，前j列（从0开始下标）的所有线程，每列有M行
            // ldas 和 lda 分别表示局部矩阵 As 和全局矩阵子矩阵 A （对应blockIdx.x，即当前线程块所在的那一行的所有线程块组成的子矩阵）在内存中的行跨度
            As[i + k * 32 + j * ldas] = A[i + k * 32 + j * lda]; // As中第i+k*32行第j列的元素
        if (i + k * 32 < mm)
            As[i + k * 32 + (j + 16) * ldas] = A[i + k * 32 + (j + 16) * lda]; // As中第i+k*32行第j+16列的元素
    }

    __syncthreads();

    // 使用循环计算Householder反射，并更新局部矩阵 As。
    // 使用 warpAllReduceSum 函数在warp内进行归约求和操作。
    for (int k = 0; k < mnmin; k++)
    {
        // reference: house_gen.m and house_qr from Cleve Moler blog.
        float nu = 0;
        if (threadIdx.y == k % 16)
        { // threadIdx.y is the warpId; each warp takes two columns

#pragma unroll
            for (int it = 0; it < 8; it++)
            {
    
                (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k) ? (acc[it] = As[threadIdx.x + it * 32 + k * ldas] *
                                                                                        As[threadIdx.x + it * 32 + k * ldas])
                                                                           : acc[it] = 0;
            }
            nu = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            float normxsqr = (warpAllReduceSum(nu)); 
            float normx = sqrt(normxsqr);            

            float scale = 1.0 / normx; // scale = 1/(v^T*v)

#pragma unroll
            // 对下三角的每个元素执行*scale，即*1/(v^T*v)
            for (int it = 0; it < 8; it++)
            {
                if (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k)
                    As[threadIdx.x + it * 32 + k * ldas] *= scale;
            }
            __syncwarp();
            // k代表列，当x等于k时意味着处理对角线上的元素
            if (threadIdx.x == k)
            {
                float u1 = As[k + k * ldas];
                As[k + k * ldas] += (u1 >= 0) ? 1 : -1;
                Rs[k] = (u1 >= 0) ? -normx : normx;
            }
            __syncwarp();
            // 更新scale为新对角线元素绝对值开根号的倒数
            scale = 1.0 / sqrt(abs(As[k + k * ldas]));
#pragma unroll
            for (int it = 0; it < 8; it++)
            {
                if (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k)
                    As[threadIdx.x + it * 32 + k * ldas] *= scale;
            }

            __syncwarp();
        }
        __syncthreads();

        if (threadIdx.y > k)
        {
            float uxl = 0;
#pragma unroll
            for (int it = 0; it < 8; it++)
            {
                (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k) ? acc[it] = As[threadIdx.x + it * 32 + threadIdx.y * ldas] * As[threadIdx.x + it * 32 + k * ldas] : acc[it] = 0;
            }
            uxl = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            float ux = warpAllReduceSum(uxl);

#pragma unroll
            for (int it = 0; it < 8; it++)
            {
                if (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k)
                    As[threadIdx.x + it * 32 + threadIdx.y * ldas] -= ux * As[threadIdx.x + it * 32 + k * ldas];
            }
        }
        if (16 + threadIdx.y > k)
        {
            float uxl = 0;
#pragma unroll
            for (int it = 0; it < 8; it++)
            {
                (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k) ? acc[it] = As[threadIdx.x + it * 32 + (16 + threadIdx.y) * ldas] * As[threadIdx.x + it * 32 + k * ldas] : acc[it] = 0;
            }
            uxl = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
            float ux = warpAllReduceSum(uxl);

#pragma unroll
            for (int it = 0; it < 8; it++)
            {
                if (threadIdx.x + it * 32 < mm && threadIdx.x + it * 32 >= k)
                    As[threadIdx.x + it * 32 + (16 + threadIdx.y) * ldas] -= ux * As[threadIdx.x + it * 32 + k * ldas];
            }
        }
    }

    __syncthreads();

    // write to R
#pragma unroll
    for (int it = 0; it < 2; it++)
    {
        int j = it * 16 + threadIdx.y;
        int i = threadIdx.x;
        if (i == j)
            // 存放对角线元素
            R[i + i * ldr] = Rs[i];
        // 存放上三角
        else if (i < j)
        {
            R[i + j * ldr] = As[i + j * ldas];
            As[i + j * ldas] = 0;
        }
        // n是传入参数，矩阵的总列数
        else if (i < n)
        {
            R[i + j * ldr] = 0;
        }
    }

    // compute explict Q from Householder reflectors
    double Q[8 * 2];
#pragma unroll
    // Q的对角线元素全部置1，其他为0
    for (int k = 0; k < 8; k++)
    {
        Q[k] = 0;
        Q[k + 8] = 0;
    }
    if (i == j)
        Q[0] = 1.0;
    if (i == j + 16)
        Q[8] = 1.0;

    for (int k = mnmin - 1; k >= 0; k--)
    {
        float acc[8];
        if (threadIdx.y >= k)
        {
            float accum = 0;
            for (int l = 0; l < 8; l++)
                accum += As[i + l * 32 + k * ldas] * Q[l];
            float vq = warpAllReduceSum(accum);

            for (int l = 0; l < 8; l++)
                if (i + 32 * l < mm)
                    Q[l] -= vq * (As[i + 32 * l + k * ldas]);
        }
        if (threadIdx.y + 16 >= k)
        {
            float accum = 0;
            for (int l = 0; l < 8; l++)
                accum += As[i + l * 32 + k * ldas] * Q[l + 8];

            float vq = warpAllReduceSum(accum);
            for (int l = 0; l < 8; l++)
                if (i + 32 * l < mm)
                    Q[l + 8] -= vq * (As[i + 32 * l + k * ldas]);
        }
    }

// 将最终得到的Q矩阵存回A矩阵
#pragma unroll
    for (int k = 0; k < 8; k++)
    {
        if (i + k * 32 < mm)
            A[i + k * 32 + j * lda] = Q[k];
        if (i + k * 32 < mm)
            A[i + k * 32 + (j + 16) * lda] = Q[k + 8];
    }
}

// 由panelQR传入M=128,N=32
template <long int M, long int N>
void hou_caqr_panel(cublasHandle_t handle, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work)
{
    // 声明线程块布局，x轴32个线程，y轴16个线程
    dim3 blockdim(32, 16);
    if (m <= M)
    {
        // 传给hou_kernel3也是M=128，N=32
        // 启用1个grid，blockdim个线程块
        hou_kernel3<M, N><<<1, blockdim>>>(m, n, A, lda, R, ldr);
        return;
    }
    if ((m - m / M * M) % N != 0)
    {
        printf("Error: m must be i*%d + j*%d\n", M, N);
    }
    long int NB = (m + M - 1) / M;
    long int ldwork = NB * N;
    long int mm = NB * N;
    hou_kernel3<M, N><<<NB, blockdim>>>(m, n, A, lda, work, ldwork);

    hou_caqr_panel<M, N>(handle, mm, n, work, ldwork, R, ldr, work + ldwork * n);
    float done = 1.0, dzero = 0.0;
    auto status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            M, N, N,
                                            &done, A, lda, M,
                                            work, ldwork, N,
                                            &dzero, A, lda, M,
                                            m / M);

    mm = m % M;
    if (mm > 0)
    {
        auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  mm, N, N, &done, &A[m / M * M], lda, &work[m / M * N], ldwork,
                                  &dzero, &A[m / M * M], lda);
    }
}

// I - A的结果存进w工作区，划成一个独立的处理矩阵
__global__ void minusEye(long int m, long int n, float *a, long int lda, float *w, long ldw)
{
    long int i = threadIdx.x + blockDim.x * blockIdx.x;
    long int j = threadIdx.y + blockDim.y * blockIdx.y;
    // 条件 means 在矩阵合法索引范围内
    if (i < m && j < n)
    {
        // 求I - A，对角线上元素用1减，非对角线用0减
        if (i == j)
        {
            a[i + j * lda] = 1.0 - a[i + j * lda];
            w[i + j * ldw] = a[i + j * lda];
        }
        else
        {
            a[i + j * lda] = 0.0 - a[i + j * lda];
            w[i + j * ldw] = a[i + j * lda];
        }
    }
}

// 重构生成Y
// reconstructY的参数*U起到了什么作用？
// 输入dA是原矩阵，L和U都存放在dA指向的原地址空间内，其中L的对角线元素是1，dA中的对角线元素是U的对角线元素
void reconstructY(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, long m, long n, float *dA, long lda, float *U, float *work, int *info)
{
    // 用于计算单精度矩阵的LU分解
    // 假设P*A=L*U，P是置换矩阵，A是被处理矩阵
    // 此处P为NULL，即不执行轴转，直接A=L*U，输入A，work（工作空间），得到L和U
    // info用于指示是否出错，以及第几个参数出错
    cusolverDnSgetrf(cusolver_handle, m, n, dA, lda,
                     work, NULL, info);
}

// get U from LU factorization
// U是上三角矩阵，直接取a中的上半
__global__ void getU(int m, int n, float *a, int lda, float *u, int ldu)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n)
    {
        if (i > j)
            u[i + j * ldu] = 0;
        else
            u[i + j * ldu] = a[i + j * lda];
    }
}

// get L from LU factorization
// 把a的上半全部置0，对角线置1，下半就是householder向量
__global__ void getL(int m, int n, float *a, int lda)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n)
    {
        if (i < j)
            a[i + j * lda] = 0;
        else if (i == j)
            a[i + j * lda] = 1;
    }
}

// 在GPU上进行矩阵数组的拷贝，从dA拷贝到dB
__global__ void deviceCopy(long m, long n, float *dB, long ldb, float *dA, long lda)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n)
    {
        dB[i + j * ldb] = dA[i + j * lda];
    }
}

// 把矩阵I变成全0
__global__ void setZero(long m, long n, float *I, long ldi)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n)
    {
        I[i + j * ldi] = 0.0;
    }
}

void panelQR(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, long m, long n, float *A, long lda, float *W, long ldw, float *R, long ldr, float *work, int *info)
{
    // 切到最小单位（宽度不超过32）
    if (n <= 32)
    {
        // 中间输出方便纠错
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return;
        }

        startTimer();
        hou_caqr_panel<128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);
        kernel_ms += stopTimer();

        // 设置第一个网格的布局，每个线程块为32*32，网格有多少个块取决于矩阵的大小
        dim3 gridDim((m + 31) / 32, (n + 31) / 32);
        dim3 blockDim(32, 32);

        // W = I - *A = I - Q
        minusEye<<<gridDim, blockDim>>>(m, n, A, lda, W, ldw);
        // deviceCopy<<<gridDim,blockDim>>>( m, n, A, lda, W, ldw );

        // y_ms指在最底层重构Y，即householder vectors的时间
        // 每个时间用+=是因为会执行多次，最后得到总的
        startTimer();
        // means 对I-Q进行LU分解
        reconstructY(cusolver_handle, cublas_handle, m, n, A, lda, work, work + n * n, info);
        y_ms += stopTimer();
        getL<<<gridDim, blockDim>>>(n, n, A, lda);

        float done = 1.0;
        startTimer();
        // 调用cublas库的函数，求解right-rand三角形线性方程组？
        // 此函数可以调用64位int型，means用到了TensorCore？
        // 此时的A是分解后得到的 ，W是 ?
        cublasStrsm(cublas_handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                    m, n,
                    &done,
                    A, lda,
                    W, ldw);
        dtrsm_ms += stopTimer();

        return;
    }

    // 递归是把一个宽矩阵切成一个高瘦矩阵，瘦定义为列数不超过32
    panelQR(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);

    float done = 1.0, dzero = 0.0, dnegone = -1.0;
    // 计算拖尾矩阵更新的矩阵乘法时间？
    startTimer();
    cublasSgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n / 2, n / 2, m,
                &done,
                W, ldw,
                A + lda / 2 * n, lda,
                &dzero,
                work, n / 2);
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n / 2, n / 2,
                &dnegone,
                A, lda,
                work, n / 2,
                &done,
                A + lda / 2 * n, lda);
    gemm_ms += stopTimer();

    dim3 gridDim1((n / 2 + 31) / 32, (n / 2 + 31) / 32);
    dim3 blockDim1(32, 32);
    deviceCopy<<<gridDim1, blockDim1>>>(n / 2, n / 2, R + ldr / 2 * n, ldr, A + lda / 2 * n, lda);
    setZero<<<gridDim1, blockDim1>>>(n / 2, n / 2, A + lda / 2 * n, lda);
    panelQR(cusolver_handle, cublas_handle, m - n / 2, n / 2, A + lda / 2 * n + n / 2, lda, W + ldw / 2 * n + n / 2, ldw, R + n / 2 * ldr + n / 2, ldr, work, info);
    // printf("hrere\n");
    // printMatrixDeviceBlock("R--.csv",n/2,n/2,R+n/2*ldr+n/2,n);

    startTimer();
    cublasSgemm(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n / 2, n / 2, m,
                &done,
                A, lda,
                W + ldw / 2 * n, ldw,
                &dzero,
                work, n / 2);
    cublasSgemm(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n / 2, n / 2,
                &dnegone,
                W, ldw,
                work, n / 2,
                &done,
                W + ldw / 2 * n, ldw);
    gemm_ms += stopTimer();
    return;
}

// 把da中的元素拷贝到db中进行存储，再将da全部清空置零
__global__ void copyAndClear(long int m, long int n, float *da, int lda, float *db, int ldb)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n)
    {
        db[i + j * ldb] = da[i + j * lda];
        da[i + j * lda] = 0.0;
    }
}

// 调用cublasSnrm2函数计算一个单精度矩阵的二范数，即矩阵中所有元素平方和开根号的值
float snorm(long int m, long int n, float *dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sn;
    int incx = 1;
    cublasSnrm2(handle, m * n, dA, incx, &sn);
    cublasDestroy(handle);
    return sn;
}

// 分别求原本的A矩阵的二范数，和分解得到的QR相乘得到的A'的二范数，并进行误差对比
void checkResult(long int m, long int n, float *A, long int lda, float *Q, long int ldq, float *R, int ldr)
{
    float normA = snorm(m, n, A);
    float alpha = 1.0;
    float beta = -1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // printMatrixDeviceBlock("oriA.csv",m,n,A,lda);
    // printMatrixDeviceBlock("QQ.csv",m,n,Q,ldq);
    // printMatrixDeviceBlock("RR.csv",n,n,R,ldr);
    startTimer();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, n, &alpha, Q, ldq, R, ldr,
                &beta, A, lda);

    float ms = stopTimer();
    printf("SGEMM m*n*k %d*%d*%d takes %.0f (ms), exec rate %.0f TFLOPS\n",
           m, n, n, ms, 2.0 * m * n * n / (ms * 1e9));
    // printMatrixDeviceBlock("res.csv",m,n,A,lda);
    double normRes = snorm(m, n, A);
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n", normRes / normA);
}

void dorgqr(int m, int n, float *W, int ldw, float *Y, int ldy, float *work)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float sone = 1.0;
    float snegone = -1.0;

    // 线程布局，grid1是线程网格的名字，由处理矩阵的大小横竖来决定，使得一个块固定处理<=32大小的矩阵
    dim3 grid1((m + 1) / 32, (n + 1) / 32);
    // 结合grid，一个线程块就是32*32个线程，每个线程处理一个元素
    dim3 block1(32, 32);
    setEye<<<grid1, block1>>>(m, n, work, m);
    // cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,n,n,
    //     &snegone,W,CUDA_R_32F, ldw, Y, CUDA_R_32F, ldy,
    //     &sone, work, CUDA_R_32F, m, CUDA_R_32F,
    //     CUBLAS_GEMM_DEFAULT
    // );
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, n, &snegone, W, ldw, Y, ldy,
                &sone, work, m);
    cublasDestroy(handle);
    deviceCopy<<<grid1, block1>>>(m, n, W, ldw, work, m);
}

int main(int argc, char *argv[])
{
    if (parseArguments(argc, argv) == -1)
        return 0;

    float *A;
    cudaMalloc(&A, sizeof(float) * m * n);
    float *work;
    cudaMalloc(&work, sizeof(float) * m * n);
    float *R;
    cudaMalloc(&R, sizeof(float) * n * n);
    float *W;
    cudaMalloc(&W, sizeof(float) * m * n);
    float *oriA;

    int *info;
    cudaMalloc(&info, sizeof(int));

    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
    cusolverDnCreate(&cusolver_handle);
    cublasCreate(&cublas_handle);

    generateUniformMatrix(A, m, n);

    if (check)
    {
        cudaMalloc(&oriA, sizeof(float) * m * n);
        cudaMemcpy(oriA, A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice);
    }

    float done = 1.0, dzero = 0.0, dnegone = -1.0;

    for (int i = 0; i < n; i += nb) // nb是最小单位矩阵列数
    {

        panelQR(cusolver_handle, cublas_handle, m - i, nb, A + i * m + i, m, W + i * m + i, m, R + i * n + i, n, work, info);

        if (n - i > nb)
        {
            startTimer();
            cublasSgemm(cublas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        nb, n - i - nb, m - i,
                        &done,
                        W + i * m + i, m,
                        A + (i + nb) * m + i, m,
                        &dzero,
                        work, nb);

            cublasSgemm(cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        m - i, n - i - nb, nb,
                        &dnegone,
                        A + i * m + i, m,
                        work, nb,
                        &done,
                        A + (i + nb) * m + i, m);
            gemm_ms += stopTimer();
            dim3 grid((nb + 1) / 32, (n - i - nb + 1) / 32);
            dim3 block(32, 32);
            copyAndClear<<<grid, block>>>(nb, n - i - nb, A + (i + nb) * m + i, m, R + (i + nb) * n + i, n);
        }

        if (i != 0)
        {
            cublasSgemm(cublas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        i, nb, m,
                        &done,
                        A, m,
                        W + i * m, m,
                        &dzero,
                        work, i);

            cublasSgemm(cublas_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        m, nb, i,
                        &dnegone,
                        W, m,
                        work, i,
                        &done,
                        W + i * m, m);
        }
    }

    ms = kernel_ms + y_ms + dtrsm_ms + gemm_ms;
    printf("kernel: %fms, construct_y: %fms, dtrsm_ms: %fms, gemm_ms: %fms\n", kernel_ms, y_ms, dtrsm_ms, gemm_ms);
    printf("tc_dgeqrf size %dx%d takes %lf ms, tflops is %lf\n", m, n, ms, 2.0 * n * n * (m - 1.0 / 3.0 * n) / (ms * 1e9));

    if (check)
    {
        dorgqr(m, n, W, m, A, m, work);
        checkResult(m, n, oriA, m, W, m, R, n);
    }
}
