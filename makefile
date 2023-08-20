CXX=nvcc
CFLAGS=-O2 -arch sm_80 -I $(CUDA_PATH)/include -I /home/szhang94/TensorBLAS/Tensor-BLAS/include -I /home/szhang94/TensorBLAS/Tensor-BLAS/cuMpSGEMM/include
LFLAGS=-L $(CUDA_PATH)/lib64  -lcusolver -lcublas  -lcurand -lcudart -lcuda -lTensorBLAS -lcumpsgemm
LFLAGS+=-L /home/szhang94/TensorBLAS/Tensor-BLAS/build -L /home/szhang94/TensorBLAS/Tensor-BLAS/cuMpSGEMM/build
CC = gcc

all: cbench dgeqrf tc_dgeqrf

cbench: cusolverBenchmark.cu
	nvcc  $(CFLAGS) -c cusolverBenchmark.cu
	nvcc  $(LFLAGS) cusolverBenchmark.o -o cuSOLVERgeqrf

dgeqrf: dgeqrf.cu
	nvcc  $(CFLAGS) -c dgeqrf.cu
	nvcc  $(LFLAGS) dgeqrf.o -o dgeqrf

dgeqrf: tc_dgeqrf.cu
	nvcc  $(CFLAGS) -c tc_dgeqrf.cu
	nvcc  $(LFLAGS) tc_dgeqrf.o -o tc_dgeqrf

clean:
	rm -f *.o cuSOLVERgeqrf dgeqrf tc_dgeqrf