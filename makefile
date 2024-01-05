CXX=nvcc
CFLAGS=-O2 -arch sm_86 -I $(CUDA_PATH)/include -I /home/lyh/TPDS-s/Tensor-BLAS/include -I /home/lyh/TPDS-s/Tensor-BLAS/cuMpSGEMM/include
LFLAGS=-L $(CUDA_PATH)/lib64  -lcusolver -lcublas  -lcurand -lcudart -lcuda -lTensorBLAS -lcumpsgemm
LFLAGS+=-L /home/lyh/TPDS-s/Tensor-BLAS/build -L /home/lyh/TPDS-s/Tensor-BLAS/cuMpSGEMM/build
CC = gcc

#all: cbench tc_dgeqrf dgeqrf
all: cbench tc_dgeqrf tc_sgeqrf 

cbench: cusolverBenchmark.cu
	nvcc  $(CFLAGS) -c cusolverBenchmark.cu
	nvcc  $(LFLAGS) cusolverBenchmark.o -o cuSOLVERgeqrf

tc_dgeqrf: tc_dgeqrf.cu
	nvcc  $(CFLAGS) -c tc_dgeqrf.cu
	nvcc  $(LFLAGS) tc_dgeqrf.o -o tc_dgeqrf

tc_sgeqrf: tc_sgeqrf.cu
	nvcc  $(CFLAGS) -c tc_sgeqrf.cu
	nvcc  $(LFLAGS) tc_sgeqrf.o -o tc_sgeqrf



#dgeqrf: dgeqrf.cu
#	nvcc  $(CFLAGS) -c dgeqrf.cu
#	nvcc  $(LFLAGS) dgeqrf.o -o dgeqrf

clean:
#	rm -f *.o cuSOLVERgeqrf tc_dgeqrf dgeqrf
	rm -f *.o cuSOLVERgeqrf tc_dgeqrf tc_sgeqrf