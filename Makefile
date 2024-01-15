CC = gcc
NVCC = nvcc
CUDA_PATH = /usr/local/cuda
CFLAGS_GPU = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lcurand -O3 -g -D_FORCE_INLINES -Xcompiler -I/C/common/inc
CFLAGS_GPU2 = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lcurand -O3 -g -D_FORCE_INLINES -Xcompiler -I/C/common/inc -use_fast_math

all: ant_gpu.o tsp-ant-gpu mravi.o mravi mravi2.o mravi2

tsp-ant-gpu: ant_gpu.o
	$(NVCC) $(CFLAGS_GPU) ant_gpu.o -o tsp-ant-gpu

ant_gpu.o: parallel_ants.cu
	$(NVCC) -c parallel_ants.cu $(CFLAGS_GPU) -o ant_gpu.o

mravi: mravi.o
	$(NVCC) $(CFLAGS_GPU2) mravi.o -o mravi

mravi.o: parallel_ants_v2.cu
	$(NVCC) -c parallel_ants_v2.cu $(CFLAGS_GPU2) -o mravi.o

mravi2: mravi2.o
	$(NVCC) $(CFLAGS_GPU2) mravi2.o -o mravi2

mravi2.o: parallel_ants_v3.cu
	$(NVCC) -c parallel_ants_v3.cu $(CFLAGS_GPU2) -o mravi2.o