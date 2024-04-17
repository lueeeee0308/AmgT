#compilers
CC=${CUDA_HOME}/bin/nvcc

# #ENVIRONMENT_PARAMETERS
# CUDA_INSTALL_PATH = ${CUDA_HOME}
NVCC_FLAGS=-O3 -ccbin gcc -m64 -gencode arch=${ARCHS}
#includes
# INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I/usr/local/cuda-12.0/samples/common/inc 

CUDA_LIBS = -lcusparse -lcublas
LIBS =  -lm -lineinfo $(CUDA_LIBS)

#options
OPTIONS = -Xcompiler -fopenmp

make:
	$(CC) $(NVCC_FLAGS) AmgT_convert.cu -o convert $(INCLUDES) $(OPTIONS) $(LIBS) 

clean:
	rm -rf spmv