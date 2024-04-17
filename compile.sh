#The config info
export CUDA_HOME=/usr/local/cuda-12.2
export GPU=A100
#end config

pwd_file=$(pwd)
chmod -R *
echo $pwd_file
Hypre_HOME=${pwd_file}/hypre_all/src
cd ${Hypre_HOME}

if [ "$GPU" = "A100" ]
then
    #A100
    echo "The GPU is A100"
    ./configure --with-cuda --with-gpu-arch='80 80' --enable-unified-memory
    export ARCHS=compute_80,code=sm_80
    export CUDA_ARCH=-gencode arch=compute_80,code=sm_80
else
    #H100
    echo "The GPU is H100"
    # ./configure --with-cuda --with-gpu-arch='90 90' --enable-unified-memory
    export ARCHS=compute_90,code=sm_90
    export CUDA_ARCH=-gencode arch=compute_90,code=sm_90
fi

make clean
export HYPRE_DIR=${Hypre_HOME}/hypre
for execuative in {CuSparse,Mixed,No_Mixed,CuSparse_PrintKernel,Mixed_PrintKernel,No_Mixed_PrintKernel}
do 
    cp ${pwd_file}/config_files/${execuative}.h ${Hypre_HOME}/seq_mv/seq_mv.h
    echo ${pwd_file}/config_files/${execuative}.h
    cd ${Hypre_HOME}
    make install -j
    cd ${pwd_file}/hypre_test
    make clean && make test_new
    mv test_new ${pwd_file}/hypre_test/runnable_files/${execuative}
done 

#preprocess compile
cd ${pwd_file}/preprocess_test
make clean && make
mv convert  ${pwd_file}/hypre_test/runnable_files/preprocess
cd $pwd_file

