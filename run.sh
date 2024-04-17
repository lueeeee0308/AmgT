pwd_file=$(pwd)
export RUNNING_FILES=${pwd_file}/hypre_test/runnable_files
MM_path=${pwd_file}/matrix
#all time test

input=${pwd_file}"/matrix.csv"
{
read
i=1
while IFS=',' read -r mid Group Name rows cols nonzeros
do
    for execuative in {No_Mixed,Mixed,CuSparse}
    do
        RUNING_FILE=${pwd_file}/hypre_test/runnable_files/${execuative}
        EXPORT_FILE=${pwd_file}/hypre_test/data/${execuative}.out
        echo ${RUNING_FILE}
        echo matrix=$Name >>${EXPORT_FILE}
        echo rows=$rows >>${EXPORT_FILE}
        echo nonzeros=$nonzeros >>${EXPORT_FILE}
        echo $MM_path/$Name.mtx
        ${RUNING_FILE} $MM_path/$Name.mtx >>${EXPORT_FILE}
    done 
    i=`expr $i + 1`
done 
} < "$input"

#kernel test
input=${pwd_file}"/matrix.csv"
{
read
i=1
while IFS=',' read -r mid Group Name rows cols nonzeros
do
    for execuative in {CuSparse_PrintKernel,Mixed_PrintKernel,No_Mixed_PrintKernel}
    do
        RUNING_FILE=${pwd_file}/hypre_test/runnable_files/${execuative}
        EXPORT_FILE=${pwd_file}/hypre_test/data/${execuative}.out
        echo ${RUNING_FILE}
        echo matrix=$Name >>${EXPORT_FILE} 
        echo rows=$rows >>${EXPORT_FILE}
        echo nonzeros=$nonzeros >>${EXPORT_FILE}
        ${RUNING_FILE} $MM_path/$Name.mtx >>${EXPORT_FILE}
    done 
    i=`expr $i + 1`
done 
} < "$input"

#preprocess test

echo "preprocess test"

cd ${pwd_file}/preprocess_test

#!/bin/bash
input=${pwd_file}"/matrix.csv"
{
  read
  i=1
  while IFS=',' read -r mid Group Name rows cols nonzeros
  do
    echo "==="
    echo "matrix $mid $Group $Name $rows $cols $nonzeros"
    RUNING_FILE=${pwd_file}/hypre_test/runnable_files/preprocess
    echo matrix $Name >>${pwd_file}/hypre_test/data/preprocess_data.out 
    ${RUNING_FILE}  $MM_path/$Name.mtx 1 >> ${pwd_file}/hypre_test/data/preprocess_data.out
    i=`expr $i + 1`
  done 
} < "$input"