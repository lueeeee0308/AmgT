# AmgT
SC24-AmgT

Firstly, download the 16 matrices from SuitSparse.

# Download the matrix
Run the command `bash matrix.py` and the 16 matrices are automatically downloaded to the folder “./matrix” .


When you want to run this files, you should change the `#The config info`  of `compile.sh`.

- Step 1: Change the `CUDA_HOME`
- Step 2: Change the `GPU` to `A100` or `H100`.

# compile

`source compile.sh`

When execuate `source compile.sh`, it will generate 7 running files in `./hypre_test/runnable_files`, which including:

- CuSparse: Test case using the CuSparse kernel.
- CuSparse_PrintKernel: Test case using the CuSparse kernel and will print the kernel performance at each time.
- Mixed: Test case using the mixed-precision AMGT kernel.
- Mixed_PrintKernel: Test case using the mixed-precision AMGT kernel and will print the kernel performance at each time.
- No_Mixed: Test case using the double-precision AMGT kernel.
- No_Mixed_PrintKernel: Test case using the double-precision AMGT kernel and will print the kernel performance at each time.

# run

`nohup bash run.sh 2>&1`

When execuate `nohup bash run.sh 2>&1`, the output_files will be stored in `./hypre_test/data`.


If you want to kill the  `run.sh`.

`ps -ef|grep run.sh`

Then `kill -9 process_id`. 

# figures

When you get the output files.

`cd ./figures`.

Then `bash figures.sh`, you will get `Fig7.pdf`, `Fig8.pdf` and `Fig9.pdf`.


