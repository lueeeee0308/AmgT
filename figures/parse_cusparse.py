import sys

if len(sys.argv) < 2:
    print("Usage: python script.py argument")
    sys.exit(1)

filename = sys.argv[1]

def printMatrix(tag,end_tag,word_):
    if(word_[0]==tag):
        print()
        print(word_[len(word_)-1],end=end_tag)
with open(filename, 'r') as f:
    file_lines = f.readlines()
    words = [file_line.strip().split('=') for file_line in file_lines]
    t_all_V_cycle=0 #AMG solve时间
    t_smooth=0 
    t_residual=0
    t_interpolation=0 # smooth+residual+interpolation是solve阶段的spmv时间
    solve_time=0    #pcg_solve时间
    amg_Solve_time=0
    amg_SpMV_time=0
    time_pcg_spmv=0
    time_spmv_kernel=0
    pcg_time=0
    pcg_SpMV_time=0
    SpMV_preprocess_time=0
    SpMV_function_time=0
    SpMV_times=0
    setup_time=0
    spgemm_time=0
    spgemm_kernel_time=0
    spgemm_preprocess_time=0
    spgemm_times=0
    Iterations=0
    residual=0
    nrows=0
    nnz=0
    for word_ in words:
        # print(word_)
        printMatrix("matrix",",",word_)
        if(word_[0]=="rows"):
            nrows=int(word_[len(word_)-1])
        if(word_[0]=="nonzeros"):
            nnz=int(word_[len(word_)-1])
        if(word_[0]=="Iterations "):
            Iterations=int(word_[len(word_)-1])
        if(word_[0]=="Final Relative Residual Norm "):
            residual=str(word_[len(word_)-1])
        if(word_[0]=="setup_time"):
            setup_time=float(word_[len(word_)-1])
        if(word_[0]=="solve_time"):
            solve_time=float(word_[len(word_)-1]) 
        if(word_[0]=="time_spmv_sum"):
            time_spmv_kernel=float(word_[len(word_)-1])  
        if(word_[0]=="time_spmv"):
            SpMV_function_time=float(word_[len(word_)-1])  
            # SpMV_preprocess_time=SpMV_function_time-time_spmv_kernel
        if(word_[0]=="time_spmv_preprocess"):
            SpMV_preprocess_time=float(word_[len(word_)-1])
        if(word_[0]=="spmv_times"):
            SpMV_times=int(word_[len(word_)-1])
        if(word_[0]=="time_spgemm"):
            spgemm_time=float(word_[len(word_)-1])
        if(word_[0]=="cusparse_spgemm_time"): #cusparse的纯计算时间
            spgemm_kernel_time=float(word_[len(word_)-1])
            spgemm_preprocess_time=spgemm_time-spgemm_kernel_time #cusparse的矩阵乘时间
        if(word_[0]=="spgemm_times"):
            spgemm_times=int(word_[len(word_)-1])
            print(str(nrows)+","
            +str(nnz)+","
            +str(spgemm_times)+","
            +str(SpMV_times)+","
            +str(Iterations)+","
            +str(residual)+","
            +str(setup_time)+","
            +str(spgemm_time)+","
            +str(spgemm_kernel_time)+","
            +str(spgemm_preprocess_time)+","
            +str(solve_time)+","
            +str(SpMV_function_time)+","
            +str(time_spmv_kernel)+","
            +str(SpMV_preprocess_time))
            # print(str(residual))
        

# def printMatrix(tag,end_tag,word_):
#     if(word_[0]==tag):
#         print()
#         print(word_[len(word_)-1],end=end_tag)
# with open('/home/weifeng/wtc/AMG/AMG/run_tests/data_no_print/all_spmv_AMGT_new_0331.out', 'r') as f:
#     file_lines = f.readlines()
#     words = [file_line.strip().split('=') for file_line in file_lines]
#     t_all_V_cycle=0 #AMG solve时间
#     t_smooth=0 
#     t_residual=0
#     t_interpolation=0 # smooth+residual+interpolation是solve阶段的spmv时间
#     solve_time=0    #pcg_solve时间
#     amg_Solve_time=0
#     amg_SpMV_time=0
#     time_pcg_spmv=0
#     time_spmv_kernel=0
#     pcg_time=0
#     pcg_SpMV_time=0
#     SpMV_preprocess_time=0
#     SpMV_function_time=0
#     SpMV_times=0
#     setup_time=0
#     spgemm_time=0
#     spgemm_kernel_time=0
#     spgemm_preprocess_time=0
#     spgemm_times=0
#     Iterations=0
#     residual=0
#     for word_ in words:
#         # print(word_)
#         printMatrix("matrix",",",word_)
#         if(word_[0]=="Iterations "):
#             Iterations=int(word_[len(word_)-1])
#         if(word_[0]=="Final Relative Residual Norm "):
#             residual=str(word_[len(word_)-1])
#         if(word_[0]=="setup_time"):
#             setup_time=float(word_[len(word_)-1])
#         if(word_[0]=="solve_time"):
#             solve_time=float(word_[len(word_)-1]) 
#         if(word_[0]=="time_spmv_sum"):
#             time_spmv_kernel=float(word_[len(word_)-1])  
#         if(word_[0]=="time_spmv"):
#             SpMV_function_time=float(word_[len(word_)-1])  
#         #     SpMV_preprocess_time=SpMV_function_time-time_spmv_kernel
#         if(word_[0]=="time_spmv_preprocess"):
#             SpMV_preprocess_time=float(word_[len(word_)-1])
#             # SpMV_function_time=time_spmv_kernel+SpMV_preprocess_time
#         if(word_[0]=="spmv_times"):
#             SpMV_times=int(word_[len(word_)-1])
#         if(word_[0]=="time_spgemm"):
#             spgemm_kernel_time=float(word_[len(word_)-1])
#         if(word_[0]=="time_spgemm_preprocess"): #cusparse的纯计算时间
#             spgemm_preprocess_time=float(word_[len(word_)-1])
#             spgemm_time=spgemm_preprocess_time+spgemm_kernel_time
#             # spgemm_preprocess_time=spgemm_time-spgemm_kernel_time #cusparse的矩阵乘时间
#         if(word_[0]=="spgemm_times"):
#             spgemm_times=int(word_[len(word_)-1])
#             print(str(setup_time)+","
#             +str(spgemm_time)+","
#             +str(spgemm_kernel_time)+","
#             +str(spgemm_preprocess_time)+","
#             +str(solve_time)+","
#             +str(SpMV_function_time)+","
#             +str(time_spmv_kernel)+","
#             +str(SpMV_preprocess_time))
#             # str(spgemm_times)+","
            # +str(SpMV_times)+","
            # +str(Iterations)+","
            # +str(residual)+","
            # +