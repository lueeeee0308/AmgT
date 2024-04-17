import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, sqrt
import math
import numpy as np
import csv


def parse_files(file1, file2, file3, output_file):
    columns = {
        'matrix':[],
        'number':[],
        'cusparse': [],
        'amgt': [],
        'amgt_mixed': []
    }

    with open(file1, 'r') as file:
        file_lines = file.readlines()
        words = [file_line.strip().split('=') for file_line in file_lines]
        matrix_name=''
        spgemm_cnt=1
        spmv_cnt=1
        for word_ in words:
            if(word_[0]=="matrix"):
                matrix_name=word_[len(word_)-1]
                spgemm_cnt=1
                spmv_cnt=1
            if(word_[0]=="spgemm_kernel_time"):
                columns['matrix'].append("SpGEMM+"+matrix_name)
                columns['number'].append(spgemm_cnt)
                columns['cusparse'].append(word_[len(word_)-1])
                spgemm_cnt=spgemm_cnt+1
            if(word_[0]=="spmv_kernel_time"):
                columns['matrix'].append("SpMV+"+matrix_name)
                columns['number'].append(spmv_cnt)
                columns['cusparse'].append(word_[len(word_)-1])
                spmv_cnt=spmv_cnt+1

    with open(file2, 'r') as file:
        file_lines = file.readlines()
        words = [file_line.strip().split('=') for file_line in file_lines]
        matrix_name=''
        spgemm_cnt=1
        spmv_cnt=1
        for word_ in words:
            if(word_[0]=="spgemm_kernel_time"):
                columns['amgt'].append(word_[len(word_)-1])
            if(word_[0]=="spmv_kernel_time"):
                columns['amgt'].append(word_[len(word_)-1])

    with open(file3, 'r') as file:
        file_lines = file.readlines()
        words = [file_line.strip().split('=') for file_line in file_lines]
        matrix_name=''
        spgemm_cnt=1
        spmv_cnt=1
        for word_ in words:
            if(word_[0]=="spgemm_kernel_time"):
                columns['amgt_mixed'].append(word_[len(word_)-1])
            if(word_[0]=="spmv_kernel_time"):
                columns['amgt_mixed'].append(word_[len(word_)-1])

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # 写入标题行
        writer.writerow(['matrix','number','cusparse', 'amgt', 'amgt_mixed'])

        # 获取最大列数据长度
        max_length = max(len(column) for column in columns.values())

        # 逐行写入数据
        for row_index in range(max_length):
            row = [columns[column_name][row_index] if row_index < len(columns[column_name]) else '' for column_name in columns.keys()]
            writer.writerow(row)

parse_files(
            '../hypre_test/data/CuSparse_PrintKernel.out', 
            '../hypre_test/data/No_Mixed_PrintKernel.out',
            '../hypre_test/data/Mixed_PrintKernel.out',
            'Fig8.csv')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_csv('Fig8.csv')

matrix_order = df['matrix'].unique()

num_groups = len(matrix_order)
num_cols = 8
num_rows = ceil(num_groups / num_cols)

fig = plt.figure(figsize=(30, 14))

title_fontsize = 18
legend_fontsize = 10
label_fontsize = 17
line_color = ['#46958f', '#ee695b', '#f6b753']
line_style = ['solid', 'dashed']
marker_style = ['o', 'o', 'o']
cnt = 0
line_width = 2

dot_size=1

for i, matrix_name in enumerate(matrix_order):
    if(i%2==0):
        group = df[df['matrix'] == matrix_name]
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.grid(True)
        cusparse_val=list(map(float, group['cusparse'].values.tolist()))
        cusparse_val=np.log10(cusparse_val)
        amgt_val=list(map(float, group['amgt'].values.tolist()))
        amgt_val=np.log10(amgt_val)
        amgt_mixed_val=list(map(float, group['amgt_mixed'].values.tolist()))
        amgt_mixed_val=np.log10(amgt_mixed_val)

        scatter1=ax.scatter(group['number'], amgt_mixed_val, label="AmgT (Mixed)",  color=line_color[2],
                marker=marker_style[1], linewidth=line_width)
        scatter2=ax.scatter(group['number'], amgt_val, label="AmgT (FP64)",  color=line_color[1],
                marker=marker_style[1], linewidth=line_width)
        scatter3=ax.scatter(group['number'], cusparse_val, label="Hypre (FP64)",  color=line_color[0],
                marker=marker_style[0], linewidth=line_width)
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        if (cnt % num_cols == 0):
            ax.set_ylabel('Time (ms)\n($\\log_{10}$ scale)', fontsize=title_fontsize + 5)
        # ax.legend(loc='upper right', fontsize=legend_fontsize)


        # new_order = [2,1, 0]
        # handles = [handles[i] for i in new_order]
        # labels = [labels[i] for i in new_order]
        # plt.legend(handles, labels,loc='upper right', fontsize=legend_fontsize)
        list_label=matrix_name.split('+')
        label_name="SpGEMM"+"\n"+list_label[len(list_label)-1]
        ax.set_xlabel(label_name, fontsize=title_fontsize)
        cnt += 1
    else:
        group = df[df['matrix'] == matrix_name]
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.grid(True)

        cusparse_val=list(map(float, group['cusparse'].values.tolist()))
        cusparse_val=np.log10(cusparse_val)
        amgt_val=list(map(float, group['amgt'].values.tolist()))
        amgt_val=np.log10(amgt_val)
        amgt_mixed_val=list(map(float, group['amgt_mixed'].values.tolist()))
        amgt_mixed_val=np.log10(amgt_mixed_val)

        ax.scatter(group['number'], amgt_mixed_val, label="AmgT (Mixed)", s=dot_size,  color=line_color[2],
                marker=marker_style[1], linewidth=line_width)
        ax.scatter(group['number'], amgt_val, label="AmgT (FP64)",  s=dot_size, color=line_color[1],
                marker=marker_style[1], linewidth=line_width)
        ax.scatter(group['number'], cusparse_val, label="Hypre (FP64)", s=dot_size, color=line_color[0],
                marker=marker_style[0], linewidth=line_width)


        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        if (cnt % num_cols == 0):
            ax.set_ylabel('Time (ms)\n($\\log_{10}$ scale)', fontsize=title_fontsize + 5)
        # ax.legend(loc='lower left', fontsize=legend_fontsize)
        list_label=matrix_name.split('+')
        label_name="SpMV"+"\n"+list_label[len(list_label)-1]

        ax.set_xlabel(label_name, fontsize=title_fontsize)
        # ax.set_yscale('log')
        cnt += 1
        
handles, labels = plt.gca().get_legend_handles_labels()
new_order = [2,1, 0]
handles = [handles[i] for i in new_order]
labels = [labels[i] for i in new_order]
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=title_fontsize + 5, bbox_to_anchor=(0.5, 1.05),markerscale=15)
# plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以留出空间给图例
# if num_groups < num_rows * num_cols:
#     for i in range(num_groups, num_rows * num_cols):
#         fig.delaxes(fig.axes[i])

plt.tight_layout()
plt.savefig("Fig8.pdf", bbox_inches='tight', dpi=300)
# plt.show()