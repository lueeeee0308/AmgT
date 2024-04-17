import math
import numpy as np
import matplotlib.pyplot as plt
# This restores the same behavior as before.
import pandas as pd
from matplotlib.hatch import Shapes, _hatch_types
from matplotlib.patches import Rectangle

class SquareHatch(Shapes):
    """
    Square hatch defined by a path drawn inside [-0.5, 0.5] square.
    Identifier 's'.
    """
    def __init__(self, hatch, density):
        self.filled = False
        self.size = 1
        self.path = Rectangle((-0.25, 0.25), 0.5, 0.5).get_path()
        self.num_rows = (hatch.count('s')) * density
        self.shape_vertices = self.path.vertices
        self.shape_codes = self.path.codes
        Shapes.__init__(self, hatch, density)



def truncate_strings(array):
    truncated_array = []
    for string in array:
        if len(string) > 5:
            truncated_array.append(string[:5] + "..")
        else:
            truncated_array.append(string)
    return truncated_array

# 示例用法
# my_array = ["apple", "banana", "orange", "watermelon", "kiwi"]
# result_array = truncate_strings(my_array)
# print(result_array)
# label_font_size = 88
# dot_size = 70
# reduction_number = 5
# SSSM_reduction_number = 20
# legend_size = 120
# mark_scale_size = 6
# grid_alpha = 0.8
# text_font_size = 80
# rotation = 90
# line_width = 20





amgt_mixed_space=185
cusparse_space=amgt_mixed_space-10
amgt_space=amgt_mixed_space-13
matrix_space=amgt_mixed_space+35
xlabel_space=100
        
label_font_size = 9
dot_size = 7
reduction_number = 0.5
legend_size = 12
mark_scale_size = 0.6
grid_alpha = 0.08
text_font_size = 8
rotation = 0
line_width = 1
colors = ["#66976e", 'black', "#8c9ecb",'black']
spgemm_hatch = 'sss'
spmv_hatch = 'xxx'
bar_space=1.3

# single_labels=['CuSparse\nAMGT\nMixed']

def AMGT(plt, num):
    _hatch_types.append(SquareHatch)
    if (num == 0):
        df1 = pd.read_csv(
            "finaldata.csv")
        matrix1 = list(map(str, df1.matrix.values.tolist()))
        cusparse_setup = list(map(float, df1.cusparse_setup.values.tolist()))
        amgt_setup = list(map(float, df1.amgt_setup.values.tolist()))
        amgt_mixed_setup = list(map(float, df1.amgt_mixed_setup.values.tolist()))
        cusparse_spgemm = list(map(float, df1.cusparse_spgemm.values.tolist()))
        amgt_spgemm = list(map(float, df1.amgt_spgemm.values.tolist()))
        amgt_mixed_spgemm = list(map(float, df1.amgt_mixed_spgemm.values.tolist()))
        cusparse_solve = list(map(float, df1.cusparse_solve.values.tolist()))
        amgt_solve = list(map(float, df1.amgt_solve.values.tolist()))
        amgt_mixed_solve = list(map(float, df1.amgt_mixed_solve.values.tolist()))
        cusparse_spmv = list(map(float, df1.cusparse_spmv.values.tolist()))
        amgt_spmv = list(map(float, df1.amgt_spmv.values.tolist()))
        amgt_mixed_spmv = list(map(float, df1.amgt_mixed_spmv.values.tolist()))


        cusparse_sum = []
        for i in range(len(cusparse_setup)):
                cusparse_sum.append(cusparse_setup[i] + cusparse_solve[i])
        amgt_sum = []
        for i in range(len(amgt_setup)):
                amgt_sum.append(amgt_setup[i] + amgt_solve[i])
        amgt_mixed_sum = []
        for i in range(len(amgt_mixed_setup)):
                amgt_mixed_sum.append(amgt_mixed_setup[i] + amgt_mixed_solve[i])

        x = np.arange(len(matrix1))
        width = 0.2

        plt.bar(x -bar_space*width, cusparse_setup, width, color=colors[0],
                edgecolor='black', linewidth=line_width, label='Setup', zorder=102)
        plt.bar(x, amgt_setup, width, color=colors[0], edgecolor='black',
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_setup, width, color=colors[0],
                edgecolor='black', linewidth=line_width, zorder=102)
        
        plt.bar(x -bar_space*width, cusparse_spgemm, width,hatch=spgemm_hatch, color='none',
                edgecolor=colors[1], linewidth=line_width, label='SpGEMM', zorder=102)
        plt.bar(x, amgt_spgemm, width, hatch=spgemm_hatch,color='none', edgecolor=colors[1],
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_spgemm, width, hatch=spgemm_hatch,color='none',
                edgecolor=colors[1], linewidth=line_width, zorder=102)
        
        plt.bar(x -bar_space*width, cusparse_solve, width,bottom=cusparse_setup, color=colors[2],
                edgecolor='black', linewidth=line_width, label='Solve', zorder=102)
        plt.bar(x, amgt_solve, width, bottom=amgt_setup, color=colors[2], edgecolor='black',
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_solve, width, bottom=amgt_mixed_setup,color=colors[2],
                edgecolor='black', linewidth=line_width, zorder=102)
        

        plt.bar(x -bar_space*width, cusparse_spmv, width,bottom=cusparse_setup, color='none',
                hatch=spmv_hatch, edgecolor=colors[3], linewidth=line_width, label='SpMV', zorder=102)
        plt.bar(x, amgt_spmv, width, bottom=amgt_setup, color='none', hatch=spmv_hatch, edgecolor=colors[3],
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_spmv, width, bottom=amgt_mixed_setup,color='none',
                hatch=spmv_hatch, edgecolor=colors[3], linewidth=line_width, zorder=102)
        
        for a, b, c in zip(x, cusparse_sum, cusparse_sum):
            plt.text(a-bar_space*width, b + 8, '%.2f' % c, ha='center',
                        va='bottom', fontsize=text_font_size, zorder=102,rotation=90)
        for a, b, c in zip(x, amgt_sum, amgt_sum):
            plt.text(a, b + 8, '%.2f' % c, ha='center',
                        va='bottom', fontsize=text_font_size, zorder=102,rotation=90)
        for a, b, c in zip(x, amgt_mixed_sum, amgt_mixed_sum):
            plt.text(a+bar_space*width, b + 8, '%.2f' % c, ha='center',
                        va='bottom', fontsize=text_font_size, zorder=102,rotation=90)            



        plt.grid(zorder=1)

        
        plt.tick_params(axis='both', labelsize=label_font_size)
        plt.set_ylabel('Time (ms)', fontsize=label_font_size*1.8)
        plt.set_xlabel('Performance comparison', loc="center",
                       fontsize=label_font_size*1.8,labelpad=xlabel_space)
        plt.set_ylim(0, 510)
        plt.margins(x=0.01)
        matrix1=truncate_strings(matrix1)
        plt.set_xticks([])
        for i in range(len(x)):
             plt.text(i-bar_space*width, -cusparse_space, 'Hypre (FP64)', ha='center',rotation=90,fontsize=label_font_size)
             plt.text(i, -amgt_space, 'AmgT (FP64)', ha='center',rotation=90,fontsize=label_font_size)
             plt.text(i+bar_space*width, -amgt_mixed_space, 'AmgT (Mixed)', ha='center',rotation=90,fontsize=label_font_size)
            #  plt.text(i,-amgt_mixed_space,my_list[i],ha='center',rotation=90,fontsize=label_font_size)
             plt.text(i, -matrix_space, matrix1[i], ha='center',rotation=0,fontsize=label_font_size*1.2)

        # plt.set_xticklabels(matrix1, rotation=rotation,
        #                     fontsize=label_font_size)
        plt.legend(loc='upper center',bbox_to_anchor=(0, 1.07, 1, 0.1),
                   borderaxespad=0, fontsize=legend_size*1.2, ncol=4)
    else:
        df1 = pd.read_csv("data2.csv")
        matrix1 = list(map(str, df1.matrix.values.tolist()))
        cusparse_setup = list(map(float, df1.cusparse_setup.values.tolist()))
        amgt_setup = list(map(float, df1.amgt_setup.values.tolist()))
        amgt_mixed_setup = list(map(float, df1.amgt_mixed_setup.values.tolist()))
        cusparse_spgemm = list(map(float, df1.cusparse_spgemm.values.tolist()))
        amgt_spgemm = list(map(float, df1.amgt_spgemm.values.tolist()))
        amgt_mixed_spgemm = list(map(float, df1.amgt_mixed_spgemm.values.tolist()))
        cusparse_solve = list(map(float, df1.cusparse_solve.values.tolist()))
        amgt_solve = list(map(float, df1.amgt_solve.values.tolist()))
        amgt_mixed_solve = list(map(float, df1.amgt_mixed_solve.values.tolist()))
        cusparse_spmv = list(map(float, df1.cusparse_spmv.values.tolist()))
        amgt_spmv = list(map(float, df1.amgt_spmv.values.tolist()))
        amgt_mixed_spmv = list(map(float, df1.amgt_mixed_spmv.values.tolist()))


        cusparse_sum = []
        for i in range(len(cusparse_setup)):
                cusparse_sum.append(cusparse_setup[i] + cusparse_solve[i])
        amgt_sum = []
        for i in range(len(amgt_setup)):
                amgt_sum.append(amgt_setup[i] + amgt_solve[i])
        amgt_mixed_sum = []
        for i in range(len(amgt_mixed_setup)):
                amgt_mixed_sum.append(amgt_mixed_setup[i] + amgt_mixed_solve[i])

        x = np.arange(len(matrix1))
        width = 0.2

        plt.bar(x -bar_space*width, cusparse_setup, width, color=colors[0],
                edgecolor='black', linewidth=line_width, label='Setup', zorder=102)
        plt.bar(x, amgt_setup, width, color=colors[0], edgecolor='black',
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_setup, width, color=colors[0],
                edgecolor='black', linewidth=line_width, zorder=102)
        
        plt.bar(x -bar_space*width, cusparse_spgemm, width,hatch=spgemm_hatch, color='none',
                edgecolor=colors[1], linewidth=line_width, label='SpGEMM', zorder=102)
        plt.bar(x, amgt_spgemm, width, hatch=spgemm_hatch,color='none', edgecolor=colors[1],
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_spgemm, width, hatch=spgemm_hatch,color='none',
                edgecolor=colors[1], linewidth=line_width, zorder=102)
        
        plt.bar(x -bar_space*width, cusparse_solve, width,bottom=cusparse_setup, color=colors[2],
                edgecolor='black', linewidth=line_width, label='Solve', zorder=102)
        plt.bar(x, amgt_solve, width, bottom=amgt_setup, color=colors[2], edgecolor='black',
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_solve, width, bottom=amgt_mixed_setup,color=colors[2],
                edgecolor='black', linewidth=line_width, zorder=102)
        

        plt.bar(x -bar_space*width, cusparse_spmv, width,bottom=cusparse_setup, color='none',
                hatch=spmv_hatch, edgecolor=colors[3], linewidth=line_width, label='SpMV', zorder=102)
        plt.bar(x, amgt_spmv, width, bottom=amgt_setup, color='none', hatch=spmv_hatch, edgecolor=colors[3],
                linewidth=line_width, zorder=102)
        plt.bar(x +bar_space*width, amgt_mixed_spmv, width, bottom=amgt_mixed_setup,color='none',
                hatch=spmv_hatch, edgecolor=colors[3], linewidth=line_width, zorder=102)

        for a, b, c in zip(x, cusparse_sum, cusparse_sum):
            plt.text(a-bar_space*width, b + 8, '%.2f' % c, ha='center',
                        va='bottom', fontsize=text_font_size, zorder=102,rotation=90)
        for a, b, c in zip(x, amgt_sum, amgt_sum):
            plt.text(a, b + 8, '%.2f' % c, ha='center',
                        va='bottom', fontsize=text_font_size, zorder=102,rotation=90)
        for a, b, c in zip(x, amgt_mixed_sum, amgt_mixed_sum):
            plt.text(a+bar_space*width, b + 8, '%.2f' % c, ha='center',
                        va='bottom', fontsize=text_font_size, zorder=102,rotation=90)    

        plt.grid(zorder=1)

        
        plt.tick_params(axis='both', labelsize=label_font_size)
        plt.set_ylabel('Time (ms)', fontsize=label_font_size*1.8)
        plt.set_xlabel('(b) Performance comparison on H100', loc="center",
                    fontsize=label_font_size*1.8,labelpad=xlabel_space)
        plt.set_ylim(0, 510)
        plt.margins(x=0.01)
        matrix1=truncate_strings(matrix1)
        plt.set_xticks([])
        for i in range(len(x)):
             plt.text(i-bar_space*width, -cusparse_space, 'Hypre (FP64)', ha='center',rotation=90,fontsize=label_font_size)
             plt.text(i, -amgt_space, 'AmgT (FP64)', ha='center',rotation=90,fontsize=label_font_size)
             plt.text(i+bar_space*width, -amgt_mixed_space, 'AmgT (Mixed)', ha='center',rotation=90,fontsize=label_font_size)
            #  plt.text(i,-amgt_mixed_space,my_list[i],ha='center',rotation=90,fontsize=label_font_size)
             plt.text(i, -matrix_space, matrix1[i], ha='center',rotation=0,fontsize=label_font_size*1.2)


def main():
        fig, ax = plt.subplots(dpi=100, figsize=(24, 4))
        plt.subplots_adjust(hspace=0.6, wspace=0.1)

        for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_linewidth(1)

        AMGT(ax, 0)

        plt.savefig("Fig7.pdf", bbox_inches='tight', dpi=300)
#     AMGT(axes[0], 0)
#     # AMGT(axes[1], 1)

#     plt.savefig("Fig7.pdf",
#                 bbox_inches='tight',dpi=300)

    # plt.show()


if __name__ == '__main__':
    main()
