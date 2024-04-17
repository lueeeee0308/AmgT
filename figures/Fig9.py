import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

label_font_size = 28
dot_size = 70
legend_size = 120
mark_scale_size = 6
grid_alpha = 0.8
text_font_size = 80
rotation = 90
width = 0.4

df1 = pd.read_csv('preprocess_test.csv')
matrix = list(map(str, df1.matrix.values.tolist()))
csr2bsr = list(map(float, df1.uscsr2bsr.values.tolist()))
cusparse_csr2bsr = list(
    map(float, df1.cusparse_csr2bsr.values.tolist()))



def speedup(n):
    k = [1] * len(n)
    for i in range(len(n)):
        k[i] = n[i]
    return k


def draw(plt):
    
    x = np.arange(len(matrix))
    now_csr2bsr = speedup(csr2bsr)
    now_cusparse_csr2bsr = speedup(cusparse_csr2bsr)

    plt.bar(x - 0.5*width, now_csr2bsr, width, color='moccasin', edgecolor='black', linewidth=1, label='AmgT (CSR to mBSR)', zorder=102)
    plt.bar(x+0.5*width, now_cusparse_csr2bsr, width, color='darksalmon', edgecolor='black', linewidth=1, label='cuSPARSE (CSR to BSR)', zorder=102)
    
    
    plt.xticks(range(len(matrix)),matrix, rotation=rotation, fontsize=26)
    plt.yticks(fontsize=18)
    plt.ylabel('Runtime (ms)', fontsize=35)
    plt.ylim(0,10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(bbox_to_anchor=(0.6, 0.98),fontsize = 25)



def main():
    plt.figure(figsize=(12, 5))
    # print(matrix)
    # print(csr2bsr)
    # print(cusparse_csr2bsr)
    draw(plt)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)
    plt.savefig("Fig9.pdf",
                bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
