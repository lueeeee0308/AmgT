import csv
import pandas as pd
from functools import reduce

# 要合并的csv文件列表
csv_files = ["cusparse.csv", "nomix.csv", "mix.csv"]

# 读取第一个csv文件以获取合并键的列名
first_dataframe = pd.read_csv(csv_files[0])
# 假设合并键是每个csv文件的第一列
merge_key = first_dataframe.columns[0]  # 这里是0，意味着我们用第一列作为主键

# 创建一个DataFrame列表来存储所有csv数据
# 第一个DataFrame已经被读取了，所以从第二个DataFrame开始
dataframes = [first_dataframe] + [pd.read_csv(csv) for csv in csv_files[1:]]

# 使用reduce函数和merge进行横向合并
merged_data = reduce(lambda left, right: pd.merge(left, right, on=merge_key), dataframes)

# 将合并的数据写入新的csv文件
merged_data.to_csv("merged.csv", index=False)

# 处理数据得到新的csv文件

# input_file = "merged.csv"
# output_file = "finaldata.csv"

def read_and_transform_csv(input_file, output_file):
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        new_headers = ['matrix', 'cusparse_setup', 'cusparse_spgemm', 'cusparse_solve', 'cusparse_spmv', 
                       'amgt_setup', 'amgt_spgemm', 'amgt_solve', 'amgt_spmv', 
                       'amgt_mixed_setup', 'amgt_mixed_spgemm', 'amgt_mixed_solve', 'amgt_mixed_spmv']
        csv_writer.writerow(new_headers)

        for row in csv_reader:
            transformed_row = [row[0], row[7], row[8], row[11], row[12], 
                                       float(row[17]) + (float(row[7]) - float(row[8])), row[17], float(row[21]) + (float(row[11]) - float(row[12])), row[21],
                                       float(row[26]) + (float(row[7]) - float(row[8])), row[26], float(row[31]) + float(row[32]) + (float(row[21]) - float(row[22]) - float(row[23])) + (float(row[11]) - float(row[12])), 
                                       float(row[31]) + float(row[32]) + (float(row[21]) - float(row[22]) - float(row[23]))
                                       ]
            csv_writer.writerow(transformed_row)

read_and_transform_csv('merged.csv', 'finaldata.csv')
