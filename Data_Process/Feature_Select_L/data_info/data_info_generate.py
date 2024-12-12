
import pickle
import pandas as pd

def read_pcap(file_path):
    packets = []
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            if not len(eth.data.data.data):
                continue
            packets.append(eth.data.data.data[:32])
    return packets


import dpkt
from collections import defaultdict
import re
def natural_sort_key(s):
    # 将文件名按照数字和非数字的部分拆分，然后按照数字部分进行排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


# 构建约束集
def build_constraint_set(csv,labeled_num):

    x = pd.read_csv(csv)

    flow_id = x.iloc[:, 0].values
    times = x.iloc[:, 2].values
    labels = x.iloc[:, -2].values

    n = len(labels)  # 样本数量
    ml_set = set()
    cl_set = set()

    for i in range(n):
        label_i = labels[i]
        # 如果两个样本的标签相同且标签不是 -1，或者两个样本属于同一流且时间间隔不超过60秒，且其中一个样本的标签为 -1，则将其加入必连集 ml_set。
        # 如果两个样本的标签不同且都不是 -1，则将其加入勿连集 cl_set。
        for j in range(i + 1, n):
            label_j = labels[j]

            if (label_i == label_j and label_i != -1) or \
                    ((flow_id[i] == flow_id[j] and abs(times[i] - times[j]) <= 60) and (label_i == -1 or label_j == -1)):
                ml_set.add((i, j))
            if (label_i != label_j and label_i != -1 and label_j != -1 and (j,i) not in cl_set) :

                cl_set.add((i,j))

    file1 = "./ml/ml" + str(labeled_num) + ".pkl"
    file2 = "./cl/cl" + str(labeled_num) + ".pkl"
    with open(file1, 'wb') as f:
        pickle.dump(ml_set,f)
    with open(file2, 'wb') as f:
        pickle.dump(cl_set,f)
    return ml_set, cl_set


if __name__ == '__main__':

    labeled_num = [25]#控制每种类型的样本的标记样本数量，例如labeled_num = [5,10,15,20,25,30,35,40,45,50]
    for k in labeled_num:
        fea_file = "../flow_feature/flow_fea" + str(k) + ".csv"
        build_constraint_set(fea_file,k)
