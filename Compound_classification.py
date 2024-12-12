import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import itertools
from NCC_Build import ncc_classifier, bulid_NCC
import os

def compound_classification(test_path, T_path, E_path, k):
    centroids, cluster_tags = bulid_NCC(T_path, E_path, k)
    # 读取测试集
    Test = pd.read_csv(test_path)
    BoFs = {}  # 存储每个 BoF 的流

    # 根据流的 3-tuple 构建 BoFs
    for index, row in Test.iterrows():
        flow_tuple = row['3-tuple']
        if flow_tuple not in BoFs:
            BoFs[flow_tuple] = []
        BoFs[flow_tuple].append(row)

    classification_results = []
    # 遍历每个 BoF
    for flow_group in BoFs.values():
        g = len(flow_group)#g 是当前 BoF 中流的数量
        # print(f"BoF: {flow_group[0]['3-tuple']}, 流的数量 g: {g}")
        votes = np.zeros((g, len(cluster_tags)))  # 初始化投票矩阵,行数为 g（流的数量），列数为 cluster_tags 的数量。这个矩阵用于记录每个流对各个簇的投票情况

        for i in range(g):#遍历当前 BoF 中的每个流。
            features = flow_group[i][2:-2].astype(float).values  # 获取统计特征
            predicted_tag = ncc_classifier(features, centroids, cluster_tags)  # 预测标签
            for j in range(len(cluster_tags)):#对于每个簇标签，如果预测的标签与当前簇标签相同，则在投票矩阵中记录一票。
                if predicted_tag == cluster_tags[j]:
                    votes[i][j] = 1  # 标记为投票

        # 根据投票结果为所有流分配类别
        assigned_class = cluster_tags[np.argmax(np.sum(votes, axis=0))]
        for flow in flow_group:
            # 获取当前流的信息
            flow_info = flow.to_dict()  # 将流的信息转换为字典
            flow_info['assigned_class'] = assigned_class  # 添加assigned_class

            # 将信息添加到分类结果中
            classification_results.append(flow_info)
    return classification_results


# 使用方法
if __name__ == '__main__':
    T_path = "./Data_Process/Input/Train_25normalized.csv"
    E_path = "./Data_Process/Input/extended_25labeled_features.csv"
    Test_path = "./Data_Process/Input/Test_25normalized.csv"
    k = 100

    results = compound_classification(Test_path, T_path, E_path,k)


    # 将结果保存到文件
    results_df = pd.DataFrame(results)
    results_df.to_csv("./result/compound_classification_results.csv", index=False)