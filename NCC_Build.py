import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle

def bulid_NCC(T_path, E_path, k):
    T = pd.read_csv(T_path)
    T = shuffle(T, random_state=0)  # 打乱数据集T的顺序
    E = pd.read_csv(E_path)
    # 获取流量特征（第3到第42列是特征列）
    T_features = T.iloc[:, 2:-2].values
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=k, random_state=None, n_init=10, max_iter=10, tol=1e-3).fit(T_features)
    cluster_labels = kmeans.labels_  # 聚类结果,一维的数组或列表，表示每个样本所属的簇。它的长度等于输入样本的数量
    centroids = kmeans.cluster_centers_  # 簇质心,一个二维的数组T_features = T.iloc[:, 2:22].values

    # 初始化簇的标签
    cluster_tags = np.full(k, -1)  # 初始标签为-1表示“未知”

    # 遍历每个簇，确定其标签
    for i in range(k):
        # 获取簇 i 中的已标记流量数据
        cluster_indices = np.where(cluster_labels == i)[0]
        # labeled_in_cluster = E[E['flow_id'].isin(T.iloc[cluster_indices]['flow_id'])]

        # 获取该簇中已标记的样本
        labeled_in_cluster = T.iloc[cluster_indices][T.iloc[cluster_indices]['tag'] != -1]

        # 获取该簇中未标记的样本
        unlabeled_in_cluster = T.iloc[cluster_indices][T.iloc[cluster_indices]['tag'] == -1]
        # 对于每个未标记的样本，找到E中具有相同flow_id的样本并确定其标签
        for idx, sample in unlabeled_in_cluster.iterrows():
            sample_flow_id = sample['flow_id']

            # 获取E中具有相同flow_id的样本
            samples_in_E = E[E['flow_id'] == sample_flow_id]

            if not samples_in_E.empty:
                # 获取这些样本的tag分布
                tag_counts = samples_in_E['tag'].value_counts()
                # print(tag_counts)

                # 找到出现次数最多的tag
                most_common_tag = tag_counts.idxmax()

                # 为当前未标记的样本设置标签
                sample['tag'] = most_common_tag

                # 将该未标记样本加入labeled_in_cluster
                labeled_in_cluster = pd.concat([labeled_in_cluster, pd.DataFrame([sample])], ignore_index=True)

        # 如果簇中没有已标记的数据，则标记为“未知”
        if labeled_in_cluster.empty:
            cluster_tags[i] = -1
        else:
            # 计算每个标签的频率，选择出现最多的标签作为该簇的标签
            most_common_tag = labeled_in_cluster['tag'].value_counts().idxmax()
            cluster_tags[i] = most_common_tag

    return centroids, cluster_tags




# 构建NCC分类器
def ncc_classifier(x, centroids, cluster_tags):
    """最近邻簇分类器（NCC），对新样本进行分类"""
    # 计算样本x与每个簇质心的距离
    distances = cdist([x], centroids, metric='euclidean')
    # 找到距离最近的簇
    nearest_cluster = np.argmin(distances)
    # 返回该簇的标签
    return cluster_tags[nearest_cluster]

if __name__ == '__main__':
    # 读取数据集
    T_path = "../Data_Process/Input/Train_25normalized.csv"
    E_path = "../Data_Process/Input/extended_25labeled_features.csv"
    # 测试NCC分类器
    centroids, cluster_tags = bulid_NCC(T_path, E_path, 50)
