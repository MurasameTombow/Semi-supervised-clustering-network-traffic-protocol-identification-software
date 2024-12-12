import pickle

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, dia_matrix, diags

from sklearn.feature_selection import SelectKBest
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#初始化 LaplacianScore 类，接受四个参数：
# k_neighbors: 用于近邻图的邻居数量。
# lambda_value: 用于计算权重矩阵的参数。
# must_link_constraints: 必连约束集。
# cannot_link_constraints: 勿连约束集。
class LaplacianScore:
    def __init__(self, k_neighbors, lambda_value,must_link_constraints,cannot_link_constraints):
        self.k_neighbors = k_neighbors
        self.lambda_value = lambda_value
        self.must_link_constraints = must_link_constraints
        self.cannot_link_constraints = cannot_link_constraints

    def fit(self, X):
        num_samples, num_features = X.shape

        scaler = StandardScaler()
        X = scaler.fit_transform(X)#使用 StandardScaler 对数据进行标准化处理

        # 计算近邻图 Gkn,使用 NearestNeighbors 计算近邻图，其中每个点与其最近的 k_neighbors 个点相连
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)

        knn.fit(X)

        _, indices = knn.kneighbors(X)

        # print(cannot_link_constraints)
        # print("GKn")

        Gkn = lil_matrix((num_samples, num_samples),dtype=int)
        for i, neighbors in enumerate(indices):
            Gkn[i, neighbors[1:]] = 1
        # print(Gkn.tocsc())
        # 构建must-link约束图
        for i, j in self.must_link_constraints:
            Gkn[i, j] = 1
            Gkn[j, i] = 1
        # print(Gkn)
        # print("GCL")
        # 构建负约束图 GCL
        GCL = lil_matrix((num_samples, num_samples),dtype=int)
        # 构建负约束图 GCL
        for i, j in self.cannot_link_constraints:
            GCL[i, j] = 1
            GCL[j, i] = 1
        # print(GCL)
        # print("SKN")
        # 构建权重矩阵 Skn
        Skn = lil_matrix((num_samples, num_samples),dtype=np.float32)

        for i,row in enumerate(Gkn.rows):
            for j in row:
                distance = np.linalg.norm(X[i] - X[j])
                # Skn[i, j] = np.exp(-distance ** 2)
                Skn[i, j] = np.exp(-distance ** 2 / self.lambda_value)
                # Skn[i, j] = 1

        # 构建权重矩阵 SCL
        SCL = lil_matrix((num_samples, num_samples),dtype=int)
        for i,row in enumerate(GCL.rows):
            for j in row:
                SCL[i, j] = 1

        del Gkn
        del GCL
        # 构建拉普拉斯矩阵 Lkn
        diagonal_data1 = Skn.sum(axis=1).A.flatten()
        Dkn = diags(diagonal_data1, offsets=0, shape=Skn.shape, format='lil')
        # print(Dkn)
        Lkn = Dkn - Skn

        # 构建拉普拉斯矩阵 LCL
        diagonal_data2 = SCL.sum(axis=1).A.flatten()
        DCL = diags(diagonal_data2, offsets=0, shape=SCL.shape, format='lil')
        # DCL = dia_matrix(SCL.sum(axis=1))
        LCL = DCL - SCL
        del diagonal_data1,diagonal_data2
        del DCL
        CLS = []
        for i in range(len(X[0])):
            fr = lil_matrix(X[:, i])  #(1,n)
            # print(fr.shape)
            # print(fr.T.shape)
            # print(Lkn.shape)
            numerator = fr.dot(Lkn).dot(fr.T)
            denominator = fr.dot(LCL).dot(Dkn).dot(fr.T)

            if (fr.nnz==0):
                score  = 0
            else:
                # print(numerator,denominator)
                score = numerator / denominator
                # print(score[0,0])
            if (type(score) == int):
                CLS.append(score)
            else:

                CLS.append(score[0,0])

        self.scores_ = CLS
        self.indices_ = np.argsort(CLS)[::-1]
        return self


if __name__ == '__main__':
    labeled_num = [25]
    for k in labeled_num:
        fea_file = "../flow_feature/flow_fea" + str(k) + ".csv"
        data = pd.read_csv(fea_file)
        X = np.array(data.iloc[:, 3:-2])
        ml = "../data_info/ml/ml" + str(k) + ".pkl"
        cl = "../data_info/cl/cl" + str(k) + ".pkl"
        with open(ml, 'rb') as f:
            must_link_constraints = pickle.load(f)
        with open(cl, 'rb') as f:
            cannot_link_constraints = pickle.load(f)
        laplacian_score = LaplacianScore(k_neighbors=5, lambda_value=0.1,
                                         must_link_constraints=must_link_constraints,
                                         cannot_link_constraints=cannot_link_constraints)
        laplacian_score.fit(X)
        data = data.iloc[:, 3:-2]
        data = data.iloc[:, laplacian_score.indices_[:40]]
        flow_id = pd.read_csv(fea_file).iloc[:, 0]
        three_tuple = pd.read_csv(fea_file).iloc[:, 1]
        label = pd.read_csv(fea_file).iloc[:, -2]
        real_label = pd.read_csv(fea_file).iloc[:, -1]
        concatenated_df = pd.concat([flow_id, three_tuple, data, label, real_label], axis=1)
        # concatenated_df = concatenated_df.reset_index(drop=True)
        # 写入拼接后的数据到新的 CSV 文件
        concatenated_df.to_csv("./fea" + str(k) + ".csv",index=False)
