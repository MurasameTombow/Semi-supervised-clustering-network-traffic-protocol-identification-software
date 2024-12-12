import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scaler(T_path, Test_path,k):
    # 读取 Train.csv 文件
    T = pd.read_csv(T_path)
    Test = pd.read_csv(Test_path)

    # 提取需要归一化的特征列（假设第 3 到第 22 列为需要归一化的特征列）
    features_train = T.iloc[:, 2:42].values
    features_test = Test.iloc[:, 2:42].values
    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()

    # 对特征列进行归一化
    normalized_features_train = scaler.fit_transform(features_train)
    normalized_features_test = scaler.fit_transform(features_test)

    # 将归一化的特征替换到原数据中
    T.iloc[:, 2:42] = normalized_features_train
    Test.iloc[:, 2:42] = normalized_features_test

    # 保存归一化后的数据到新文件
    Train_normalized = "./Train_" + str(k) + "normalized.csv"
    Test_normalized = "./Test_" + str(k) + "normalized.csv"
    T.to_csv(Train_normalized, index=False)
    Test.to_csv(Test_normalized, index=False)

if __name__ == '__main__':
    labeled_num = [25]
    for k in labeled_num:
        T_path = "../Train_Test_TrainUnlabeled/train/Train" + str(k) + ".csv"
        Test_path = "../Train_Test_TrainUnlabeled/test/Test" + str(k) + ".csv"
        scaler(T_path, Test_path, k)