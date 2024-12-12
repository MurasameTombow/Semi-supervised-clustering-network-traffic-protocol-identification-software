import pandas as pd
import numpy as np


def split(flow_labeled, flow_unlabeled,k):

    # 1. 加载数据集
    labeled_df = pd.read_csv(flow_labeled)  # 标记数据集
    unlabeled_df = pd.read_csv(flow_unlabeled)  # 未标记数据集

    # 2. 对未标记数据集按流类型随机划分
    # 假设'tag'字段用于标记流的类型，未标记数据集的'tag'为-1，按照'real_tag'进行分组
    grouped = unlabeled_df.groupby('real_tag')

    # 创建两个空数据框来存放分割后的Train和Test
    train_unlabeled = pd.DataFrame()
    test_unlabeled = pd.DataFrame()

    # 遍历每个流类型，随机划分
    for tag, group in grouped:
        # 将当前流类型的数据随机打乱
        shuffled_group = group.sample(frac=1, random_state=42)

        # 按照一半比例划分数据
        split_idx = len(shuffled_group) // 2
        train_unlabeled = pd.concat([train_unlabeled, shuffled_group.iloc[:split_idx]], ignore_index=True)
        test_unlabeled = pd.concat([test_unlabeled, shuffled_group.iloc[split_idx:]], ignore_index=True)

    # 3. 将标记数据和一半的未标记数据集结合起来作为Train
    train = pd.concat([labeled_df, train_unlabeled], ignore_index=True)

    # 4. 另一半未标记数据集作为Test
    test = test_unlabeled

    # 5. 保存结果
    train_file = "./train/Train" + str(k) + ".csv"
    test_file = "./test/Test" + str(k) + ".csv"
    train_unlabeled_file = "./train_unlabeled/train_" + str(k) + "unlabeled.csv"
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    train_unlabeled.to_csv(train_unlabeled_file)#用于流标签传播的未标记数据集

if __name__ == '__main__':
    labeled_num = [25]
    for k in labeled_num:
        flow_labeled = "../Feature_Select_L/fea_split/fea_" + str(k) + "labeled_features.csv"
        flow_unlabeled = "../Feature_Select_L/fea_split/fea_" + str(k) + "unlabeled_features.csv"
        print(flow_labeled)
        split(flow_labeled,flow_unlabeled,k)
