import pandas as pd

def propagate_labels(A_path, B_path, output_path):
    # 读取数据集A和B
    A = pd.read_csv(A_path)
    B = pd.read_csv(B_path)

    # 初始化输出集E和标签集Le
    E = A.copy()
    Le = A['tag'].copy()

    # 遍历数据集A和B中的流
    for i in range(len(A)):
        xai = A.iloc[i]['3-tuple']  # 获取A中的3元组
        yai = A.iloc[i]['tag']      # 获取A中的tag

        for j in range(len(B)):
            xbj = B.iloc[j]['3-tuple']  # 获取B中的3元组

            # 比较三元组，如果相同则传播标签
            if xai == xbj:
                # 更新B中的标签
                B.at[j, 'tag'] = yai
                # 将未标记流添加到E中
                E = pd.concat([E, B.iloc[[j]]], ignore_index=True)
                # 将对应的标签yai添加到Le中
                Le = pd.concat([Le, pd.Series(yai)], ignore_index=True)

    E.to_csv(output_path, index=False)

    return E, B
if __name__ == '__main__':

    labeled_num = [25]
    for k in labeled_num:
        A_path = "../Feature_Select_L/fea_split/fea_" + str(k) + "labeled_features.csv"
        B_path = "../Train_Test_TrainUnlabeled/train_unlabeled/train_" + str(k) + "unlabeled.csv"
        output_path = "../Input/extended_" + str(k) + "labeled_features.csv"
        print(A_path)
        propagate_labels(A_path, B_path, output_path)