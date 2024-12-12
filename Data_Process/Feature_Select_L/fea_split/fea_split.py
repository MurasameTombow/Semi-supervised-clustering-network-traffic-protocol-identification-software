import pandas as pd

def split(k):
    # 读取CSV文件
    file_path = "../feature_select/fea" + str(k) + ".csv"
    print(file_path)
    data = pd.read_csv(file_path)

    # 按照 tag 列是否为 -1 分割数据
    data_tag_minus1 = data[data['tag'] == -1]  # tag 为 -1 的数据
    data_tag_not_minus1 = data[data['tag'] != -1]  # tag 不为 -1 的数据

    # 保存到新的 CSV 文件中
    data_tag_minus1.to_csv("fea_" + str(k) + "unlabeled_features.csv", index=False)  # 保存 tag = -1 的数据
    data_tag_not_minus1.to_csv("fea_" + str(k) + "labeled_features.csv", index=False)  # 保存 tag != -1 的数据

    print("数据划分完成并已保存到新文件中！")

if __name__ == '__main__':
    labeled_num = [25]
    for k in labeled_num:
        split(k)