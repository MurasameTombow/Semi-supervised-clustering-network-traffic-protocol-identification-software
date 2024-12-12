import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 读取 CSV 文件
data = pd.read_csv("compound_classification_results.csv", delimiter=",", encoding="utf-8")

# 过滤掉 real_tag = -1 的无效标签（真实标签未知的样本不参与评估）
valid_data = data[data["real_tag"] != -1]

# 1. 计算已识别流量类别的总体准确率和加权 F1-Score
recognized_data = valid_data[valid_data["assigned_class"] != -1]
real_tags_recognized = recognized_data["real_tag"]
assigned_classes_recognized = recognized_data["assigned_class"]

# 总体准确率
accuracy = accuracy_score(real_tags_recognized, assigned_classes_recognized)

# 总体加权 F1-Score
weighted_f1 = f1_score(real_tags_recognized, assigned_classes_recognized, average="weighted")

# 2. 计算每种类别的 F1-Score
# 使用分类报告获取每个类别的 F1-Score
classification_report_dict = classification_report(
    real_tags_recognized,
    assigned_classes_recognized,
    output_dict=True
)

# 提取每种类别的 F1-Score
category_f1_scores = {
    f"Class {label}": scores["f1-score"]
    for label, scores in classification_report_dict.items()
    if label.isdigit()
}


# 输出结果
print(f"Accuracy (recognized flows): {accuracy:.4f}")
print(f"F1-Score (recognized flows, weighted): {weighted_f1:.4f}")
print("F1-Score for each class:")
for cls, f1 in category_f1_scores.items():
    print(f"  {cls}: {f1:.4f}")