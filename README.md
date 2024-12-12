# 半监督聚类网络流量协议识别软件

1. ## 特点

   ​	本软件提取已标记和未标记网络流量统计特征，结合少量已知协议样本与流相关性构建约束信息，通过计算约束拉普拉斯分数实现特征选择，同时使用流标签传播算法扩充标记流量样本。通过K-means算法，对训练样本进行分类并依靠扩充后的标记流量样本计算后验概率完成对流量簇的识别，可以识别出从未出现或不在数据库中的协议类型。

   ​	本软件还构造了流量复合分类器，根据三元组将待识别的测试集流量划分成为一组组具有相关性的流量集合，以这些流量集合为单位采用复合专家投票的方式对测试集流量进行识别，提升了未知协议识别的准确率与效率，并有效减少对标记样本的依赖。实验结果表明，该方法在少量标记样本和较低聚类数条件下，能实现对未知协议的高准确率和纯度识别。

2. ## 目录介绍

   Data文件夹

   该文件夹下存储网络流量PCAP文件。文件夹中已经创建好不同协议类型的文件夹用于存放不同协议的流量，可以自行更改。注意：每个PCAP文件应当先通过五元组（源IP、目的IP、源端口、目的端口、传输协议）组流再存入。

   Data_Process文件夹

   - Feature_Select_L
     - flow_feature
       - flow_features_extract.py：对Data文件夹中的原始流量进行流统计特征提取，并对部分流量进行标记
     - data_info
       - data_info_generate.py：结合少量已知协议样本与流相关性构建约束信息，包括ml与cl，分别表示相关的流的集合与不相关的流的集合
     - feature_select
       - feature_select.py：结合约束信息，计算拉普拉斯分数进行特征选择
     - fea_split
       - fea_split.py：对特征选择后的流样本进行划分，分为已标记流量样本与无标记流量样本
   - Train_Test_TrainUnlabeled
     - split_train_and_test.py：将流量样本划分为训练集Train和测试集Test，同时准备Train中的无标记流量样本train_unlabeled.py用于流标签传播算法
   - Label_Propagation
     - flow_label_propagation.py：对训练集进行流标签传播
   - Input
     - Min-Max_Normalization.py：将训练集和测试集最大最小值归一化后当作模型输入
   - NCC_Build.py：执行K-mean算法并构建NCC分类器的文件
   - Compound_classification.py：输入训练集，调用NCC_Build.py完成NCC分类器的构建。输入测试集，使用复合分类方法对测试集进行识别

3. ## 使用方法

   ​	本软件采用模块化设计，每一部分的功能独立，准备好每个模块的需求文件即可可以单独执行。

   1. 首先执行Data_Process/Feature_Select_L/flow_feature/flow_features_extract.py文件，在文件中修改

      ```python 
      p = "../../../Data"#原始流量PCAP文件位置
      labeled_num = [25]#控制每种协议流量标记个数，可以有多种标记。
      #以labeled_num = [25]为例，生成的文件为flow_fea25.csv
      ```

   2. 执行Data_Process/Feature_Select_L/data_info/data_info_generate.py文件，在文件中修改flow_fea25.csv的位置：

      ```python
      labeled_num = [25]#控制每种协议流量标记个数，可以有多种标记。
      fea_file = "../flow_feature/flow_fea" + str(k) + ".csv"#
      ```

      生成的文件分别保存在当前目录下的cl、ml文件夹中，以labeled_num = [25]为例，分别为cl25.pkl、ml25.pkl

   3. 执行Data_Process/Feature_Select_L/feature_select/feature_select.py文件，在文件中修改:

      ```python
      fea_file = "../flow_feature/flow_fea" + str(k) + ".csv"#修改成为flow_fea25.csv的位置
      data = pd.read_csv(fea_file)
      X = np.array(data.iloc[:, 3:-2])#修改为流特征向量的特征列
      ml = "../data_info/ml/ml" + str(k) + ".pkl"
      cl = "../data_info/cl/cl" + str(k) + ".pkl"
      ......
      data = data.iloc[:, 3:-2]#修改为流特征向量的特征列
      data = data.iloc[:, laplacian_score.indices_[:40]]#选取拉普拉斯分数排名前40的特征，可根据自己需要修改
      flow_id = pd.read_csv(fea_file).iloc[:, 0]#修改为流特征向量中flow_id特征列
      three_tuple = pd.read_csv(fea_file).iloc[:, 1]#修改为流特征向量中3-tuple特征列
      label = pd.read_csv(fea_file).iloc[:, -2]#修改为流特征向量中tag特征列
      real_label = pd.read_csv(fea_file).iloc[:, -1]#修改为流特征向量中real_tag特征列
      ```

      生成的文件保存在当前目录下，以labeled_num = [25]为例，生成的文件为fea25.csv

   4. 执行Data_Process/Feature_Select_L/fea_split./fea_split.py文件，在文件中修改:

      ```python
      file_path = "../feature_select/fea" + str(k) + ".csv"#修改成为fea25.csv的位置
      ```

      以labeled_num = [25]为例，生成的文件保存在当前目录下：fea_25labeled_features.csv、fea_25unlabeled_features.csv

   5. 执行Data_Process/Train_Test_TrainUnlabeled/split_train_and_test.py文件，在文件中修改:

      ```python
      flow_labeled = "../Feature_Select_L/fea_split/fea_" + str(k) + "labeled_features.csv"#修改为fea_25labeled_features.csv文件位置
      flow_unlabeled = "../Feature_Select_L/fea_split/fea_" + str(k) + "unlabeled_features.csv"#修改为fea_25unlabeled_features.csv文件位置
      ```

      以labeled_num = [25]为例，生成的文件保存在当前目录test、train、train_unlabeled目录下：Test25.csv、Train25.csv、train_25unlabeled.csv

   6. 执行Data_Process/Label_Propagation/flow_label_propagation.py文件，在文件中修改:

      ```python
      A_path = "../Feature_Select_L/fea_split/fea_" + str(k) + "labeled_features.csv"#修改为fea_25labeled_features.csv文件位置
      B_path = "../Train_Test_TrainUnlabeled/train_unlabeled/train_" + str(k) + "unlabeled.csv"#修改为train_25unlabeled.csv文件位置
      ```

      以labeled_num = [25]为例，生成的文件保存在Input目录中：extended_25labeled_features.csv

   7. 执行Data_Process/Input/Min-Max_Normalization.py文件，在文件中修改:

      ```python
      T_path = "../Train_Test_TrainUnlabeled/train/Train" + str(k) + ".csv"
      Test_path = "../Train_Test_TrainUnlabeled/test/Test" + str(k) + ".csv"
      ```

      以labeled_num = [25]为例，生成的文件保存在Input目录中：Test_25normalized.csv、Train_25normalized.csv

   8. 执行Compound_classification.py文件，在文件中修改：

      ```python
      T_path = "./Data_Process/Input/Train_25normalized.csv"#修改为Train_25normalized.csv文件位置
      E_path = "./Data_Process/Input/extended_25labeled_features.csv"#修改为extended_25labeled_features.csv文件位置
      Test_path = "./Data_Process/Input/Test_25normalized.csv"#修改为Test_25normalized.csv文件位置
      k = 100#初始聚类中数量
      ```

      生成的文件保存在result目录中

4. # 识别结果

   ![](Figure/4af0f2b6-9dff-4fa1-ae99-f9aa87be031b.PNG)