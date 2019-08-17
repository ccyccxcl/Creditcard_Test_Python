# Creditcard-Test-Python
Python分析信用卡反欺诈。

数据集主要显示信用卡交易，284,807笔交易中有492起欺诈行为。存在不平衡。

数据集并非一手数据，经过PCA变换的结果。特征用序号V1，V2，... V28。故缺少更多数据特征，但也省掉数据处理过程。

思路主要是：拿到数据寻找规律、选何种模型来构建反欺诈模型，不停尝试


此项目详细步骤包括：

1 概览数据---数据不平衡

2 各特征下正负反例与标签之间关系----图表

3 数据标准化处理---StandardScaler

4 划分测试集和训练集，引入算法

  #采用回归模型，给出混淆矩阵和精确度以及AUC
  
  #采用随机森林模型，给出混淆矩阵和精确度以及AUC---效果相对较好
  
  #采用SVM模型，给出混淆矩阵和精确度以及AUC
  
5 下采样数据集

6 划分测试集和训练集(下采样之后)，引入算法

  #采用回归模型，给出混淆矩阵和精确度以及AUC---效果相对较好
  
  #采用随机森林模型，给出混淆矩阵和精确度以及AUC
  
  #采用SVM模型，给出混淆矩阵和精确度以及AUC
  
7 通过K阶交叉验证寻找回归模型当中合适的C参数

8 下采样数据集选取C参数

9 最优C参数应用于下采样数据集，得到recall

10 最优C参数应用于原始数据集，得到recall

11 原始数据集选取C参数

12 最优C参数应用于下采样数据集，得到recall

13 最优C参数应用于原始数据集，得到recall

14 固定C参数，选取threhold参数

15 threshold参数分别应用与下采样和原始数据集，比较结果


总结：#看数据的结构，是否平衡；

#若不平衡，采取下采样或过采样获取全新的数据集，再选模型、算法；

#模型的调参不断的试，选最佳的参数；

#预测应综合考虑精度、recall、混淆矩阵等多个参数

