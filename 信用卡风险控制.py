
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#原始数据为银行的个人交易记录，每一条信息代表一次交易，原始数据已经进行了类似PCA的处理，现在已经把特征数据提取好了，检测的目的是通过数据找出那些交易存在潜在的欺诈行为。


# In[19]:


data = pd.read_csv('C:\\Users\\ccy\\Desktop\\creditcard.csv')
data.head()


# In[ ]:


#先观察数据
#数据已经经过降维处理，不需要对数据再进行预处理，但数据具体代表的含义不是很清楚
#Amount的浮动范围很大，因此在稍后的过程中要进行归一化处理


# In[ ]:


#计算不同的属性个数以柱状图的形式绘制出


# In[7]:


count_classes = pd.value_counts(data['Class'], sort = True).sort_index() 
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[8]:


#标签为0的很多，而标签为1的却很少，说明样本的分布情况是非常不均衡


# In[20]:


data.info()


# In[21]:


data['Class'].describe()


# In[22]:


print(count_classes)


# In[24]:


f,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(12,6))
bins=50
ax1.hist(data.Time[data.Class == 1],bins=bins)
ax1.set_title('Fraud',fontsize=22)
ax1.set_ylabel('Total',fontsize=15)

ax2.hist(data.Time[data.Class == 0],bins=bins)
ax2.set_title('Normal',fontsize=22)

plt.xlabel('Time',fontsize=15)
plt.xticks(fontsize=15)

plt.ylabel('Total',fontsize=15)
# plt.yticks(fontsize=22)
plt.show()


# In[ ]:


#欺诈与时间并没有必然联系，不存在周期性；正常交易有明显的周期性，有类似双峰这样的趋势


# In[26]:


f,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(12,6))
bins=30
ax1.hist(data.Amount[data.Class == 1],bins=bins)
ax1.set_title('Fraud',fontsize=22)
ax1.set_ylabel('Total',fontsize=15)

ax2.hist(data.Amount[data.Class == 0],bins=bins)
ax2.set_title('Normal',fontsize=22)

plt.xlabel('Amount($)',fontsize=15)
plt.xticks(fontsize=15)

plt.ylabel('Total',fontsize=15)
plt.yscale('log')
plt.show()


# In[27]:


#看看各个变量与正常、欺诈之间是否存在联系


# In[30]:


import matplotlib.gridspec as gridspec
import seaborn as sns; plt.style.use('ggplot')
features=[x for x in data.columns if x not in ['Time','Amount','Class']]
plt.figure(figsize=(12,28*4))
gs =gridspec.GridSpec(28,1)

import warnings
warnings.filterwarnings('ignore')
for i,cn in enumerate(data[features]):
    ax=plt.subplot(gs[i])
    sns.distplot(data[cn][data.Class==1],bins=50,color='red')
    sns.distplot(data[cn][data.Class==0],bins=50,color='green')
    ax.set_xlabel('')
    ax.set_title('直方图：'+str(cn))
plt.savefig('各个变量与class的关系.png',transparent=False,bbox_inches='tight')
plt.show()


# In[34]:


#红色表示欺诈，绿色表示正常
#两个分布的交叉面积越大，欺诈与正常的区分度最小，如V15；
#两个分布的交叉面积越小，则该变量对因变量的影响越大，如V14；


# In[33]:


#对Amount的值进行标准化处理，引入标准化函数,#StandardScaler作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。


# In[35]:


from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))


# In[36]:


#删除Time和Amount所在的列


# In[37]:


data = data.drop(['Time','Amount'],axis=1)
data.head()


# In[ ]:


#划分测试集和训练集


# In[39]:


from sklearn.model_selection import train_test_split
X = data.loc[:,data.columns != 'Class']
Y = data.loc[:,data.columns == 'Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3,random_state=0)


# In[44]:


from sklearn import metrics
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report


# In[ ]:


#采用回归模型，给出混淆矩阵和精确度以及AUC


# In[59]:


from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression(penalty='l2')
lrmodel.fit(X_train, Y_train)
ypred_lr=lrmodel.predict(X_test)
print('confusion_matrix')
print(metrics.confusion_matrix(Y_test,ypred_lr))
print('classification_report')
print(metrics.classification_report(Y_test,ypred_lr))
print('Accuracy:%f'%(metrics.accuracy_score(Y_test,ypred_lr)))
print('Area under the curve:%f'%(metrics.roc_auc_score(Y_test,ypred_lr)))


# In[ ]:


##采用随机森林模型，给出混淆矩阵和精确度以及AUC


# In[47]:


from sklearn.ensemble import RandomForestClassifier
rfmodel=RandomForestClassifier()
rfmodel.fit(X_train, Y_train)
ypred_rf=rfmodel.predict(X_test)
print('confusion_matrix')
print(metrics.confusion_matrix(Y_test,ypred_rf))
print('classification_report')
print(metrics.classification_report(Y_test,ypred_rf))
print('Accuracy:%f'%(metrics.accuracy_score(Y_test,ypred_rf)))
print('Area under the curve:%f'%(metrics.roc_auc_score(Y_test,ypred_rf)))


# In[ ]:


#采用SVM模型，给出混淆矩阵和精确度以及AUC


# In[48]:


from sklearn.svm import SVC
svcmodel=SVC(kernel='sigmoid')
svcmodel.fit(X_train,Y_train)
ypred_svc=svcmodel.predict(X_test)
print('confusion_matrix')
print(metrics.confusion_matrix(Y_test,ypred_svc))
print('classification_report')
print(metrics.classification_report(Y_test,ypred_svc))
print('Accuracy:%f'%(metrics.accuracy_score(Y_test,ypred_svc)))
print('Area under the curve:%f'%(metrics.roc_auc_score(Y_test,ypred_svc)))


# In[ ]:


#文章中涉及到数据严重不平衡，需进行样本数据处理，主要有2种思路：下采样，过采样，这里采用下采样


# In[88]:


normal=data[data['Class']==0]
fraud=data[data['Class']==1]
normal=normal.sample(n=len(Y[Y.Class==1]))
lowerdata=normal.append(fraud)
lowerdata.head()


# In[ ]:


#下采样之后的数据集（特征和标签）


# In[67]:


X_lowerdata = lowerdata.iloc[:,lowerdata.columns != 'Class']
Y_lowerdata = lowerdata.iloc[:,lowerdata.columns == 'Class']


# In[ ]:


#下采样之后的训练集和测试集划分（特征和标签）


# In[68]:


X_train_lowerdata,X_test_lowerdata,Y_train_lowerdata,Y_test_lowerdata = train_test_split(X_lowerdata,Y_lowerdata,test_size=0.3,random_state=0)


# In[69]:


print(X_train_lowerdata.shape, Y_train_lowerdata.shape,'\n',X_test_lowerdata.shape,Y_test_lowerdata.shape)


# In[ ]:


#采用回归模型，给出混淆矩阵和精确度以及AUC


# In[71]:


from sklearn.linear_model import LogisticRegression
lrmodellower = LogisticRegression(penalty='l2')
lrmodellower.fit(X_train_lowerdata, Y_train_lowerdata)
ypred_lrlower=lrmodellower.predict(X_test_lowerdata)
print('confusion_matrix')
print(metrics.confusion_matrix(Y_test_lowerdata,ypred_lrlower))
print('classification_report')
print(metrics.classification_report(Y_test_lowerdata,ypred_lrlower))
print('Accuracy:%f'%(metrics.accuracy_score(Y_test_lowerdata,ypred_lrlower)))
print('Area under the curve:%f'%(metrics.roc_auc_score(Y_test_lowerdata,ypred_lrlower)))


# In[ ]:


#采用随机森林模型，给出混淆矩阵和精确度以及AUC


# In[74]:


from sklearn.ensemble import RandomForestClassifier
rfmodellower=RandomForestClassifier()
rfmodellower.fit(X_train_lowerdata, Y_train_lowerdata)
ypred_rflower=rfmodellower.predict(X_test_lowerdata)
print('confusion_matrix')
print(metrics.confusion_matrix(Y_test_lowerdata,ypred_rflower))
print('classification_report')
print(metrics.classification_report(Y_test_lowerdata,ypred_rflower))
print('Accuracy:%f'%(metrics.accuracy_score(Y_test_lowerdata,ypred_rflower)))
print('Area under the curve:%f'%(metrics.roc_auc_score(Y_test_lowerdata,ypred_rflower)))


# In[ ]:


#采用SVM模型，给出混淆矩阵和精确度以及AUC


# In[75]:


from sklearn.svm import SVC
svcmodellower=SVC(kernel='sigmoid')
svcmodellower.fit(X_train_lowerdata, Y_train_lowerdata)
ypred_svclower=svcmodellower.predict(X_test_lowerdata)
print('confusion_matrix')
print(metrics.confusion_matrix(Y_test_lowerdata,ypred_svclower))
print('classification_report')
print(metrics.classification_report(Y_test_lowerdata,ypred_svclower))
print('Accuracy:%f'%(metrics.accuracy_score(Y_test_lowerdata,ypred_svclower)))
print('Area under the curve:%f'%(metrics.roc_auc_score(Y_test_lowerdata,ypred_svclower)))


# In[119]:


#通过K阶交叉验证寻找合适的C参数


# In[121]:


from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report
def printing_Kfold_scores(X_train,Y_train):
    fold = KFold(len(Y_train),5,shuffle=False)
    print (fold)
    c_param_range = [0.01,0.1,1,10,100]
    # results_table为创建的DataFrame对象，来存储不同参数交叉验证后所得的recall值
    results_table = pd.DataFrame(index=range(len(c_param_range)),columns=['C_Parameter','Mean recall score'])
    results_table['C_Parameter'] = c_param_range

    j=0
    for c_param in c_param_range:
        print ('c_param:',c_param)
        recall_accs = []
        #enumerate将一个可遍历对象（如列表、字符串）组成一个索引序列，
        #获得索引和元素值，start=1表示索引从1开始（默认为0）
        for iteration,indices in enumerate(fold, start=1):
            lr = LogisticRegression(C = c_param, penalty = 'l1')
            lr.fit(X_train.iloc[indices[0],:],Y_train.iloc[indices[0],:].values.ravel())
            Y_pred= lr.predict(X_train.iloc[indices[1],:].values)
            recall_acc = recall_score(Y_train.iloc[indices[1],:].values,Y_pred)
            recall_accs.append(recall_acc)
            #print ('Iteration:',iteration,'recall_acc:',recall_acc)
        #求每个C参数的平均recall值
        print ('Mean recall score',np.mean(recall_accs))
        results_table.loc[j,'Mean recall score'] = np.mean(recall_accs)
        j+=1
    # 最佳C参数
    # 千万注意results_table['Mean recall score']的类型是object，要转成float64！
    results_table['Mean recall score']=results_table['Mean recall score'].astype('float64')
    best_c = results_table['C_Parameter'].iloc[results_table['Mean recall score'].idxmax()]
    print ('best_c is :',best_c)
    return best_c


# In[120]:


#下采样数据得到的C参数


# In[122]:


best_c = printing_Kfold_scores(X_train_lowerdata,Y_train_lowerdata)


# In[ ]:


#画混淆矩阵


# In[123]:


import itertools
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",fontsize=15,color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)


# In[ ]:


#最优C参数应用于下采样数据集


# In[81]:


lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_lowerdata,Y_train_lowerdata.values.ravel())
Y_pred_lowerdata = lr.predict(X_test_lowerdata.values)
cnf_matrix = confusion_matrix(Y_test_lowerdata,Y_pred_lowerdata)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))
class_names = [0,1]
f,ax=plt.subplots(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


#最优C参数应用于原始数据集


# In[82]:


lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train,Y_train.values.ravel())
Y_pred= lr.predict(X_test.values)
cnf_matrix = confusion_matrix(Y_test,Y_pred)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))
class_names = [0,1]
f,ax=plt.subplots(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


#通过原始数据得到的C参数


# In[124]:


best_c = printing_Kfold_scores(X_train,Y_train)


# In[ ]:


#最优C参数应用于应用于下采样数据集


# In[84]:


lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_lowerdata,Y_train_lowerdata.values.ravel())
Y_pred_lowerdata = lr.predict(X_test_lowerdata.values)
cnf_matrix = confusion_matrix(Y_test_lowerdata,Y_pred_lowerdata)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))
class_names = [0,1]
f,ax=plt.subplots(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


#最优C参数应用于应用于原始数据集


# In[85]:


lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train,Y_train.values.ravel())
Y_pred= lr.predict(X_test.values)
cnf_matrix = confusion_matrix(Y_test,Y_pred)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))
class_names = [0,1]
f,ax=plt.subplots(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


#通过C=0.01,结合回归模型及下采样数据集求threshold参数


# In[86]:


lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_lowerdata,Y_train_lowerdata.values.ravel())
y_pred_lowerdata_proba = lr.predict_proba(X_test_lowerdata.values)
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.figure(figsize=(15,15))
recall_accs = []
j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_lowerdata_proba[:,1] > i
    plt.subplot(3,3,j)
    j += 1
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test_lowerdata,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    recall_acc = float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1])
    print('Threshold>=%s Recall: '%i, recall_acc)
    recall_accs.append(recall_acc)
    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix , classes=class_names , title='Threshold>=%s'%i)


# In[ ]:


#通过C=0.01,结合回归模型及原始数据集求threshold参数


# In[87]:


lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train,Y_train.values.ravel())
y_pred_proba = lr.predict_proba(X_test.values)
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plt.figure(figsize=(15,15))
recall_accs = []
j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_proba[:,1] > i
    plt.subplot(3,3,j)
    j += 1
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    recall_acc = float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1])
    print('Threshold>=%s Recall: '%i, recall_acc)
    recall_accs.append(recall_acc)
    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix , classes=class_names , title='Threshold>=%s'%i)


# In[ ]:


#注：看数据的结构，是否平衡；
#若不平衡，采取下采样或过采样获取全新的数据集，再选模型、算法；
#模型的调参不断的试，选最佳的参数；
#预测应综合考虑精度、recall、混淆矩阵等多个参数


