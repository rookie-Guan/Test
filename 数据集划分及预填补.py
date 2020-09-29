import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import copy


## 导入数据
X = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_del.csv', delimiter=',')
label = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_label.csv', delimiter=',')
M = X.shape[0]  # 输出data行数
N = X.shape[1]  # 输出data列数
p=0
m=0
Xp = []
Xm = []
Xindex = []  #数据集X的索引（index），用来记录对应样本是否位完整样本

# X= {Xp，Xm}
for i in range(M):
    if label[i].sum()==0:
        Xp.append(X[i])
        Xindex.append(1)  # 完整样本就往标签数组中添加1
        p += 1
    else:
        Xm.append(X[i])
        Xindex.append(0)  # 不完整样本就标记0
        m += 1
print('完整样本个数:',p)
print('不完整样本个数:',m)

# 在数据集第 0 列添加样本是否完整的标签，1为完整样本，0为不完整样本
X = np.insert(X,0,Xindex,axis=1)
print(X)


Xpfilm = np.savetxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_Xp.csv',Xp, fmt='%0.4f', delimiter=',')
Xmfilm = np.savetxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_Xm.csv',Xm, fmt='%0.4f', delimiter=',')
Xp = np.asarray(Xp)
Xm = np.asarray(Xm)

## 预填补 -- 均值填补
# 求每列均值
X_temp = copy.deepcopy(X[:,1:])
print(type(X_temp))
print(type(X))
print(np.isnan(X_temp[3][1]))
print(np.isnan(X[3][2]))
for i in range(M):
    for j in range(N):
        if label[i,j] == 1:
            X_temp[i,j] = 0
Xcolumn_mean = np.round(X_temp.mean(axis=0),2)
print('均值:',Xcolumn_mean)

# 均值代替缺失值
queshizhi = 0
for i in range(M):
    for j in range(1,N+1):
        if np.isnan(X[i][j]):
            queshizhi += 1
            X[i][j] = Xcolumn_mean[j-1]

print(queshizhi)
print(X)

np.savetxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_X_meanpre.csv', X , fmt='%0.2f', delimiter = ',')
