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
Xindex = []  #数据集X的索引（index）
# X= {Xp，Xm}
for i in range(M):
    if label[i].sum()==0:
        Xp.append(X[i])
        Xindex.append('Xp')
        p += 1
    else:
        Xm.append(X[i])
        Xindex.append('Xm')
        m += 1
print(p)
print(m)

Xpfilm = np.savetxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_Xp.csv',Xp, fmt='%0.4f', delimiter=',')
Xmfilm = np.savetxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_Xm.csv',Xm, fmt='%0.4f', delimiter=',')
Xp = np.asarray(Xp)
Xm = np.asarray(Xm)

Xcolumns = []  #数据集X的列名（columns）
for i in range(N):
    Xcolumns.append('x'+str(i+1))

X_frame = DataFrame(X,columns=Xcolumns,index=Xindex)
print(X_frame)


## 预填补 -- 均值填补
# 求每列均值
X_temp = X
for i in range(M):
    for j in range(N):
        if label[i,j] == 1:
            X_temp[i,j] = 0
Xcolumn_mean = np.round(X_temp.mean(axis=0),4)
print(Xcolumn_mean)

# 均值代替缺失值
for i in range(M):
    for j in range(N):
        if label[i,j] == 1:
            X[i,j] = Xcolumn_mean[j]
print(X_frame)

# 将 X_frame 写为csv文件
outputpath='C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_Xframe.csv'
X_frame.to_csv(outputpath,sep=',',index=True,header=True)

