import numpy as np

data = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_del.csv',delimiter=',')
# 最大值归一化 返回每列的最大值以及归一化后的数据
def normalization(data):
     max = np.nanmax(data,axis=0)
     data = data / max
     return max,data

_,new = normalization(data)
print(data)
print(new)