import numpy as np
np.set_printoptions(threshold=10000,precision=4,suppress=True)

def normalization(data):
    max = np.nanmax(data, axis=0)
    data = data / max
    return max, data

data = np.loadtxt("C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Seed.csv",delimiter=',')
data = data[:,1:]
max , data = normalization(data)
corr = abs(np.corrcoef(data.T))
var = np.std(data,axis=0,ddof=1)
print(corr)
print("标准差:")
print(var)
# res = []
# for i in range(4):
#     tmp = (np.sum(corr[i]) - 1)/3
#     res.append(tmp)
# res = np.round(res,4)
# print(res)