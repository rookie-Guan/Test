import numpy as np
import copy
from 三层神经网络 import BPNN3layers
from 四层神经网络 import BPNN4layers
np.set_printoptions(threshold=10000,precision=4,suppress=True)

## 函数定义
def delete(data,del_filename,label_filename,rate ):

    M,N = data.shape[0],data.shape[1]  # data行数 data列数
    flag,loop_count= 1,0
    while flag:
        flag = 0
        loop_count += 1
        data_del = copy.deepcopy(data)

        range_set = np.arange(M * N)
        rand_num = np.random.choice(range_set, int(np.round(M * N * rate)), replace=False)
        """ np.random.choice():对整数或者一维数组（列表），不能超过一维，
                               默认可以重复抽样，要想不重复地抽样，需要设置replace参数为False
        """
        # range_set = [i for i in range(M*N)]
        # rand_num = random.sample(range_set,int(np.round(M*N*rate)))
        """ random.sample(set,s)从列表set中不重复抽样s个"""

        label = np.zeros((M, N))
        for i in rand_num:
            row = int(np.floor(i / N))
            column = i % N
            label[row, column] = 1

        for i in range(M):
            if N < 5:
                if label[i].sum() == N:
                    flag = 1
                    break
            else:
                if label[i].sum() >= np.ceil(0.6*N):
                    flag = 1
                    break
            for j in range(N):
                if label[i, j] == 1:
                    data_del[i, j] = np.nan

    print('数据集: ' + str(M) + ' X ' + str(N))
    print("缺失率:", rate, "\n缺失值个数:", int(label.sum()))
    print('循环次数: ' + str(loop_count))
    np.savetxt(del_filename, data_del, fmt='%0.4f', delimiter=',')
    np.savetxt(label_filename, label, fmt='%0.4f', delimiter=',')

    return data_del,label

def meanpre(data_del,label,data_meanpre):

    M = data_del.shape[0]  # 输出data行数
    N = data_del.shape[1]  # 输出data列数

    ## 预填补 -- 均值填补
    # 求每列均值
    X_temp = copy.deepcopy(data_del)
    for i in range(M):
        for j in range(N):
            if label[i,j] == 1:
                X_temp[i,j] = 0
    Xcolumn_mean = np.round(X_temp.mean(axis=0),2)   # np.round：四舍五入，保留小数点后两位 ； .mean(axis=0)：求列均值
    print('均值:',Xcolumn_mean)

    # 均值代替缺失值
    meanpre = copy.deepcopy(data_del)
    for i in range(M):
        for j in range(0,N):
            if np.isnan(meanpre[i][j]):
                meanpre[i][j] = Xcolumn_mean[j]
    np.savetxt(data_meanpre,meanpre, fmt='%0.2f', delimiter = ',')

    return Xcolumn_mean,meanpre

def normalization(data,normed_path):
    max = np.nanmax(data, axis=0)
    print("最大值:",max)
    data = data / max
    np.savetxt(normed_path,data, fmt='%0.4f', delimiter = ',')
    return max, data


## 数据导入以及保存路径
data_file_path = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Seed.csv'
del_path = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Seed_del.csv'
label_path = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Seed_label.csv'
meanpre_path = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Seed_meanpre.csv'
normed_path = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Seed_normed.csv'
filled_path = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Seed_filled.csv'

# 导入原始数据
data = np.loadtxt(data_file_path, delimiter=',')
data = data[:, 1:]
N = data.shape[0]  # 样本个数
M = data.shape[1]  # 属性个数
# 生成缺失值 和 标签
datadel,label = delete(data,del_path,label_path,rate=0.3)
# 均值填补 —— 生成每列均值 和 预填补后的数据集
data_mean,data_pre = meanpre(datadel,label,meanpre_path)
# 标准化 —— 生成每列最大值 和 标准化后的数据集（0-1之间）
data_max,data_normed = normalization(data_pre,normed_path)


# 打乱数据集样本顺序
shuffle_index = np.array([i for i in range(N)])# 用来统一打乱四个数据集的索引矩阵
np.random.shuffle(shuffle_index)
label = label[shuffle_index]
data_normed = data_normed[shuffle_index]


## 新建神经网络M个 M个属性
net= []
for i in range(M):
    neti = BPNN4layers(inodes=M-1,hnodes1=15,hnodes2=5,onodes=1,lr=0.02,mf=0.1)
    # neti = BPNN3layers(inodes=M - 1, hnodes=15, onodes=1, lr=0.09,mf=0.1)
    net.append(neti)

# 训练神经网络
delta = 1  # 迭代判断条件，记录每次迭代与上一次迭代的输出的差值，若差值小于阈值结束迭代
epoch = 0  # 记录迭代次数
Loss = []
# for e in range(2000):
while delta >= 5e-6:
    nan_count,delta = 0,0  # nan_count记录缺失值个数
    epoch += 1
    loss_matrix = []
    for i in range(M-1,-1,-1):
        target_set = data_normed[:,i]
        input_set = np.vstack((data_normed.T[0:i],data_normed.T[i+1:])).T
        for j in range(N):
            input,target = input_set[j],target_set[j]
            l,y = net[i].train(input,target)
            loss_matrix.append(l)
            # 若data_meanpre[i][j]为缺失值位置，动态更新缺失值
            if label[j][i] == 1:
                nan_count += 1
                delta += abs(data_normed[j][i] - y)
                data_normed[j][i] = y
    delta /= nan_count
    loss = np.sum(loss_matrix) / (N * M)
    Loss.append(loss)
    print('epoch=', epoch)
    print('delta:', float(delta))
    print('loss=', loss)
    print('+-' * 20)


# 填补完成，重新使样本顺序恢复原来顺序
sort_index = np.array(shuffle_index).argsort()
data_filled = data_normed[sort_index]
label = label[sort_index]
# 数据反标准化
data_filled = data_filled * data_max

# MAPE = 0
# k = 0
# for i in range(N):
#     for j in range(M):
#         if label[i][j] == 1:
#             k += 1
#             error = abs((data[i][j] - data_filled[i][j]) / (data[i][j]+0.1))
#             MAPE += error
#             print(k, i, j)
#             print(data[i][j], data_filled[i][j])
#             print(error)
#             print('+-' * 20)
# MAPE = MAPE / k
# print('MAPE = ',MAPE)

MAPE = np.zeros([N,M])

for i in range(N):
    for j in range(M):
        if label[i][j] == 1:
            error = abs((data[i][j] - data_filled[i][j]) / (data[i][j]+0.1))
            MAPE[i][j] = error

print(MAPE)

