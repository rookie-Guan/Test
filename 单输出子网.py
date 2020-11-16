from 三层神经网络 import BPNN3layers
import numpy as np
import matplotlib.pyplot as plt

M_MAPE = []
for E in range(5):
    # 导入数据集
    data = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris.csv', delimiter=',')
    data = data[:, 1:]   # 完整的原始数据集
    data_label = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_label.csv', delimiter=',')  # 不完整数据集标签集，不缺失的位置标 0 ，缺失的位置标 1
    data_meanpre = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_X_meanpre.csv', delimiter=',')
    data_meanpre = data_meanpre[:, 1:]  # 用均值填补预填补后的数据集 第一列为该行样本是否完整的标记，完整标1，不完整标0


    N = data.shape[0] # 样本个数
    M = data.shape[1] # 属性个数

    # 打乱数据集样本顺序
    shuffle_index = np.array([i for i in range(N)])# 用来统一打乱四个数据集的索引矩阵
    np.random.shuffle(shuffle_index)
    data_label = data_label[shuffle_index]
    data_meanpre = data_meanpre[shuffle_index]


    # 新建神经网络M个 M个属性
    net= []
    for i in range(M):
        neti = BPNN3layers(inodes=M-1,hnodes=5,onodes=1,lr=0.01)
        net.append(neti)

    # 训练神经网络
    delta = 1  # 迭代判断条件，记录每次迭代与上一次迭代的输出的差值，若差值小于阈值结束迭代
    epoch = 0  # 记录迭代次数
    Loss = []
    # for e in range(2000):
    while delta >= 1e-6:
        nan_count,delta = 0,0  # nan_count记录缺失值个数
        epoch += 1
        loss_matrix = []
        for i in range(M):
            target_set = data_meanpre[:,i]
            input_set = np.vstack((data_meanpre.T[0:i],data_meanpre.T[i+1:])).T
            for j in range(N):
                input,target = input_set[j],target_set[j]
                l,y = net[i].train(input,target)
                loss_matrix.append(l)
                # 若data_meanpre[i][j]为缺失值位置，动态更新缺失值
                if data_label[j][i]==1:
                    nan_count += 1
                    delta += abs(data_meanpre[j][i]-y)
                    data_meanpre[j][i] = y
        delta /= nan_count
        loss = np.sum(loss_matrix)/(N*M)
        Loss.append(loss)
        print('E = ',E)
        print('epoch=',epoch)
        print('delta:',delta)
        print('loss=',loss)
        print('+-'*20)

    # 填补完成，重新使样本顺序恢复原来顺序
    sort_index = np.array(shuffle_index).argsort()
    data_filled = data_meanpre[sort_index]
    data_label = data_label[sort_index]
    np.savetxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_filled.csv',data_filled,fmt='%0.2f', delimiter = ',')

    # 计算填补精度
    MAPE = 0
    k = 0
    for i in range(N):
        for j in range(M):
            if data_label[i][j]==1:
                k += 1
                error = abs((data[i][j] - data_filled[i][j]) / data[i][j])
                MAPE += error
                print(k,i,j)
                print(data[i][j],data_filled[i][j])
                print(error)
                print('+-'*20)
    MAPE = MAPE / k
    # print('MAPE = ',MAPE)
    M_MAPE.append(MAPE)

print(k)
print(M_MAPE)
print(np.mean(M_MAPE))




