import numpy as np
import copy

def delete(filename,del_filename,label_filename,rate ):
    flag = 1
    loop_count = 0
    while flag:
        flag = 0
        loop_count += 1
        data = np.loadtxt(filename, delimiter=',')
        data = data[:, 1:]
        M = data.shape[0]  # 输出data行数
        N = data.shape[1]  # 输出data列数
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

data_file = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris.csv'
data_del = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_del.csv'
data_label = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_label.csv'
data_meanpre = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_meanpre.csv'
data_filled = 'C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_filled.csv'

delete(data_file,data_del,data_label,0.1)

