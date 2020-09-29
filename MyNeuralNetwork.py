import numpy as np

# 激活函数 sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 激活函数 sigmoid导数
def d_sigmoid(x):
    return  sigmoid(x) * (1 - sigmoid(x))

# 损失函数  mean_squared_error（均方误差）
def loss(y,t):
    return  0.5 * np.sum((y-t)**2)

# 归一化
def normalization(li):
    n_max = np.max(li)
    n_min = np.min(li)
    li_nor = ((li-n_min) / (n_max-n_min))
    return li_nor,n_max,n_min

# 反归一化
def re_normalization(li_nor,max,min):
    return li_nor * (max - min) + min


class MyNeuralNetwork:
    def __init__(self,nodes, learning_rate):
        # 设定层数layer_num、节点数nodes、学习率lr
        self.layer_num = len(nodes) # 网络层数
        self.nodes = nodes # 网络各层节点数
        self.lr = learning_rate # 学习率

        # 设定权重值和阈值
        self.W,self.b= [0],[0]

        self.Z,self.A = [0 for _ in range(self.layer_num)],[0 for _ in range(self.layer_num)]
        self.dZ,self.dA = [0 for _ in range(self.layer_num)],[0 for _ in range(self.layer_num)]
        self.dW,self.db = [0 for _ in range(self.layer_num)],[0 for _ in range(self.layer_num)]

        # 参数初始化
        for i in range(1,self.layer_num):
            self.W.append(np.random.normal(0.0, pow(self.nodes[i], -0.5), (self.nodes[i], self.nodes[i-1])))
            self.b.append(np.random.normal(0.0, 1.0, (self.nodes[i], 1)))
        pass

    # 训练
    def train(self, inputs_org, target_org):
        # 将输入转化为2d矩阵，输入向量的shape为[feature_dimension,1]
        inputs = np.array(inputs_org, ndmin=2).T
        target = np.array(target_org, ndmin=2).T

        # repeat:
        # 　for all(xk, yk)∈D do
        # 　　 1、根据当前参数计算当前样本的输出
        # 　　 2、计算输出层神经元的梯度项；
        # 　　 3、计算隐层神经元的梯度项；
        # 　　 4、更新连接权与阈值
        # 　end

        # 前向传播
        # 输入层
        self.A[0]= inputs
        # 中间隐藏层
        for i in range(1,self.layer_num-1):
            self.Z[i] = np.dot(self.W[i],self.A[i-1]) + self.b[i]
            self.A[i] = sigmoid(self.Z[i])
        #         # 最后一层输出层
        self.Z[-1] = np.dot(self.W[-1],self.A[-2]) + self.b[-1]
        self.A[-1] = self.Z[-1]

        # 反向传播
        loss_error = loss(self.A[-1],target)
        error = self.A[-1] - target
        # 最后一层
        self.dA[-1] = error
        self.dZ[-1] = self.dA[-1]
        self.dW[-1] = np.dot(self.dZ[-1], self.A[-2].T)
        self.db[-1] = self.dZ[-1]
        self.dA[-2] = np.dot(self.W[-1].T, self.dZ[-1])

        self.W[-1] -= self.lr * self.dW[-1]
        self.b[-1] -= self.lr * self.db[-1]

        # 倒数第2层--第2层（隐层）
        for i in range(self.layer_num - 2, 0, -1):
            self.dZ[i] = d_sigmoid(self.Z[i]) * self.dA[i]
            self.dW[i] = np.dot(self.dZ[i], self.A[i - 1].T)
            self.db[i] = self.dZ[i]
            self.dA[i-1] = np.dot(self.W[i].T, self.dZ[i])

            # 更新参数
            self.W[i] -= self.lr * self.dW[i]
            self.b[i] -= self.lr * self.db[i]
        pass

    def query(self,inputs_org):
        # 将输入转化为2d矩阵，输入向量的shape为[feature_dimension,1]
        inputs = np.array(inputs_org, ndmin=2).T
        # 前向传播
        # 输入层
        self.A[0] = inputs
        # 中间隐藏层
        for i in range(1, self.layer_num - 1):
            self.Z[i] = np.dot(self.W[i], self.A[i - 1]) + self.b[i]
            self.A[i] = sigmoid(self.Z[i])
        #         # 最后一层输出层
        self.Z[-1] = np.dot(self.W[-1], self.A[-2]) + self.b[-1]
        self.A[-1] = self.Z[-1]
        outputs = self.A[-1]
        return outputs


nodes = [4,15,4]
learning_rate = 0.1
net = MyNeuralNetwork(nodes,learning_rate)

# load the mnist training data CSV file into a list
data_file = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris.txt',delimiter=',')[:, 1:]
training_data = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_Xp.csv',delimiter=',')

print(training_data.shape)

# normaliza_res = normalization(training_data_file)
# training_data = normaliza_res[0]
# training_data_max, training_data_min = normaliza_res[1],normaliza_res[2]
# go through all records in the training data set
for i in range(1000):

    for record in training_data:
        inputs = record
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = record
        # all_values[0] is the target label for this record
        net.train(inputs, targets)


test_data = np.loadtxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_X_meanpre.csv',delimiter=',')

error = 0
m = 0
res = []
for record in test_data:
    if record[0] == 0:
        m += 1
        record_arr = np.array(record[1:],ndmin=2).T
        y = np.array(net.query(record[1:]))

        res.append(y.reshape(4))
        error += loss(y,record_arr)
    else:
        res.append(record[1:])

print(m)
error /= m
print(error)

np.savetxt('C:\\Users\\user\\Desktop\\UCI数据集txt格式\\Iris_X_done.csv', res , fmt='%0.2f', delimiter = ',')









