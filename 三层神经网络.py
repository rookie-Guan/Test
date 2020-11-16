import numpy as np

# 激活函数 sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 激活函数 sigmoid导数
def d_sigmoid(x):
    return  sigmoid(x) * (1 - sigmoid(x))

class BPNN3layers():
    def __init__(self,inodes,hnodes,onodes,lr=0.01,mf=0.1):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.lr = lr
        self.mf = mf

        self.w1 = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.inodes, self.hnodes))
        self.b1 = np.random.normal(0.0, 1.0, (1,self.hnodes))
        self.w2 = np.random.normal(0.0, pow(self.onodes, -0.5), (self.hnodes, self.onodes))
        self.b2 = np.random.normal(0.0, 1.0, (1,self.onodes))
        self.dw1 = np.zeros((self.inodes, self.hnodes))
        self.db1 = np.zeros((1,self.hnodes))
        self.dw2 = np.zeros((self.hnodes, self.onodes))
        self.db2 = np.zeros((1,self.onodes))

    def train(self,input,target):

        input = np.array(input, ndmin=2)
        target = np.array(target, ndmin=2)

        # 前向传播
        h_input = np.dot(input,self.w1) + self.b1 # Z2
        h_output = sigmoid(h_input) # A2=sigmoid(Z2) 隐层激活函数为sigmoid函数
        output = np.dot(h_output,self.w2) + self.b2  # Z3=A3 输出层激活函数为线性函数

        # 损失函数
        loss = 0.5 * np.sum((output - target)**2)

        # 反向传播
        error = output - target
        h_error = np.dot(error,self.w2.T) * d_sigmoid(h_input)

        # 参数更新方式 1: 不含动量，只有偏导
        # self.w2 -=  self.lr * np.dot(h_output.T,error)
        # self.b2 -= self.lr * error
        # self.w1 -= self.lr * np.dot(input.T,h_error)
        # self.b1 -= self.lr * h_error

        # 参数更新方式 2：含有动量
        w2new = self.w2 - self.lr * np.dot(h_output.T,error) + self.mf * self.dw2
        self.dw2 = w2new - self.w2
        self.w2 = w2new

        b2new = self.b2 - self.lr * error + self.mf * self.db2
        self.db2 = b2new -self.b2
        self.b2 = b2new

        w1new = self.w1 - self.lr * np.dot(input.T,h_error) + self.mf * self.dw1
        self.dw1= w1new - self.w1
        self.w1 = w1new

        b1new = self.b1 - self.lr * h_error + self.mf * self.db1
        self.db1 = b1new - self.b1
        self.b1 = b1new

        return loss,output


    def test(self,input):

        input = np.array(input, ndmin=2)

        # 前向传播
        h_input = np.dot(input, self.w1) + self.b1  # Z2
        h_output = sigmoid(h_input)  # A2=sigmoid(Z2) 隐层激活函数为sigmoid函数
        output = np.dot(h_output, self.w2) + self.b2  # Z3=A3 输出层激活函数为线性函数

        return  output
