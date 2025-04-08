import numpy as np
import os

PATH = os.path.split(os.path.realpath(__file__))[0]
os.chdir(PATH)
print('Changed cwd to :',os.getcwd())

class data_getter:
    def __init__(self,dataset):
        order = np.arange(dataset[0].shape[0])
        np.random.shuffle(order)
        self.xs = dataset[0][order]
        self.ys = dataset[1][order]
        self.index_st = 0
        
    def get_batch(self,batch_size):
        if self.index_st >= self.xs.shape[0]:
            self.index_st = 0
            return None
        
        if self.index_st+batch_size >= self.xs.shape[0]: 
            # 如果继续按batch_size取会超出数据集范围，则取从当前索引开始到数据集最后一个样本
            out = (self.xs[self.index_st:-1],
               self.ys[self.index_st:-1])
            self.index_st += batch_size
            return out
            
          
        out = (self.xs[self.index_st:self.index_st+batch_size],
               self.ys[self.index_st:self.index_st+batch_size])
        self.index_st += batch_size
        return out
    
    
        
    
    
        
    

class Linear:
    def __init__(self, input_dim, output_dim):
        self.parameters = dict()
        self.parameters['w'] = np.random.normal(size=(output_dim, input_dim))
        self.parameters['b'] = np.random.normal(size=(output_dim, 1))

        self._dZ_d=dict()
        self._dZ_d['w'] = np.zeros((output_dim, input_dim))
        self._dZ_d['b'] = np.zeros((output_dim, 1))

    def __call__(self, x: np.ndarray):
        # x.shape should be (input_dim, batch)
        self.last_output = np.dot(self.parameters['w'], x) + self.parameters['b']
        self.last_input = x
        return self.last_output

    def backward(self, delta_ip1,lr):
        self.parameters['w'] -= lr * np.dot(delta_ip1, self.last_input.T)
        self.parameters['b'] -= lr * np.sum(delta_ip1, 1, keepdims=True)
        return np.dot( self.parameters['w'].T, delta_ip1 ) / self.last_output.shape[1]





def to_one_hot(labels):
    rlabels = np.zeros((labels.shape[0], 10))
    rlabels[np.arange(labels.shape[0]), labels-1] = 1.
    return rlabels

def sigmoid(x):
    return 1/(1+np.exp(-x))


def d_sigmoid(x):
    m = sigmoid(x)
    return m*(1-m)  # 这里的乘法是点乘或按元素乘法


def softmax(x):
    epsilon = 1e-12 # 防止分母为0
    x_exp = np.exp(x)
    partition = np.sum(x_exp, axis=0, keepdims=True)  # 以横轴（axis=0）为基准
    out = x_exp / (partition + epsilon)  # 点除运算，x.exp中第i行的元素除以partition中的第i行元素
    return out



if __name__ == '__main__':
    l1 = Linear(3, 2)
    x = np.random.rand(3, 16) # (input_dim, batch)
    y = l1(x)

    print(y.shape)