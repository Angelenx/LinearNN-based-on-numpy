import numpy as np
import mynn_ver2 as nn
import matplotlib.pyplot as plt

def normallize_dataset(dataset:list,maxs=[]):
    # mins = [10000]*len(dataset[0])
    
    if len(maxs) == 0:
        maxs = [-1]*len(dataset[0])
    
    
        for i in range(len(dataset[0])):
            for j in range(len(dataset)):
                if dataset[j][i] > maxs[i]:
                    maxs[i] = dataset[j][i]
                # if dataset[j][i] < mins[i]:
                #     mins[i] = dataset[j][i]
                
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            dataset[i][j] /= maxs[j]
            
    return maxs
            

def load_dataset(train_path = "./iris_train.txt",test_path = "./iris_test.txt"):
    with open(train_path) as f:
        train_dataset = [[],[]]
        while True:
            str_line = f.readline()
            if len(str_line) == 0:
                break
            train_data = str_line[0:-2].split(' ')
            train_data = [float(it) for it in train_data]
            train_dataset[0].append(train_data[0:-1])
            train_dataset[1].append([train_data[-1]])
        f.close()
        
    with open(test_path) as f:
        test_dataset = [[],[]]
        while True:
            str_line = f.readline()
            if len(str_line) == 0:
                break
            test_data = str_line[0:-2].split(' ')
            test_data = [float(it) for it in test_data]
            test_dataset[0].append(test_data[0:-1])
            test_dataset[1].append([test_data[-1]])
        f.close()
    test_maxs = normallize_dataset(train_dataset[0]) #对每个样本进行归一化
    train_dataset[0] = np.asarray(train_dataset[0])
    train_dataset[1] = np.asarray(train_dataset[1])*0.01
    
    normallize_dataset(test_dataset[0],test_maxs) # 对每个样本进行归一化，使用和训练集一样的计算公式。
    test_dataset[0] = np.asarray(test_dataset[0])
    test_dataset[1] = np.asarray(test_dataset[1])*0.01
    
    
    return train_dataset,test_dataset

def get_acc(y_pre, label): 
    _, n_test_samples = label.shape 
    n_right = 0
    for i in range(n_test_samples):
        if(abs(y_pre[0][i] - label[0][i])<=0.08):
            n_right+=1
    
    return n_right/n_test_samples

class ann:
    def __init__(self):
        self.layer1 = nn.Linear(8, 64)
        
        self.layer3 = nn.Linear(64, 1)
    
    def forward(self,x:np.ndarray):
        # 输入维度应为(input_dim，batch)
        out = self.layer1(x)
        out = nn.sigmoid(out)

        out = self.layer3(out)
        return out
    
    def backward(self,y_pre,label,lr):
        delta3 = (y_pre - label)/label.shape[1]
        
        delta1 = (nn.d_sigmoid(self.layer1.last_output) * self.layer3.backward(delta3,lr))
        self.layer1.backward(delta1,lr)


def get_RMSE(y_pre, label):
    _, n_test_samples = label.shape 
    sum_of_MSE = np.mean((y_pre-label)*(y_pre-label))
    return np.sqrt(sum_of_MSE)

if __name__ == '__main__':
    net = ann()
    origin_train_dataset,origin_test_dataset = load_dataset("./concrete.txt","./concrete_test.txt")
    
    train_dataset = (origin_train_dataset[0],origin_train_dataset[1]) # 转成元组
    test_dataset = (origin_test_dataset[0],origin_test_dataset[1])
    
    train_data_getter = nn.data_getter(train_dataset)

    ACClst = []
    RMSElst = []
    for epoch in range(40000):
        while True:
            train_batch = train_data_getter.get_batch(6) # 从dataloader 中取一个batch
            if train_batch == None:
                break
            xs = train_batch[0].T
            labels = train_batch[1].T
            y_pres = net.forward(xs)
            net.backward(y_pre=y_pres,label=labels,lr=0.008)
        
        # acc 是测试集的
        xs_test = test_dataset[0].T
        labels = test_dataset[1].T
        y_pres_test = net.forward(xs_test)
        acc = get_acc(y_pres_test,labels)
        rmse = get_RMSE(y_pres_test,labels)
        ACClst.append(acc)
        RMSElst.append(rmse)
        if epoch % 500 == 0:
            
            print('epoch:%d finished, acc: %.4f , RMSE:%.4f'%(epoch+1,acc,rmse))
            
        if len(RMSElst)>2000:
                if np.max(RMSElst[-200:-1]) - np.min(RMSElst[-200:-1]) <= 1e-15 or acc>=0.85:
                    break
                
                
        if(len(ACClst)>=800000):
            ACClst = ACClst[-799999:-1]
            # if(len(RMSElst)>=800000):
            #     RMSElst = RMSElst[-799999:-1]
    
    plt.plot(np.arange(len(ACClst)),ACClst)
    plt.plot(np.arange(len(RMSElst)),RMSElst)
    plt.show()
    
        
        
            