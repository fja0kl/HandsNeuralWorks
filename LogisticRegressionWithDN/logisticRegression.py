import h5py
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    加载数据集
    :return: 训练集、标签、测试集、标签、类别信息；
    """
    train_dataset = h5py.File("datasets/train_catvnoncat.h5","r")
    train_x_orig = np.array(train_dataset['train_set_x'][:])
    train_y_orig = np.array(train_dataset['train_set_y'][:]) # shape：(209,)-->不是列向量---向量化,or bugs
    # for key in train_dataset.keys():
    #     print(key)

    test_dataset = h5py.File("datasets/test_catvnoncat.h5","r")
    test_x_orig = np.array(test_dataset['test_set_x'][:])
    test_y_orig = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:]) # 类别信息

    train_y_orig = train_y_orig.reshape((1,train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1,test_y_orig.shape[0]))
    return train_x_orig,train_y_orig, test_x_orig, test_y_orig,classes

def sigmoid(Z):
    """
    sigmoid激活函数;
    :param Z:
    :return:
    - A: 激活函数值sigmoid(z),
    """
    A = 1.0/(1+np.exp(-Z))

    return A

def initialize_with_zeros(dim):
    """
    单层神经网络：只有一个输出层
    :param dim: 输入数据层的维度
    :return:
    - w:权重系数矩阵
    - b:标量
    """
    w = np.zeros((dim, 1))
    b = 0

    return w, b

def propagate(w, b, X, y):
    """
    单层神经网路的前向传播过程和反向传播过程;使用for-loop循环简单易懂--牺牲效率
    :param w: 权重系数矩阵，(n_px * n_px * 3, 1)
    :param b: 偏置，标量
    :param X: 训练集,size （n_px * n_px * 3, m）
    :param y: 训练集标签，size (1, m)
    :return:
    - cost 成本函数值
    - grads 梯度计算值
    """
    m = X.shape[1]
    # 前向传播过程
    Z = np.dot(w.T, X) + b # 计算线性部分
    A = sigmoid(Z) # 计算激活函数

    assert (w.shape == (X.shape[0], 1))
    cost = - 1/m * np.sum(y*np.log(A) + (1-y)*np.log(1-A))
    # Z = np.zeros((1, m))
    # A = np.zeros((1, m))
    # cost = 0 # 成本函数值
    #
    # for i in range(m): # 遍历每个样本，计算loss函数值
    #     Z[0, i] = w.T * X[:, i]
    #     A[0, i] = sigmoid(Z[0, i])
    #     cost_tmp = -y[0,i]*np.log(A[0, i]) -(1-y[0, i])*np.log(1-A[0, i]) # 单个样本的loss函数值
    #     cost += cost_tmp
    # cost = cost / m # 取平均值；得到cost成本函数

    ## 反向传播过程
    dw = 1/m * np.dot(X, (A-y).T)
    db = 1/m * np.sum(A-y)

    grads = {'dw': dw,
             'db': db}
    return grads, cost

def optimize(w,b,X,y,num_iters,learning_rate,print_cost=True):
    """
    参数优化过程
    :param w: 系数矩阵
    :param b: 偏置
    :param X: 测试集
    :param y: 测试集标签
    :param num_iters: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 是否打印输出cost变化;每100次打印输出一次
    :return:
    - params: 更新后的参数
    - grads: 梯度计算值
    - costs：cost变化过程；每100次为一个记录值
    """
    costs = []

    for i in range(num_iters):
        grads, cost = propagate(w, b, X, y)
        dw = grads['dw']
        db = grads['db']
        #参数更新
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:#添加到costs
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}:{}".format(i, cost))

        params = {'w': w,
                  'b': b}

    return params, grads, costs

def predict(w,b,X):
    """
    给定一张图片预测分类标签
    :param w: 训练后的权重w参数 (n_px * n_px * 3, 1)
    :param b: 训练后的偏置b参数
    :param X: 测试图片 (n_px * n_px * 3, m)
    :return: 分类标签yHat
    """
    m = X.shape[1]
    yHat = np.zeros((1, m))
    assert (w.shape == (X.shape[0], 1))
    yHat = sigmoid(np.dot(w.T, X) + b) # 前向传播过程

    # 确定预测的分类标签 threshold为0.5
    for i in range(m):
        if yHat[0, i] > 0.5:
            yHat[0, i] = 1
        else:
            yHat[0, i] = 0

    return yHat

def model(X_train, y_train, X_test, y_test, num_iters=2000, learning_rate=0.05, print_cost=True):
    """
    将所有的函数整合到一起形成一个完整的模型
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param num_iters: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 是否打印输出cost成本函数值
    :return:
    - d: 模型信息字典
    """
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, y_train, num_iters, learning_rate, print_cost)

    w = params['w']
    b = params['b']

    yHat_train = predict(w, b, X_train)
    yHat_test = predict(w, b, X_test)

    print("Accuracy on Training set:{:.2f}%".format(100*np.mean(y_train == yHat_train)))
    print("Accuracy on Test set:{:.2f}%".format(100*np.mean(y_test == yHat_test)))

    d = {
        'costs': costs,
        'yHat_train': yHat_train,
        'yHat_test': yHat_test,
        'w': w,
        'b': b,
        'learning_rate': learning_rate,
        'num_iters': num_iters
    }

    return d

if __name__ == '__main__':
    X_train_org, y_train, X_test_org, y_test, classes = load_data()
    X_train = X_train_org.reshape(X_train_org.shape[0], -1).T
    X_test = X_test_org.reshape(X_test_org.shape[0], -1).T

    X_train = X_train/255
    X_test = X_test/255

    print(X_train.shape, y_train.shape)
    d = model(X_train,y_train,X_test,y_test,num_iters=500,learning_rate=0.001)

    print(d['w'])
    print(d['b'])







