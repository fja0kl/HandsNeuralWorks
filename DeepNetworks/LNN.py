#coding:utf8
import warnings
import matplotlib.pyplot as plt
from utils import *

warnings.filterwarnings('ignore')

class LNNModle(object):

    def __init__(self):
        pass

    def __sigmoid(self, Z):
        cache = Z
        A = 1./(1+np.exp(-Z))
        return A, cache

    def __sigmoid_backward(self, dA, cache):
        """
        sigmoid激活函数的反向传播
        :param dA: loss对A的偏导
        :param cache: 本层前向传播用到的参数
        :return:dZ
        """
        Z = cache
        A = 1./(1+np.exp(-Z))
        dZ = dA * A * (1-A)

        return dZ

    def __relu(self, Z):
        """
        relu激活函数
        :param Z:
        :return:
        - A：激活值
        - cache：用到的参数
        """
        A = np.maximum(Z, 0)
        cache = Z

        return A, cache

    def __relu_backward(self, dA, cache):
        """
        relu反向传播
        :param dA:
        :param cache:
        :return: dZ
        """
        Z = cache
        dZ = np.array(dA,copy=True)

        dZ[Z <= 0] = 0
        assert (dZ.shape == dA.shape)

        return dZ

    def __initialize_parameters(self, layers_dim, type='he'):
        """
        权重系数初始化
        :param layers_dim: 网络层维度定义
        :param type: 系数初始化方法: zeros, random, he
        :return:
        params: 系数字典
        """
        params = {}
        L = len(layers_dim)#一共有L-1层

        if type == 'zeros':
            for i in range(1, L):  # L-1层网络
                params['W' + str(i)] = np.zeros((layers_dim[i], layers_dim[i - 1]))
                params['b' + str(i)] = np.zeros((layers_dim[i], 1))

                assert (params['W' + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
                assert (params['b' + str(i)].shape == (layers_dims[i], 1))
        elif type == 'random':
            for i in range(1, L): # L-1层网络
                params['W'+str(i)] = np.random.randn(layers_dim[i], layers_dim[i-1]) * 0.01
                params['b'+str(i)] = np.zeros((layers_dim[i], 1))

                assert (params['W' + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
                assert (params['b' + str(i)].shape == (layers_dims[i], 1))
        elif type == 'he':
            for i in range(1, L):
                params['W'+str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) / np.sqrt(layers_dims[i-1])
                params['b'+str(i)] = np.zeros((layers_dims[i], 1))

                assert (params['W' + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
                assert (params['b' + str(i)].shape == (layers_dims[i], 1))

        return params

    def __linear_forward(self, A_prev, W, b):
        """
        线性运算
        :param A_prev: [n_(l-1), m];m样本数
        :param W: [n_l，n_(l-1)]
        :param b: [n_l, 1]
        :return:
        - Z：运算结果
        - cache：用到的参数（A_prev, W, b）
        """
        Z = np.dot(W, A_prev) + b
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))
        cache = (A_prev, W, b)

        return Z, cache

    def __linear_activation_forward(self, A_prev, W, b, activation):
        """
        一层网络层运算过程：线性部分+激活层
        :param A_prev: [n_l-1, m]
        :param W: [n_l, n_l-1]
        :param b: [n_l, 1]
        :param activation: 'sigmoid', 'relu'
        :return:
        - A: 本层网络的激活函数值
        - cache: 用到的参数(linear_cache, activation_cache)
        """
        Z, linear_cache = self.__linear_forward(A_prev, W, b)
        if activation == 'sigmoid':
            A, activation_cache = self.__sigmoid(Z)
        elif activation == 'relu':
            A, activation_cache = self.__relu(Z)
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def __model_forward(self, X, params):
        """
        网络的线性传播部分
        :param X:
        :param params:
        :return:
        - AL: 激活值
        - caches: 各层的缓冲部分，方便进行反向传播
        """
        caches = []
        L = len(params) // 2 # 网络长度
        A = X

        # (L-1) 层 激活函数为 relu；最后一层L激活函数为sigmoid--->分开计算
        for i in range(1, L):#左闭右开
            A_prev = A
            A, cache = self.__linear_activation_forward(A_prev, params['W'+str(i)], params['b'+str(i)],\
                                                       activation='relu')
            caches.append(cache)
        AL, cache = self.__linear_activation_forward(A, params['W'+str(L)], params['b'+str(L)],\
                                                    activation='sigmoid')
        caches.append(cache)
        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def __compute_cost(self, AL, y):
        """
        计算成本函数
        :param AL: 预测值
        :param y: 标签值
        :return: cost成本函数值；成本函数为交叉熵函数
        """
        m = y.shape[1]
        cost = -1/m * (np.dot(y, np.log(AL).T) + np.dot(1-y, np.log(1-AL).T))
        cost = np.squeeze(cost) # 确保cost是一个实数;标量

        assert (cost.shape == ())

        return cost

    def __linear_backward(self, dZ, cache):
        """
        线性部分的反向传播；Z_l = W_l*A_(l-1) + b_l
        :param dZ: [n_l, m]
        :param cache: (A_prev, w, b)
        :return:
        - dA_prev: [n_prev, m]
        - dW: [n_l, n_prev]
        - db: [n_l, 1]
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dA_prev = np.dot(W.T, dZ)
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1,keepdims=True)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def __linear_activation_backward(self, dA, cache, activation):
        """
        一层网络层的反向传播计算过程: linear + activation
        :param dA: [n_l, m]
        :param cache: (linear_cache, activation_cache)
        :param activation: 激活函数类型 relu,sigmoid
        :return:
        - dA_prev: [n_prev, m]
        - dW: [n_l, n_prev]
        - db: [n_l, 1]
        """
        linear_cache, activation_cache = cache

        if activation == 'sigmoid':
            dZ = self.__sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        elif activation == 'relu':
            dZ = self.__relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def __model_backward(self, AL, y, caches):
        """
        神经网络的反向传播过程
        :param AL: 测试样本的输出值 [n_y, m]
        :param y: 测试样本的标签值 [n_y, m]
        :param caches: 各个网络层的cache 值
        :return: grads 各个网络层参数的梯度值
        """
        grads = {}
        L = len(caches)
        m = y.shape[1]
        y = y.reshape(AL.shape)  # 确保AL和y shape相同；
        # 成本函数[交叉熵]的关于AL的导数
        dAL = -(np.divide(y, AL) - np.divide(1-y, 1-AL))

        #最后一层 单独计算--激活函数不同
        current_cache = caches[-1]
        grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = self.__linear_activation_backward(dAL,current_cache,activation='sigmoid')

        # print("dW2.shape：", grads['dW2'].shape)
        for i in reversed(range(L-1)):
            current_cache = caches[i]
            grads['dA'+str(i+1)], grads['dW'+str(i+1)], grads['db'+str(i+1)] = \
                self.__linear_activation_backward(grads['dA'+str(i+2)],current_cache,activation='relu')

        return grads

    def __update_parameters_with_gd(self,params,grads,learning_rate):
        """
        梯度更新算法
        :param params: 系数字典 w,b
        :param grads: 梯度字典 dw,db
        :param learning_rate: 学习率
        :return:
        params: 更新后的系数字典
        """
        L = len(params) // 2 #网络长度

        for i in range(L):#使用梯度更新每层的W和b参数
            # print("{},{}:{}".format(i,str('W'+str(i+1)), params['W'+str(i+1)].shape))
            # print("{},{}:{}".format(i, 'dW' + str(i + 1),grads['dW' + str(i + 1)].shape))
            params['W'+str(i+1)] = params['W'+str(i+1)] - learning_rate*grads['dW'+str(i+1)]
            params['b'+str(i+1)] = params['b'+str(i+1)] - learning_rate*grads['db'+str(i+1)]


        return params

    def model(self,X,y,layers_dim,num_iters=1000,learning_rate=0.001,print_cost=True):
        """
        整合到一起，网络模型
        :param X: 训练数据
        :param y: 训练数据标签
        :param layers_dim: 网络维度定义
        :param num_iters: 迭代次数
        :param learning_rate: 学习率
        :param print_cost: 是否打印输出
        :return:
        params: 训练后的系数字典
        costs: 成本函数值列表
        """
        np.random.seed(10)
        costs = []
        #layer_dims = self.layers_dims
        params = self.__initialize_parameters(layers_dim)
        for i in range(num_iters):
            AL, caches = self.__model_forward(X, params)
            # print("W1.shape:", caches[0][0][1].shape)
            grads = self.__model_backward(AL, y, caches)

            cost = self.__compute_cost(AL, y)
            params = self.__update_parameters_with_gd(params, grads,learning_rate)

            if i % 100 == 0:
                costs.append(cost)
            if print_cost and i%100 == 0:
                print("Cost after iteration {}:{:.3f}".format(i, cost))

        return params, cost

    def score(self, params, X, y):
        """
        由测试集判断训练模型的好坏
        :param params: 训练得到的参数
        :param X: 测试集 [n_px*n_px*3, m]
        :param y: 测试集标签 [1, m]
        :return: accuracy 准确率
        """
        m = X.shape[1]
        result = np.zeros((1, m))

        probs, _ = self.__model_forward(X, params)
        L = len(params) // 2

        for i in range(probs.shape[1]):
            if probs[0, i] >= 0.5:
                result[0, i] = 1

        accuracy = np.mean(result == y)

        return accuracy

    def predict(self, params, X):
        """
        给定图片进行测试，输出预测标签
        :param params: 训练的参数
        :param X: 待预测数据
        :return: 预测结果
        """
        preds = np.zeros((1,X.shape[1]))
        probs, _ = self.__model_forward(X,params)

        for i in range(X.shape[1]):
            if probs[0, i] >= 0.5:
                preds[0, i] = 1

        preds = np.squeeze(preds)

        return preds

if __name__ == '__main__':
    # 加载数据集
    train_X_orig, train_Y, test_X_orig, test_Y, classes = load_data()
    print(train_X_orig.shape,train_Y.shape)
    print(test_X_orig.shape, test_Y.shape)
    # flatten the pictures
    train_X = train_X_orig.reshape(train_X_orig.shape[0], -1).T
    test_X = test_X_orig.reshape(test_X_orig.shape[0], -1).T
    print(train_X.shape, test_X.shape)
    # 数据预处理--缩放
    train_X = train_X / 255
    test_X = test_X / 255

    layers_dims = [12288, 100, 20, 1]
    model = LNNModle()
    parameters,_ = model.model(train_X, train_Y, layers_dims, num_iters=1000,learning_rate=0.001, print_cost=True)
    results = model.score(parameters,test_X,test_Y)
    print(results)
    index = 20
    img = test_X[:,index]
    img = img.reshape(test_X.shape[0], -1)
    print(img.shape)
    label = test_Y[0, index]
    plabel = model.predict(parameters, img)
    print(plabel, label)

    print_mislabeled_images(classes,test_X,test_Y,results)