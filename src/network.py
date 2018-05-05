#coding:utf8
import random

import numpy as np

def sigmoid(z):
	"""
	激活函数；
	"""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""sigmoid激活函数的导数"""
	return sigmoid(z)*(1-sigmoid(z))

class Network(object):
	"""
	神经网络：
	-初始化；
	-前向传播；
	-SGD：反向传播---求梯度；
	-评估
	"""	
	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		# y，x：y是后面一层，x是前面一层；；；以后面的神经元为目标点；
		#左闭右开
		
	def feedforward(self, a):
		"""
		前向传播，计算下一层；
		"""
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a)+b)
		return a
	
	def SGD(self,training_data,epochs,mini_batch_size,eta,
			test_data=None):
		"""
		随机梯度下降：梯度计算---反向传播；
		"""
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			#将训练数据分成若干个mini batch；
			mini_batches = [training_data[k:k+mini_batch_size] \
						for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if test_data:
				print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
			else:
				print("Epoch {0} complete".format(j))
	
	def update_mini_batch(self,mini_batch,eta):
		nable_b = [np.zeros(b.shape) for b in self.biases]
		nable_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			delta_nable_b,delta_nable_w = self.backprop(x,y)
			nable_b = [nb+dnb for nb,dnb in zip(nable_b,delta_nable_b)]
			nable_w = [nw+dnw for nw,dnw in zip(nable_w,delta_nable_w)]
		self.weights = [w-(eta/len(mini_batch))*nw \
						for w,nw in zip(self.weights,nable_w)]
		self.biases = [b-(eta/len(mini_batch))*nb \
						for b,nb in zip(self.biases,nable_b)]

	def backprop(self,x,y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
		"""
		nable_b = [np.zeros(b.shape) for b in self.biases] # gradiant
		nable_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x] # list to store all the activations,layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activation[-1],y) * sigmoid_prime(zs[-1])
		nable_b[-1] = delta # 损失函数对b偏导数----最后一层；
		nable_w[-1] = np.dot(delta,activations[-2].transpose()) # 损失函数对w权重的偏导数----最后一层；
		for l in range(2,self.num_layers):#反向传播
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # 更新前一层的delta，error
			nable_b[-l] = delta
			nable_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nable_b, nable_w)
	
	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
		return sum(int(x == y) for (x,y) in test_results)
	
	def cost_derivative(self,output_activations,y):
		"""平方差损失函数：对a的导数；
		"""
		return (output_activations-y)
	
	
	