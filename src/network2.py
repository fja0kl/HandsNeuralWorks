#coding:utf8
"""network2.py
~~~~~~~~~~~~~~
改进版
包括：
- 新的损失函数：交叉熵；
- 正则化 L2；
- 权重、偏置系数初始化；使用高斯分布
"""
import json
import random
import sys

import numpy as np


class CrossEntropyCost(object):
	@staticmethod
	def fn(a,y):
		return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

	@staticmethod
	def delta(z,a,y):
		return (a-y)

class QuadraticCost(object):
	@staticmethod
	def fn(a,y):
		return 0.5*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(z,a,y):
		return (a-y) * sigmoid_prime(z)

class Network(object):
	def __init__(self,sizes,cost=CrossEntropyCost):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.default_weight_initializer()
		self.cost = cost

	def default_weight_initializer(self):
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y,x)/np.sqrt(x) for x, y in zip(self.sizes[:-1],self.sizes[1:])]

	def large_weight_initializer(self):
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

	def feed_forward(self,a):
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a) + b)
		return a

	def SGD(self,train_data,epochs,mini_batch_size,eta,
		lmbda = 0.0,
		evaluation_data = None,
		monitor_evaluation_cost=False,
		monitor_evaluation_accuracy=False,
		monitor_train_cost=False,
		monitor_train_accuracy=False):
		if evaluation_data: n_data = len(evaluation_data)
		n = len(train_data)
		evaluation_cost,evaluation_accuracy = [],[]
		train_cost,train_accuray = [],[]
		for j in range(epochs):
			random.shuffle(train_data)
			mini_batches = [train_data[k:k+mini_batch_size]\
							for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta,lmbda,len(train_data))
			print "Epoch %s training complete" % j

			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data,lmbda,convert=True)
				train_cost.append(cost)
				print "Cost on evaluation data:{}".format(cost)
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				print "Accuracy on evaluation data:{}/{}".format(\
					self.accuracy(evaluation_data),n_data)
			if monitor_train_cost:
				cost = self.total_cost(train_data,lmbda)
				train_cost.append(cost)
				print "Cost on training data:{}".format(cost)
			if monitor_train_accuracy:
				accuracy = self.accuracy(train_data,convert=True)
				train_accuray.append(accuracy)
				print "Accuracy on training data:{}/{}".format(\
					self.accuracy(train_data),n)
			print
			return evaluation_cost,evaluation_accuracy,\
				train_cost,train_accuray

	def update_mini_batch(self,mini_batch,eta,lmbda,n):
		nable_b = [np.zeros(b.shape) for b in self.biases]
		nable_weights = [np.zeros(w.shape) for w in self.weights]

		for x,y in mini_batch:
			delta_nable_b,delta_nable_w = self.backprop(x,y)
			nable_b = [nb+dnb for nb,dnb in zip(nable_b,delta_nable_b)]
			nable_w = [nw+dnw for nw,dnw in zip(nable_weights,delta_nable_w)]
		self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))/nw\
				for w,nw in zip(self.weights,nable_w)]
		self.biases = [(1-eta*(lmbda/n))*b - (eta/len(mini_batch))/nb\
				for b,nb in zip(self.biases,nable_b)]

	def backprop(self,x,y):
		nable_b = [np.zeros(b.shape) for b in self.biases]
		nable_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x]
		zs = []
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		delta = (self.cost).delta(zs[-1],activations[-1],y)
		nable_b[-1] = delta
		nable_w[-1] = np.dot(delta,activations[-2].transpose())
		for l in range(2,self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
			nable_b[-l] = delta
			nable_w[-l] = np.dot(delta,activations[-l-1].transpose())
		return (nable_b,nable_w)

	def accuracy(self,data,convert=False):
		"""convert:由于train_data的标签是向量形式one-hot的，而test_data，evaluation_data的标签是类别性的"""
		if convert:
			results = [(np.argmax(self.feed_forward(x)),np.argmax(y)) \
							for (x,y) in data]
		else:
			results = [(np.argmax(self.feed_forward(x)),y)\
						for (x,y) in data]
		return sum(int(x == y) for(x,y) in results)

	def total_cost(self,data,lmbda,convert=False):
		"""convert:由于train_data的标签是向量形式one-hot的，而test_data，evaluation_data的标签是类别性的"""
		cost = 0.0
		for x,y in data:
			a = self.feed_forward(x)
			if convert: y = vectorized_result(y)
			cost += self.cost.fn(a,y)/len(data)
		cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def save(self,filename):
		data = {"sizies":self.sizes,
				"weights":[w.tolist() for w in self.weights],
				"biases":[b.tolist() for b in self.biases],
				"cost":str(self.cost.__name__)}
		f = open(filename,'w')
		json.dump(data,f)
		f.close()

def load(filename):
	with open(filename,'r') as f:
		data = json.load(f)
	cost = getattr(sys.modules[__name__],data['cost'])
	net = Network(data['sizes'],cost=cost)
	net.weights = [np.array(w) for w in data['weights']]
	net.biases = [np.array(b) for b in data['biases']]
	return net

def vectorized_result(j):
	e = np.zeros((10,1))
	e[j] = 1.0
	return e

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))



