#coding:utf8

import random

import numpy as np

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
	#sigmoid的导数
	return sigmoid(z)*(1-sigmoid(z))

class Network(object):
	def__init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		#左闭右开
		
	def feedforward(self, a):
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a)+b)
		return a
	
	def SGD(self,training_data,epochs,mini_batch_size,eta,
			test_data=None):
		if test_data:n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] \
						for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if test_data:
				print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
			else:
				print("Epoch {} complete".format(j))
	
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
						
	
	