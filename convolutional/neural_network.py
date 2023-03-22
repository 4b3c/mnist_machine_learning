import numpy as np, gzip, random, math

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def ReLU(z):
	return max(z, 0)

def ReLU_prime(z):
	return 1 if z > 0 else 0
