import numpy as np, gzip, random, math


def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def get_images(path):
	with gzip.open(path, 'r') as f:
		magic_number = int.from_bytes(f.read(4), 'big')
		image_count = int.from_bytes(f.read(4), 'big')
		row_count = int.from_bytes(f.read(4), 'big')
		column_count = int.from_bytes(f.read(4), 'big')
		image_data = f.read()
		images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
		return images

def get_labels(path):
	with gzip.open(path, 'r') as f:
		magic_number = int.from_bytes(f.read(4), 'big')
		label_count = int.from_bytes(f.read(4), 'big')
		label_data = f.read()
		labels = np.frombuffer(label_data, dtype=np.uint8)
		return labels

def get_data(tr_set_size = 60000, tst_set_size = 10000):
	train_i = get_images("training_data/train-images-idx3-ubyte.gz") / 255
	labels = get_labels("training_data/train-labels-idx1-ubyte.gz")
	train_l = np.zeros((tr_set_size, 10))
	for lbl, cnt in zip(labels, range(tr_set_size)):
		train_l[cnt, lbl] += 1

	train_data = np.array([[img, lbl] for img, lbl in zip(train_i, train_l) ], dtype = object)

	test_i = get_images("training_data/t10k-images-idx3-ubyte.gz") / 255
	test_l = get_labels("training_data/t10k-labels-idx1-ubyte.gz")


	return (train_data, test_i, test_l)


class layer:
	def __init__(self, num_of_inp, num_of_nrn):
		self.biases = np.random.randn(num_of_nrn)
		self.b_adj = np.zeros(num_of_nrn)
		self.weights = np.random.randn(num_of_inp, num_of_nrn)
		self.w_adj = np.zeros((num_of_inp, num_of_nrn))
		self.values = np.zeros(num_of_nrn)
		self.activations = sigmoid(self.values)

	def forwardprop(self, input_):
		self.values = np.dot(input_, self.weights) + self.biases
		self.activations = sigmoid(self.values)


class neural_network:
	def __init__(self, n_per_l):
		self.layers = [layer(inp, nrn) for inp, nrn in zip(n_per_l, n_per_l[1:])]
		self.num_of_layers = len(n_per_l)
		print("Network created")

	def forwardprop(self, input_):
		self.input_ = input_
		for layer in self.layers:
			layer.forwardprop(input_)
			input_ = layer.activations

		self.output = layer.activations

	def backwardprop(self, target):
		cost = [-math.log(-x + 1) for x in self.output]
		if self.output[np.argmax(target)] != 0:
			cost[np.argmax(target)] = 2 * (math.log(self.output[np.argmax(target)]))
		else:
			cost[np.argmax(target)] = 2 * (math.log(0.01))

		cost_prime = cost * sigmoid_prime(self.layers[-1].values)

		self.layers[-1].b_adj = cost_prime
		shaped_cost = np.array([cost_prime]).transpose()
		shaped_activations = np.array([self.layers[-2].activations])
		self.layers[-1].w_adj = np.dot(shaped_cost, shaped_activations).transpose()

		for l in range(2, len(self.layers) + 1):
			cost_prime = np.dot(self.layers[-l + 1].weights, cost_prime) * sigmoid_prime(self.layers[-l].values)

			if l == len(self.layers):
				prev_act = self.input_
			else:
				prev_act = self.layers[-l - 1].activations

			self.layers[-l].b_adj += cost_prime
			self.layers[-l].w_adj += np.dot(np.array([cost_prime]).transpose(), np.array([prev_act])).transpose()

	def train(self, lrn_rt, batch_size, epochs, ep_size, inputs):
		train_data, test_i, test_l = inputs
		print("Data gathered")
		print("Starting training")

		for ep in range(epochs):
			random.shuffle(train_data)
			mini_batches = [train_data[k:k + batch_size] for k in range(0, len(train_data), batch_size)]
			for mini_batch in mini_batches:
				for img, lbl in mini_batch:
					self.forwardprop(np.reshape(img, (784)))
					self.backwardprop(lbl)

				for layer in self.layers:
					layer.biases = layer.biases - (layer.b_adj / batch_size) * lrn_rt
					layer.weights = layer.weights - (layer.w_adj / batch_size) * lrn_rt
					layer.b_adj = np.zeros(layer.biases.shape)
					layer.w_adj = np.zeros(layer.weights.shape)

			correct = 0
			for img, lbl in zip(test_i, test_l):
				self.forwardprop(np.reshape(img, (784)))
				if np.argmax(self.output) == lbl:
					correct += 1

			print("Epoch", ep, ":", correct, "/", len(test_l))

		return correct
