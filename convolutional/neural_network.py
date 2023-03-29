import numpy as np, gzip, random, math
from PIL import Image

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
	return max(z, 0)

def ReLU_prime(z):
	return 1 if z > 0 else 0

class connected_layer:
	# input_shape -> the number of inputs
	# layer_shape -> the number of neurons in the layer
	def __init__(self, input_shape: int, layer_shape: int):
		self.biases = np.random.randn(layer_shape)
		self.b_adj = np.zeros(layer_shape)
		self.weights = np.random.randn(input_shape, layer_shape)
		self.w_adj = np.zeros((input_shape, layer_shape))
		self.values = np.zeros(layer_shape)
		self.activations = sigmoid(self.values)

	def forward_prop(self, input_):
		self.values = np.dot(input_, self.weights) + self.biases
		self.activations = sigmoid(self.values)

class convolutional_layer:
	# filter_shape -> a list [side_length, depth, channels] for the dimensions and count of the filters
	# depth should be the number of channels/layers the input has (e.g RGB -> 3)
	# channels should be the number of filters/feature maps
	def __init__(self, filter_shape: list, pooling_size: int):
		self.filt_size = filter_shape[:-1]
		self.filt_radius = int((self.filt_size[0] - 1) / 2)
		self.channels = filter_shape[-1]
		self.filters = np.random.randn(self.channels, self.filt_size[0], self.filt_size[1], self.filt_size[0])
		self.pooling_size = pooling_size

	# input_ must be an image array [height, depth, width]
	def convolve(self, input_):
		input_ = np.array(input_)
		output_ = np.zeros((len(input_) - 2, len(input_[0][0]) - 2))
		for filter_ in self.filters:
			for y in range(self.filt_radius, len(input_) - (self.filt_radius * 2) + 1):
				for x in range(self.filt_radius, len(input_[0][0]) - (self.filt_radius * 2) + 1):

					# rearranges from [height, depth, width] to [width, height, depth]
					input2 = np.transpose(input_, (2, 0, 1))
					input2 = input2[x - self.filt_radius:x + (self.filt_radius * 2)]
					# rearranges from [width, height, depth] to [height, width, depth]
					input3 = np.transpose(input2, (1, 0, 2))
					input3 = input3[y - self.filt_radius:y + (self.filt_radius * 2)]
					# rearranges back to default [height, depth, width] then gets the dot product with the filter
					input4 = np.transpose(input3, (0, 2, 1))
					output_[y - 1][x - 1] = np.dot(filter_.flatten(), input4.flatten())

		print(output_)
		print(output_.shape)

	def forward_prop(self, input_):
		pass


input_arr = np.array(Image.open("C:/Users/Abram/Desktop/Programming/Python/mnist_machine_learning/training_data/factorio_image.png"))
input_arr = np.transpose(input_arr[:,:,:3], (0, 2, 1))
print(input_arr.shape)

layer = convolutional_layer([3, 3, 4], 2)
layer.convolve(input_arr)

# input_arr = np.random.rand(5, 1, 6)
# print(input_arr)
