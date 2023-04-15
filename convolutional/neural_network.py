import numpy as np, gzip, random, math
from PIL import Image

# np.random.seed(0)

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
		self.filters = np.transpose(self.filters, (0, 1, 3, 2))
		self.pooling_size = pooling_size

	# input_ must be an image array [height, depth, width]
	def convolve(self, input_):
		input_ = np.array(input_)
		output_ = np.zeros((len(input_) - 2, len(input_[0][0]) - 2))

		for filter_ in self.filters:
			for y in range(self.filt_radius, len(input_) - (self.filt_radius * 2) + 1):
				for x in range(self.filt_radius, len(input_[0][0]) - (self.filt_radius * 2) + 1):

					# extract desired convolution slice
					input_slice = input_[y - self.filt_radius:y + (self.filt_radius * 2), :, x - self.filt_radius:x + (self.filt_radius * 2)]
					# transpose slice to match filter dimensions
					output_[y - 1][x - 1] = np.dot(filter_.flatten(), input_slice.flatten())

		return output_

	# input must be an array [height, width]
	def max_pooling(self, input_):
		output_ = np.zeros((int(input_.shape[0] / 2), int(input_.shape[1] / 2)))
		
		for h in range(0, input_.shape[0] - 1, 2):
			for w in range(0, input_.shape[1] - 1, 2):
				output_[h//2, w//2] = np.max(input_[h:h + 2, w:w + 2])
		
		return output_

	def forward_prop(self, input_):
		convolved = self.convolve(input_)
		# pooled = self.max_pooling(convolved)

		return convolved


input_arr = np.array(Image.open("C:/Users/Abram/Desktop/Programming/Python/mnist_machine_learning/training_data/idk.png"))[:,:,:3]
input_arr = np.transpose(input_arr / 255, (0, 2, 1))

layer = convolutional_layer([3, 3, 1], 2)
feature_map = layer.forward_prop(input_arr) * 200

img = Image.fromarray(np.uint8(feature_map))
img.save('feature_map.jpg')