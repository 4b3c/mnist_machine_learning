import numpy as np, pickle, pygame, math
from PIL import Image

class un_pickle(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "imsotiredofthis"
        return super().find_class(module, name)

with open('bestnetwork.pickle', 'rb') as f:
    unpickler = un_pickle(f)
    net = unpickler.load()

f.close()

def map(min_, max_, input_):
	minimum = abs(min(list(input_)))
	input_ = [val + minimum for val in input_]
	ratio = max_ / (max(input_) + min_)
	output = np.array(input_) * ratio
	return output

def backward_forward(network, input_):
	input_list = np.zeros((10))
	input_list[input_] = 1
	for layer in network.layers[::-1]:
		weights = layer.weights
		input_lists = np.array([input_list / n_weights for n_weights in weights])
		input_list = np.array([sum(column) for column in np.array(input_lists)])

	return input_list

output = map(0, 255, backward_forward(net, 4)).round(2)

pygame.init()
window = pygame.display.set_mode((784, 784))

for val, cnt in zip(output, range(len(output))):
	val = int(val)
	if cnt != 0:
		x = len(output) % cnt
		y = int(len(output) / cnt)
	else:
		x = 0
		y = 0
	pygame.draw.rect(window, ((val, val, val)), (x * 28, y * 28, 28, 28))
	print(x, y)

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			quit()

	pygame.display.update()

