import imsotiredofthis as nn
import numpy as np
import random as rd
from PIL import Image, ImageOps

images = nn.get_data()
new_images = np.array([np.array((28, 28)), np.array((10,))])

for num in range(5):
	image = Image.fromarray(images[0][num][0] * 255)
	for i in range(5):
		image = image.rotate(angle = rd.randint(-20, 20), translate = (rd.randint(-3, 3), rd.randint(-3, 3)))
		data = np.array([np.array(image), images[0][num][1]])
		np.append(new_images, data)

images = new_images
print(images.shape)
print(images[num][1].shape)
print(list(images[0][num][1]).index(max(images[0][num][1])))

for row in images[0][0] :
	for column in row:
		if column > 0.0:
			print("  ", end = " ")
		else:
			print("00", end = " ")
	print("")