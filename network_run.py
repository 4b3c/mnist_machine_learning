import numpy as np, pickle, pygame
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



pygame.init()
window = pygame.display.set_mode((784, 784))
drawing = False

def get_averages(window):
	pygame.image.save(window, "drawing.png")

	drawing = Image.open("drawing.png")
	drawing = drawing.resize((28, 28), Image.Resampling.LANCZOS)

	pixel_values = []
	for i in range(28):
		for j in range(28):
			pixel_values.append(drawing.getpixel((j, i))[0])

	return pixel_values



print("Hold down left click to draw. Use right click to clear.")
while True:
	pygame.display.update()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			quit()

	if pygame.mouse.get_pressed()[0]:
		mouse_pos = pygame.mouse.get_pos()
		pygame.draw.circle(window, ((255, 255, 255)), (mouse_pos[0], mouse_pos[1]), 40)
		drawing = True

	elif drawing == True:
		data = np.array(get_averages(window))
		data = data / 255
		net.forwardprop(data)
		print(np.argmax(net.output))
		drawing = False

	elif pygame.mouse.get_pressed()[2]:
		window.fill((0, 0, 0))
		drawing = False