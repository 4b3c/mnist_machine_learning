import neural_network as nn
import numpy as np
import cv2
from PIL import Image



video = cv2.VideoCapture("../training_data/unfold.mp4")
layer = nn.convolutional_layer([3, 3, 1], 2)


frame_num = 0
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
images = []
while True: 
	success, image = video.read()
	if not success:
		break
	try:
		if image == None:
			break
	except:
		pass

	input_arr = np.transpose(image / 255, (1, 2, 0))
	feature_map = layer.forward_prop(input_arr) * 200
	image = Image.fromarray(np.uint8(feature_map))
	image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	images.append(image)

	print(frame_num, "of", total_frames)
	frame_num += 1



video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (images[0].shape[1], images[0].shape[0]))
for image in images:
	video_writer.write(image)
video_writer.release()