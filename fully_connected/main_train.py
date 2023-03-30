import imsotiredofthis as nn, pickle

data = nn.get_data()

prev_best = 9434
lrn_rt = 8

net = nn.neural_network([784, 70, 30, 10])
best = net.train(lrn_rt, 30, 10, 60000, data)
print(lrn_rt, ":", best)
if best > prev_best:
	prev_best = best
	file = open("bestnetwork.pickle", "wb")
	pickle.dump(net, file)
	file.close()
	print("file overwritten")
