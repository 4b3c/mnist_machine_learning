import math, random, time, pygame

learning_speed = 0.25

def mean(list_x):
  total = 0
  for i in list_x:
    total += i

  return total / len(list_x)

#Takes any number and translates it to a number between 0 and 1
def sigmoid(x):
  # return 1 / (1 + math.e**(x * -1))
  return 0.6 * x * x / 2

def sigmoid_prime(x):
  # return x * (1 - x)
  return 0.6 * x

def ReLU(x):
  if x < 0:
    return 0
  else:
    return x

def ReLU_prime(x):
  if x < 0:
    return 0
  else:
    return 1

def real_answer(x):
  # return (math.sin((x - 0.2) * 3) / 2) + 0.5
  # return 1 / (1 + math.e**(x * -9))
  return (x**2)
  # return 3 * x**4 - 0.5
 
#A class neuron which has an index indicating which neuron within a layer it is, a number 
# of input neurons which determines the number of weights and biases we need
class neuron:
  def __init__(self, index, num_of_inputs, value = 0):
    self.index = index
    self.num_of_weights = num_of_inputs
    self.weights = []
    self.biases = []
    self.weights_adjust = []
    self.biases_adjust = []
    self.value = value
    self.value_derivative = []
    for weight in range(num_of_inputs):
      self.weights.append(random.random())
    for bias in range(num_of_inputs):
      self.biases.append(random.random())
    if len(self.weights) == 0:
      self.weights.append(1)

#The input layer class is just like a layer class but it doesn't have an input of its own and 
# so no weights and biases assosiated with it
class input_layer:
  def __init__(self, num_of_neurons):
    self.neurons = []
    self.num_of_neurons = num_of_neurons
    self.step = 0
    for ne_num in range(num_of_neurons):
      self.neurons.append(neuron(ne_num, 0))

#Sets the value of the only neuron in the input layer to the random number in the first input list
  def forward(self, input_):
    for current_neuron in self.neurons:
      current_neuron.value = input_

#A layer object has its own index to determine where in the hidden layers it is, the number of 
# neurons and the layer which is going to be the input 
class layer:
  def __init__(self, index, num_of_neurons, input):
    self.index = index
    self.neurons = []
    self.num_of_neurons = num_of_neurons
    self.input = input
    for ne_num in range(num_of_neurons):
      self.neurons.append(neuron(ne_num, input.num_of_neurons))

#For every neuron in the layer, it takes the input neurons, multiplies them by their weight and adds 
# their bias and passes that through a sigmoid function to map it to a number between 0 and 1
  def forward(self, use_sig = True):
    for current_neuron in range(len(self.neurons)):
      total = 0
      for neuron in range(len(self.input.neurons)):
        value = self.input.neurons[neuron].value
        corr_weight = self.neurons[current_neuron].weights[neuron]
        corr_bias = self.neurons[current_neuron].biases[neuron]
        total = total + (value * corr_weight) + corr_bias
      if use_sig == True:
        self.neurons[current_neuron].value = sigmoid(total)
      else:
        self.neurons[current_neuron].value = ReLU(total)

#The neural network is a class so I can step everything forward all at once and have less inputs
#list of neurons is actually a list of the number of neurons per layer but idk how to shorten that
class neural_network:
  def __init__(self, list_of_neurons):
    self.input_layer = input_layer(list_of_neurons[0])
    self.hidden_layers = []
    for i in range(len(list_of_neurons) - 1):
      if i == 0:
        prev_layer = self.input_layer
      else:
        prev_layer = self.hidden_layers[-1]
      self.hidden_layers.append(layer(i+1, list_of_neurons[i+1], prev_layer))

  #Really just steps forward the input layer, then all the hidden layers
  def forward(self, input_):
    self.input_layer.forward(input_)
    for layer in self.hidden_layers:
      if layer == self.hidden_layers[-1] or layer == self.hidden_layers[-2]:
        layer.forward(False)
      else:
        layer.forward()

  def calculate_cost(self, correct_out, actual_out):
    difference = correct_out - actual_out
    return difference**2

  def backward_pass(self, target_output):
    hidden_layers = self.hidden_layers
    for layer in hidden_layers[::-1]:
      # previous_layer_derivatives = []
      for neuron in layer.neurons:
        neuron.weights_adjust.append([])
        neuron.biases_adjust.append([])
        for weight in neuron.weights:
          if hidden_layers.index(layer) != 0:
            previous_layer = hidden_layers[hidden_layers.index(layer) - 1]
          else:
            previous_layer = self.input_layer

          # if layer == hidden_layers[-1]:
          weight_index = neuron.weights.index(weight)
          prev_corr_neuron = previous_layer.neurons[weight_index]

          par_der = prev_corr_neuron.value * sigmoid_prime(neuron.value) * (neuron.value - target_output)*2
          neuron.weights_adjust[-1].append(par_der)

          par_der_bi = sigmoid_prime(neuron.value) * (neuron.value - target_output)*2
          neuron.biases_adjust[-1].append(par_der_bi)
          # previous_layer.neurons[weight_index].value_derivative.append(sigmoid_prime(neuron.value) * (neuron.value - target_output) * 2 * weight)
        #   else:
        #     weight_index = neuron.weights.index(weight)
        #     prev_corr_neuron = previous_layer.neurons[weight_index]

        #     par_der = prev_corr_neuron.value * sigmoid_prime(neuron.value) * mean(neuron.value_derivative)
        #     neuron.weights_adjust[-1].append(par_der)

        #     par_der_bi = sigmoid_prime(neuron.value) * (neuron.value - target_output) * 2
        #     neuron.biases_adjust[-1].append(par_der_bi)
        #     previous_layer.neurons[weight_index].value_derivative.append(sigmoid_prime(neuron.value) * mean(neuron.value_derivative) * weight)

        # neuron.value_derivative = []



  def adjust_weights(self):
    hidden_layers = self.hidden_layers
    for layer in hidden_layers[::-1]:
      for neuron in layer.neurons:
        for weight in neuron.weights:
          weight_index = neuron.weights.index(weight)
          
          weight_adjustment = 0
          for adjust in neuron.weights_adjust:
            weight_adjustment += adjust[weight_index]
          weight_adjustment = weight_adjustment / len(neuron.weights_adjust)
          neuron.weights[weight_index] += weight_adjustment * -1 * learning_speed
          
          bias_adjustment = 0
          for adjust in neuron.biases_adjust:
            bias_adjustment += adjust[weight_index]
          bias_adjustment = bias_adjustment / len(neuron.biases_adjust)
          neuron.biases[weight_index] += bias_adjustment * -1 * learning_speed

        neuron.weights_adjust = []
        neuron.biases_adjust = []

            
          

pygame.init()

window = pygame.display.set_mode((1000, 600))

#Yes
neurons_per_layer = [1, 3, 3, 3, 1]
network = neural_network(neurons_per_layer)

for i in range(300):
  window.fill((0, 0, 0))
  for x in range(1000):
    pygame.draw.rect(window, ((255, 255, 255)), (x, (real_answer(x / 1000) * 600), 2, 2))
    pygame.display.flip()

  for batch in range(200):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

    batch_list = []
    for i in range(30):
      batch_list.append(random.random())

    answer_list = []
    for x in batch_list:
      answer_list.append(real_answer(x))

    my_answers = []
    for input, correct in zip(batch_list, answer_list):
      network.forward(input)
      my_answers.append(network.hidden_layers[0].neurons[0].value)
      network.backward_pass(correct)


    network.adjust_weights()

    cost = 0
    for input_, my_ans in zip(batch_list, my_answers):
      cost += network.calculate_cost(real_answer(input_), my_ans)
    # print(cost / len(batch_list))

    for input_, my_ans in zip(batch_list, my_answers):
      pygame.draw.rect(window, ((100, 100, 255)), (input_ * 1000, (my_ans * 600), 2, 2))
      pygame.display.flip()

    # print(network.hidden_layers[-4].neurons[0].weights[0], network.hidden_layers[-4].neurons[1].weights[0], network.hidden_layers[-4].neurons[2].weights[0])
    
while True:
  for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        quit()

  time.sleep(0.1)