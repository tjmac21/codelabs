import math
import tensorflow as tf
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)

# Each training loop is called an epoch
# You work with a training dataset and a validation dataset
# For each epoch of the same training dataset, the "loss" should decrease
# For each epoch of the same training dataset, the recognition accuracy should increase
# A good model is defined by a loss function (how badly a system does at recognition) and subsequently minimizing the loss

# Data is stored in matrices.
# A mono 28x28 pixel image has a 28x28 matrix [28, 28] (2 dimensional)
# A color version of the same image is 3 dimensional [28, 28, 3] (R, G, B)
# A dozen of these color images is 4 dimensional [28, 28, 3, 12]
# Each of these matrices or tables are called TENSORS
# The list of their dimensions is called SHAPE

# In a neural network, the inputs are these tensors.
# The output layer contains as many "neurons" as there are labels

# Each neuron does a weighted sum on all their inputs with a some constant (BIAS).
# The result is fed into an activation function (usually non-linear).
# Training through the neural network helps determine the WEIGHTS and BIASES.
# The initial values of the weights and biases are randomly selected.

# So to begin, you will have a matrix of m by n randomly selected weights. This matrix we call W
# Where m is the number of labels (or output neurons) and n is the number of pixels in an image (28x28 = 784)
# Take W and compute the weighted sum for the first image through the first neuron.
# Repeat the above for as many images as you have by sequentially moving to the next neuron and computing the weighted sum.
# Wrap back to the first neuron when you get to the last neuron and continue.
# If we have a tensor of 100 28x28 mono images called X (your inputs), and "compute" it through our matrix W,
# what we are really doing is the dot product of X and W (matrix multiplication)
# Each neuron will have its own BIAS constant. This gives us a Bx1 matrix, where b is the number of output neurons we have.
# Lets call this matrix b
# We want to add b to X.W
# If b doesnt have the same dimensionality as X.W you must "broadcast"
# Broadcasting is when you attempt to replicate the smaller matrix as many times as needed to that you can add it to the bigger matrix
# Finally take the result and run it through a SOFTMAX (activation) function to get the formula for a single layer neural network on all the images you ran through it
# L = softmax(X.W + b)
# In Keras, this whole formula is written as:
tf.keras.layers.Dense(10, activation='softmax')
# where 10 is the number of output neurons
# Following this, it is then trivial to chain layers.
# Take the output of one layer and use it as the input to another layer
# How many neurons in each layer is up to you, as long as the final layer has the same amount of labels as you need.
# Which activation function you use is also up to you.
# Typically, one would use RELU activation function on all layers except the last layer uses SOFTMAX.
# RELU = Recitified Linear Unit
# Traditionally, a sigmoid activation function was used but results of RELU have shown to be comparable
sigmoid = 1 / ( 1 + e ^(-x))
softmax = e^Ln / abs(e^L)
# softmax outputs a probability between 0 and 1 for each index of the output layer. 

# The output layer will then contain predictions/probabilites for each label/neuron.
# Now that we have predictions we must see how well they compare to the correct answer.
# The correct answer will be encoded in a ONE-HOT encoding. 
# A loss function is fed both the one-hot correct answer and the outputed predictions, for classification problems, CROSS ENTROPY LOSS
# is most commonly used for classification.
# These loss functions compute the gradient and attempt to find a minimum via gradient descent. 
# Additionally, a function can have many minimum. Or your gradient descent can land you into a saddlepoint.
# For these, we batch training datasets together to add momentum. 
# Also keras/tensorflow exposes optimizers to add some momentum to the gradient descent.
# IE: Adam, Gradient Descent Optimizer, Root Mean Squared optimizer, SGD (stochastic gradient descent)



# Can you understand below??
# A neural network classifier is made of several layers of NEURONS. 
# For image classification these can be DENSE or, more frequently, CONVOLUTIONAL layers. 
# They are typically activated with the RELU activation function. 
# The last layer uses as many neurons as there are classes and is activated with SOFTMAX. 
# For classification, CROSS-ENTROPY is the most commonly used loss function, 
# comparing the ONE-HOT encoded LABELS (i.e. correct answers) with probabilities predicted by the neural network. 
# To minimize the loss, it is best to choose an optimizer with momentum, for example ADAM and train on BATCHES of training images and labels.

# For models built as a sequence of layers Keras offers the Sequential API. 
# For example, an image classifier using three dense layers can be written in Keras as:
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dense(60, activation="relu"),
    tf.keras.layers.Dense(10, activation='softmax') # classifying into 10 classes
])

# this configures the training of the model. Keras calls it "compiling" the model.
model.compile(
  optimizer='adam',
  loss= 'categorical_crossentropy',
  metrics=['accuracy']) # % of correct answers

# train the model
model.fit(dataset, ... )





