from tensorflow.examples.tutorials.mnist import input_data
#it will be changed
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

print("Training data size:", mnist.train.num_examples)
