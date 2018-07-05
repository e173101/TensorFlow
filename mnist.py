from tensorflow.examples.tutorials.mnist import input_data
#有警告，下一个版本不再这么用了
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

print("Training data size:", mnist.train.num_examples)
print("Validating data size:", mnist.validation.num_examples)
print("Testing data size:", mnist.test.num_examples)
print("Example training data:", mnist.train.images[0])
print("Example training data label:",mnist.train.labels[0])

#MNIST数据集相关常数
INPUT_NODE = 784 #输入层的节点数 28X28=784个像素
OUTPUT_NODE = 10 #输出层的节点数 0~10

#配置神经网络的参数
LAYER1_NODE = 500   #隐藏层节点数。使用只有一个隐藏层的网络
                    #一层500个
BATCH_SIZE = 100    #batch 大小和梯度下降有关？
LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
REGULARIZATION_RATE = 0.0001    #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000      #训练轮数
MOVING_AVERAGE_DECAY = 0,99 #滑动平均衰减率

def inferenc(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用了RelU激活函数
        layer1 = tf.nn.rulu(tf.matmul(input_tensor, weights1) + biases1)

        return tf.matmul(layer1,weights2) + biases2

    else:
        layer1 = tf.nn.relu(
            tf.matul(input_tensor, avg_class.average(weights1)) +
            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) +
            avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32,[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32,[None, OUTPUT_NODE], name = 'y-input')

    #隐藏层参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))
    #输出层参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    bianses2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))

    #计算当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None，
    #所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    #定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里这个为
    #不可训练的变量（trainable=Fasle）。在使用TensorFlow训练神经网络时，
    #一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    

    
