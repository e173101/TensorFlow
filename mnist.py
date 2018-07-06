import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#有警告，下一个版本不再这么用了




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
TRAINING_STEPS = 5000      #训练轮数
MOVING_AVERAGE_DECAY = 0,99 #滑动平均衰减率

def inference(input_tensor, avg_class, weights1, biases1, weights2,biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用了RelU激活函数
        #layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        layer1 = tf.matmul(input_tensor, weights1 + biases1)

        return tf.matmul(layer1,weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

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
    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))

    #计算当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None，
    #所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    #定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里这个为
    #不可训练的变量（trainable=Fasle）。在使用TensorFlow训练神经网络时，
    #一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0.0, trainable = False)
	
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    #给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage (
	MOVING_AVERAGE_DECAY, global_step)

    #在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量就不需要了
    #tf.trainable_variables返回的就是图上合集
    #GraphKes.TRAINABLE_VARIABLES中的元素，就是没有指定trainable=False的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用了滑动平均之后的前线传播结果。？？

    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2)

    #没看懂的损失函数
    corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels = tf.argmax(y_,1))
    #计算交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(wieghts1) + regularizer(weights2)
    #总损失等于交叉熵损失和正则化随时的和
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RAGE_BASE, #基础的学习率，随着迭代的进行，更新变量时使用的
                            #学习率在这个基础上递减
        global_step,        #当前迭代的轮数
        mnist.train.num_examples/BATCH_SIZE,
        # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY, #学习衰减速度
        staircase=True)

    #优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

    #一次完成多个操作？
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    #检验?
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session as sess:
        tf.global_variables_initializer().run()
        #准备验证数据？判断停止的条件和评判训练结果
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}

        #准备测试数据。
        test_feed = {x:mnist.test.images, y_:mnist.test.labels}

        #迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输入一次在验证数据集上的测试结果
            if i % 1000 == 0:
                #关于batch？
                validate_acc = sess.run(accuracy, feed_dict = validate_feed)
                print("After %d training step(s), validation accuracy"
                      "using average model is %g" % (i, validate_acc))
            #
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs, y_:ys})

        test_acc = sess.run(accuracy, feed_dict = test_feed)
        print("After %d training step(s), test accuracy using average "
              "model is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None):
    #哈？  
    mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
    
    print("Training data size:", mnist.train.num_examples)
    print("Validating data size:", mnist.validation.num_examples)
    print("Testing data size:", mnist.test.num_examples)
    print("Example training data:", mnist.train.images[0])
    print("Example training data label:",mnist.train.labels[0])
    train(mnist)

# 主程序入口？
if __name__=='__main__':
    tf.app.run()
	
	
