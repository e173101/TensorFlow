# -*- coding: utf-8 -*-
import tensorflow as tf

#相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

#通过tf.get_veriable来获取变量。在训练的时候创建这些变量；在测试的时候通过保存的模型加载这些变量的值。
#更加方便的一点是：？？滑动平均啥

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer = tf.truncated_normal_initializer(stddev = 0.1))
    #当给出了正则化函数时，将当前变量的正则化损失加入名字为losses的集合。
    #这是自定义的集合？？
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义神经网络的权限传播过程
def inference(input_tensor, regularizer):
    #声明第一次神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        #get_variable和Variable没有区别，因为没有在同一程序中调用多次
        weights = get_weight_variable(
            [INPUT_NODE,LAYER1_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [LAYER1_NODE],
            initializer = tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [OUTPUT_NODE],
            initializer = tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
        
    return layer2
             