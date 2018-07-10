# -*- coding:utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inference
import train

#每十秒加载一次模型！然后测试正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    # 定义输入输出的格式
    x = tf.placeholder(tf.float32,[None, inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32,[None, inference.OUTPUT_NODE], name='y-input')
    validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
    
    #第二个参数是正则化损失
    y = inference.inference(x,None)
    
    #计算正确率
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #?
    variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
                print("After %s training step(s), valiadation accuracy = %g" %(global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return
            time.sleep(EVAL_INTERVAL_SECS)
            
def main(argv= None):
    mnist = input_data.read_data_sets("../MNIST_data", one_hot = True)
    evaluate(mnist)
    
if __name__ == '__main__':
    tf.app.run()