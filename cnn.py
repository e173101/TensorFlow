import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = 'C:/Users/lot/Pictures/'

def get_file(file_dir):
    cross_walk = []
    label_cross_walk = []
    none_sign = []
    label_none_sign = []

    for file in os.listdir(file_dir+'cross_walk'):
        if os.path.splitext(file)[1] == '.jpg':
            cross_walk.append(file_dir+'cross_walk/'+file)
            label_cross_walk.append(0)
            
    for file in os.listdir(file_dir+'none_sign'):
        if os.path.splitext(file)[1] == '.jpg':
            none_sign.append(file_dir+'none_sign/'+file)
            label_none_sign.append(1)

    print('There are %d corss walk sign\nThere are %d none sign\n' \
          %(len(cross_walk),len(none_sign)))

    image_list = np.hstack((cross_walk,none_sign))
    label_list = np.hstack((label_cross_walk,label_none_sign))
    
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)    

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    #类型转换
    
    input_queue = tf.train.slice_input_producer([image,label])
    #列队？为什么这么做

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels = 3)

    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    #resize

    image = tf.image.per_image_standardization(image)
    #?

    image_batch, label_batch = tf.train.batch([image,label],batch_size = batch_size, num_threads=16,capacity = capacity)

    label_batch = tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch
    

BATCH_SIZE = 2 #??
CAPACITY = 32 #??
IMG_W = 40
IMG_H = 40

image_list,label_list = get_file(train_dir)

print(image_list)
print(label_list)

image_batch,label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

with tf.Session() as sess:
    i=0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    try:
        while not coord.should_stop() and i<1:
            img,label = sess.run([image_batch,label_batch])
            
            for j in np.arange(BATCH_SIZE):
                print('label: %d'%label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)

print('exit')

def inference(images, batch_size, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
    return softmax_linear

def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy


N_CLASSES = 2
#要分类的类别数，这里是2分类
IMG_W = 40
IMG_H = 40
#设置图片的size
BATCH_SIZE = 8
CAPACITY = 64
MAX_STEP = 1000
#迭代一千次，如果机器配置好的话，建议至少10000次以上
learning_rate = 0.0001
#学习率

def run_training():
    train_dir = 'C:/Users/lot/Pictures/'
    logs_train_dir = 'C:/Users/lot/Documents/log/'
    train,train_label = get_file(train_dir)
    train_batch,train_label_batch = get_batch(train,train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
    train_logits =inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss = losses(train_logits,train_label_batch)
    train_op = trainning(train_loss,learning_rate)
    train_acc = evaluation(train_logits,train_label_batch)
    
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
            if step %  50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
                
            if step % 200 ==0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()


def get_one_image(img_dir):
     image = Image.open(img_dir)
     #plt.imshow(image)
     image = image.resize([IMG_W, IMG_H])
     image_arr = np.array(image)
     return image_arr

def test(test_file):
    print('文件：',test_file)
    log_dir = 'C:/Users/lot/Documents/log/'
    image_arr = get_one_image(test_file)

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,IMG_W, IMG_H, 3])
        print(image.shape)
        p = inference(image,1,2)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32,shape = [IMG_W, IMG_H,3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                #调用saver.restore()函数，加载训练好的网络模型

                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction) 
            print('预测的标签为：')
            print(max_index)
            print('预测的结果为：')
            print(prediction)

            if max_index==0:
                print('This is a 人行横道路标 with possibility %.6f' %prediction[:, 0])
            else :
                print('This is a  背景 with possibility %.6f' %prediction[:, 1])

run_training();
test('C:\\Users\\lot\\Pictures\\test\\1 (1).jpg')
test('C:\\Users\\lot\\Pictures\\test\\1 (2).jpg')
test('C:\\Users\\lot\\Pictures\\test\\1 (3).jpg')
test('C:\\Users\\lot\\Pictures\\test\\1 (4).jpg')
test('C:\\Users\\lot\\Pictures\\test\\2 (1).jpg')
test('C:\\Users\\lot\\Pictures\\test\\2 (2).jpg')



