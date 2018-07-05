#神经网络三步走

import tensorflow as tf

from numpy.random import RandomState

#1 定义神经网络的结构和前向传播的结果
batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1, seed =1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1, seed =1))

#在shap的一个维度上使用None可以方便使用不同的batch大小。在训练时需要把数据分成比较小的batch
#一次太大会有可能内存溢出
x = tf.placeholder(tf.float32, shape = (None,2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None,1), name = 'y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#2 定义损失函数和反向传播算法
y = tf.sigmoid(y)
#这是交叉熵的公式
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
#信息论：交叉熵=p（x）logq（x）
#clip_by_value：限制张量中数值的范围，可以避免运算错误（log0）
#log：所有元素依次求对数
#* ：矩阵元素相乘：矩阵乘法用的是matmul
#reduce_mean：整个矩阵，所有元素取平均

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
print(X)
# 定规则给出样本的标签。在这里所有x1+x2<1的阳历都被认为是正样本
Y = [[int(x1+x2 <1)]for (x2,x1) in X]
print(Y)
# 这个循环没怎么看懂


#3 创建会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)

    print (sess.run(w1))
    print (sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size 个样本进行训练
        start = (i * batch_size) % dataset_size
        end= min(start+batch_size, dataset_size)

        #通过选取的样本训练神经网络并更新参数w1,w2
        sess.run(train_step,
                 feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并且输出
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict = {x: X,y_:Y})
            print("After %d trining step(s), cross entropy on all data is %g" % (i,total_cross_entropy))

    print (sess.run(w1))
    print (sess.run (w2))
    

