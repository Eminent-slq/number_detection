# 导入库和数据集
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

# 构建模型
# 1. 定义输入数据并预处理数据，先读取数据mnist，
# 并分别得到训练集的图片和标记的矩阵和测试集的图片和标记的矩阵
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# mnist数据集是黑白的通道为1，图片大小为28*28， -1表示不考虑输入图片的数量
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])


# 初始化权重并定义网络结构
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# patch大小为3*3， 输入维度为1， 32个通道
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
# 全连接层，输入维度为128*4*4，是上一层的输出数据
w4 = init_weights([128 * 4 * 4, 625])
# 输出层，输入维度为625，输出维度为10，代表10类
w_o = init_weights([625, 10])


# 定义一个模型函数，X表示输入数据，w表示每一层的权重，
# p_keep_conv,p_keep_hidden:dropout要保留的神经元比例
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


# 定义dropout的占位符
p_keep_conv = tf.placeholder('float')
p_keep_hidden = tf.placeholder('float')
# 生成网络模型， 得到预测值
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
# 用交叉熵比较预测值和真实值的差异，并作均值处理
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
# 定义训练的操作
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
# 返回张量维数中值最大的下标。0表示列，1表示行
predict_op = tf.argmax(py_x, 1)

batch_size = 128
test_size = 256

saver = tf.train.Saver(max_to_keep=3)
# 在会话中启动一张图
with tf.Session() as sess:
    # 初始化所有变量
    tf.global_variables_initializer().run()

    model_path = "./model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for i in range(1):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        # if i % 10 == 0:
        #     saver.save(sess, model_path + "/model.ckpt", global_step=i)
        #     print("save the model")


        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        # np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices],
        #                                 p_keep_conv: 1.0,
        #                                 p_keep_hidden: 1.0})
        print(teY[test_indices])
        print(np.argmax(teY[test_indices], axis=1))
        print(sess.run(predict_op, feed_dict={X: teX[test_indices],
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0}))
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
            sess.run(predict_op, feed_dict={X: teX[test_indices],
                p_keep_conv: 1.0,
                p_keep_hidden: 1.0})))
