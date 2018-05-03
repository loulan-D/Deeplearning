# 在神经网络的结构上，深度学习一方面使用激活函数 实现神经网络的去线性化。另一方面使用
# 一个或者多个隐藏层，使得神经网络的结构更深，以解决复杂问题。
# 使用带指数衰减的学习率设置、使用正则化来避免过拟合，使用滑动平均模型使得最终模型更加健壮
# Tensorflow训练神经网络实现MNIST

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关系数
INPUT_NODE = 784    # 输入层的节点数
OUTPUT_NODE = 10    # 输出层节点数

# 配置神经网络的参数
LAYER1_NODE = 500      # 隐藏层节点数 设置只有一层隐藏层
BATCH_SIZE = 100       # 一个批处理中训练数据个数，数字越小时，训练过程越接近随机梯度下降；数字越大时，训练过程越接近梯度下降
LEARNING_RATE_BASE = 0.8   # 基础的学习率
LEARNING_RATE_DECAY = 0.99   # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000    # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率

# 给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型
def inference(input_tensor,avg_class,weights1,biases1,weight2,biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1) + biases1)
        # 计算输出层的前向传播结果， 计算整个神经网络的前向传播时可以不加入最后的sotfmax层
        return tf.matmul(layer1,weight2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均，然后计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weight2)) + avg_class.average(biases2)

# 训练模型
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')

    # 生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal(LAYER1_NODE,OUTPUT_NODE),stddev = 0.1)
    biases2 = tf.Variable(tf.constant(0.1,shape =[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果
    y = inference(x,None,weight1,biases1,weight2,biases2)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量trainable = None
    # 在使用Tensorflow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0,trainable = False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    # 在所有代表神经网路参数的变量上使用滑动平均
    variable_average_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。滑动平均不会改变变量本身的取值，而是会维护一个影子变量来记录其滑动平均值
    average_y = inference(x,variable_averages,weight1,biases1,weight2,biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
    # 计算在当前batch中所有 样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算l2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失 一般只就散神经网络权重的正则化损失
    regularization = regularizer(weight1) + regularizer(weight2)
    # 总损失为交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step, # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE, # 所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY  # 学习速率递减速度。
    )

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
# 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值
# 为了一次完成多个操作，Tensorflow提供了tf.control_dependencies和tf.group两种机制。下面程序等价
# train_op = tf.group(train_step,variables_averages_op)
with tf.control_dependencies([train_step,variables_averages_op]):
    train_op = tf.no_op(name = 'train')

"""

"""
correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 初始化会话并开始训练过程。
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的结果
    validate_feed = {x:mnist.validation.images,
                    y_:mnist.validation.labels}
    # 准备测试数据，这个数据作为模型优劣的最后评判标准
    test_feed = {x:mnist.test.images,y_:mnist.test.labels}
    # 迭代地训练神经网络
    for i in range(TRAINING_STEPS):
        # 每1000轮输出一次在验证数据集上的测试结果
        if i% 1000 == 0:
            # 当神经网络模型比较复杂或验证数据比较大时，太大的batch会导致计算时间过长甚至发生内存溢出的错误
            validate_acc = sess.run(accuracy,feed_dict = validate_feed)
            print('atfer %d training step, validation accuracy using average model is %g'%(i,validate_acc))

        # 产生这一轮使用的一个batch的训练数据，并运行训练过程
        xs,ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_op,feed_dict={x:xs,y_:ys})

    # 在训练结束后，在测试数据上检测神经网络模型的最终正确率
    test_acc = sess.run(accuracy,feed_dict = test_feed)
    print("after %d training steps , test accuracy using average model is %g"% (TRAINING_STEPS,test_acc))

# 主程序
def main(argv = None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
    mnist = input_data.read_data_sets('/tmp/data',one_hot = True)
    train(mnist)

# tensorflow 提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
