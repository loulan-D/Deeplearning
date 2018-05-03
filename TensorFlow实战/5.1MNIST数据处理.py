# dataset introduce
# MNIST包含60000张图片训练数据，10000张图片测试数据
# 图片大小：28*28
from tensorflow.examples.tutorials.mnist import input_data
# import dataset 标签为one_hot编码形式
mnist = input_data.read_data_sets('/path/to/MNIST_data',one_hot=True)

# training data size :55000
print("training data size:{}".format(mnist.train.num_examples))

# validating data size :5000
print('validating data size:{}'.format(mnist.validation.num_examples))

# testing data size:10000
print('testing data size{}'.format(mnist.test.num_examples))

# training data :
print('training data:{}'.format(mnist.train.images[0]))

# training data label:
print('training data label:{}'.format(mnist.train.labels[0]))

# 为了方便使用随机梯度下降，input_data.read_data_sets.train.next_batch函数可以从
# 所有的训练数据中读取一部分作为一个训练batch
# 神经网络的输入是一个特征向量，把一个二维图像的像素放入一个一维数组可以方便作为输入层。
batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)
print(xs.shape,ys.shape)
