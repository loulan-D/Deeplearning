
# Section 1
#tensorflow优化带正则化的loss function
# 带l2正则化的损失函数定义
w = tf.Variable(tf.random_normal([2,1],stddev=1,seed = 1))
y = tf.matmul(x,w)

# 均方误差损失函数： 刻画了模型在训练数据上的表现
# 正则化： 防止模型过度模拟训练数据中的随机噪音
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)

# Section 2
# 计算l1正则化和l2正则化
weights = tf.constant([[1.0,-2.0],[-3.0,4.0]]) # 定义权重
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))  # 0.5为正则化项的权重
    # tensorflow 会将l2正则化损失值除以2使得求导得到的结果更加简洁
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))

# Session 3
# 使用tensorflow中提供的集合（collection）计算一个5层神经网络带l2正则化的损失函数。
import tensorflow as tf

# 获取一层神经网络边上的权重，并将这个权重的l2正则化损失加入名称为‘losses’的集合中
def get_weight(shape,lambda):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
    # add_to_collection函数将新生成变量的l2正则化损失项加入集合
    # tf.add_to_collection(集合名字，加入集合内容)
    tf.add_to_collection(
        'losses',tf.contrib.layers.l2_regularizer(lambda)(var)
    )
    return var

x = tf.placeholder(tf.float32,shape = (None,2)) # 特征
y_ = tf.placeholder(tf.float32,shape = (None,1)) # 标签
batch_size = 8 # 每次批处理大小
layer_dimension  = [2,10,10,10,1]  # 每一层网络中节点个数
n_layers = len(layer_dimension)  # 神经网络的层数

# 维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成一个5层的全连接神经网络结构
for i in range(1,n_layers):
    # layer_dimension[i] 为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的l2正则化损失加入计算图中的集合
    weight = get_weight([in_dimension,out_dimension],0.001)
    bias = tf.Variable(tf.constant(0.1,shape = [out_dimension]))
    # 使用relu激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+ bias)
    # 将下一层的节点个数更新为当前层的节点个数
    in_dimension = layer_dimension[i]

# 在定义神经网络前向传播的同时已经将所有的l2正则化损失加入了图上的集合
# 计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差损失加入到损失集合
tf.add_to_collection('losses',mse_loss)

# get_collection返回一个列表,相加得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))
