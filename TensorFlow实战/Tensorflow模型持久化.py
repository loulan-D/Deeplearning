# Tensorflow模型持久化
# 通过Tensorflow程序来持久化一个训练好的模型，并从持久化之后的模型文件中还原被保存的模型。

# 以下代码：tf.train.Saver类 保存和还原一个神经网络模型
# 保存tensorflow计算图
import tensorflow as tf

# 声明两个变量并计算他们的和
v1 = tf.Variable(tf.constant(1.0,shape = [1]),name = 'v1')
v2 = tf.Variable(tf.constant(2.0,shape =[1]),name = 'v2')

init_op = tf.global_variables_initializer()
# 声明tf.train.Saver类保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存到/path/to/model/model.ckpt文件
    saver.save(sess,"/path/to/model/model.ckpt")

"""
Tensorflow模型一般会存在后缀为.ckpt文件中，虽然只指定了一个文件路径，但是在这个文件目录下会出现三个文件。
这是因为Tensorflow会将计算图的结构和图上参数取值分开保存。

上面代码生成的第一个文件为model.ckpt.meta,他保存了tensorflow计算图的结构。
第二个文件为model.ckpt,这个文件中保存了tensorflow程序中每一个变量的取值。
最后一个文件为checkpoint文件，这个文件中保存了一个目录的所有的模型文件列表。
eg:一个保存的模型文件
checkpoint
model.ckpt-49497.data-00000-of-00001
model.ckpt-49497.index
model.ckpt-49497.meta

# tf.Saver得到的model.ckpt.index和model.ckpt.data-******-of-******文件就保存了所有变量的取值。

"""



# 以下代码： 加载这个已经保存的tensorflow模型
import tensorflow as tf

# 声明变量
v1 = tf.Variable(tf.constant(1.0,shape = [1]),name = 'v1')
v2 = tf.Variable(tf.constant(1.0,shape = [2]),name = 'v2')
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess,'/path/to/model/model.ckpt')
    print(sess.run(result))

# 最后一个文件的名字是固定的，叫checkpoint。这个文件是tf.train.Saver类自动生成且自动维护的。
# 在checkpoint文件中维护了由一个tf.train.Saver类持久化的所有Tensorflow模型文件的文件名。
# 当某个保存的tensorflow模型文件被删除时，这个模型所对应的文件名也会从checkpoint文件中删除。
# 下面给出了CheckpointState类型的定义:
message CheckpointState{
    string model_checkpoint_path = 1;
    repeated string all_model_checkpoint_paths = 2;
}
# model_checkpoint_path 属性保存了最新的tensorflow模型文件的文件名
# all_model_checkpoint_paths 属性保存了当前还没有被删除的所有tensorflow模型文件的文件名。
# 生成的checkpoint文件
model_checkpoint_path: "/path/to/model/model.ckpt"
all_model_checkpoint_paths:"/path/to/model/model.ckpt"
