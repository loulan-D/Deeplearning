# Tensorflow中通过变量名称获取变量的机制主要是通过tf.get_variable和tf.variable_scope实现

# 通过tf.get_variable创建或获取变量
v = tf.get_variable('v',shape= [1],initializer = tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0,shape = [1],name = 'v'))

"""
# tensorflow 的变量初始化函数
初始化函数                                      功能                               主要参数
tf.constant_initializer                将变量初始化为给定常量                  常量的取值
tf.random_normal_initializer         将变量初始化为满足正太分布的随机值         正态分布的均值和标准差
tf.truncated_normal_initializer      将变量初始化为满足正态分布的随机值，但如果   正态分布的均值和标准差
                                     随机出来的值偏离平均值超过2个标准差，那么这个
                                     数会重新随机
tf.random_uniform_initializer        将变量初始化为满足平均分布的随机值         最大、最小值
tf.uniform_unit_scaling_initializer   将变量初始化为满足平均分布但不影响输出数量级的随机值        factor（产生随机数时乘以的系数）
tf.zeros_initializer                   将变量设置为0                             变量维度
tf.ones_initializer                    将变量设置为1                             变量维度
"""

# section 1
# 通过tf.variable_scope函数来控制tf.get_variable函数获取已经创建过的变量。
# 在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope('foo'):
    v = tf.get_variable(
        'v',[1],initializer = tf.constant_initializer(1.0)
    )
# 在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable函数将直接获取已经声明的变量
with tf.variable_scope('foo',reuse = True):
    v1 = tf.get_variable("v",[1])
    print(v == v1) # 输出为true
    
