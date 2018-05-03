#---------------------------------------------------------
# 在linux下查看系统配置：
# free -m
# cat /proc/cpuinfo
# lspci |grep VGA
# nvidia-smi: 显示机器上gpu的情况
# nvidia-smi -l : 定时更新显示机器上gpu的情况
# watch -n 3 nvidia-smi 设定刷新时间显示gpu使用情况
# 0/1/2/3编号，表示GPU的编号，后面指定GPU时需要使用这个编号

# ---------------------------------------------------------
"""
TensorFlow 使用GPU
tensorflow程序可以通过tf.device函数来指定运行每一个操作的设备，这个设备可以是本地的CPU/GPU或者是某一台远程的服务器。
tensorflow会给每一个可用的设备一个名称，tf.device函数可以通过设备的名称来指定执行运算的设备。
如cpu在tensorflow中的名称是/cpu:0. 在默认情况下，即使机器上有多个CPU，Tensorflow也不会区分他们，所有的cpu都使用/cpu:0作为名称。
而一台机器上不同GPU的名称是不同的，第n个GPU在tensorflow中的名称是/gpu:n.如/gpu:0,/gpu:1

# tensorflow提供了一个快捷的方式来查看运行每一个运算的设备。在生成会话时，可以通过设置log_device_placement参数来打印运行每一个运算的设备。
"""
import tensorflow as tf

a = tf.constant([1.0,2.0,3.0],shape = [3],name = 'a')
b = tf.constant([1.0,3.0,3,0],shape = [3],name = 'b')
c = a + b
# 通过log_device_placement参数来输出运行每一个运算的设备。
sess = tf.Session(config=tf.ConfigProto(log_device_placement= True))
print(sess.run(c))
# add:(Add):/job:localhost/replica:0/task:0/cpu:0

# 在配置好GPU环境的tensorflow中，如果操作没有明确指定运行设备，那么tensorflow会优先选择GPU
# 默认情况下，tensorflow只会将运算优先放到/gpu:0上。

# ----------------------------------------------------------------
#如果需要将某些运算放到不同的cpu或gpu上，就需要通过tf.device手工指定。
import tensorflow as tf

with tf.device('/cpu:0'):
	a = tf.constant([1.0,3.0,3.0],shape=[3],name= 'a')
	b = tf.constant([1.0,2.0,3.0],shape = [3],name = 'b')
with tf.device('/gpu:1'):
	c = a + b

sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
print(sess.run(c))

# 在tensorflow中，不是所有的操作都可以放在gpu上，如果强行把无法放在gpu上的操作指定到gpu上，程序会报错
# 为了避免这个问题，tensorflow 在生成会话时可以指定allow_soft_placement参数,当allow_soft_placement参数设置为true时，如果运算无法由ＧＰＵ执行，
# 那么tensorflow会自动讲它放到cpu上执行。
import tensorflow as tf

a_cpu = tf.Variable(0,name = 'a_cpu')
with tf.device('/gpu:0'):
	a_gpu = tf.Variable(0,name = 'a_cpu')

# 通过allow_soft_placement参数自动将无法放在gpu上的操作放回cpu上
sess = tf.Session(config =tf.ConfigProto(allow_soft_placement = True,log_device_placement = True))
sess.run(tf.initialize_all_variables())

# -------------------------------------------------------------------

"""
一个比较好的实践是把计算密集型的运算放在gpu上，而把其他操作放到cpu上，gpu是机器中相对独立的资源，将计算放入或者转出gpu
都需要额外的时间。而且GPU需要将计算时用到的数据从内存复制到GPU设备上，需要额外的时间，TensorFlow可以自动完成这些操作不需要用户特别处理，
但为了提高程序运行的速度，需要尽量将相关的运算放在同一个设备上。

TensorFlow 默认会占用设备上所有GPU以及每个GPU的所有显存。如果在一个Tensorflow程序中只需要使用部分GPU，通过设置CUDA_VISIBLE_DEVICES环境变量控制

GPU编号从0开始
虽然TensorFlow默认会一次性占用一个GPU的所有显存，但是Tensorflow也支持动态分配GPU的显存，使得一块GPU可以同时运行多个任务。
"""
# 动态分配显存：
config = tf.ConfigProto()
# 让Tensorflow按需分配显存
config.gpu_options.allow_growth = True
# 或者直接按固定的比例分配。如占用所有Gpu的40%显存
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ... )


# ----------------------------------------------------------------------

# 在终端执行程序时指定GPU
CUDA_VISIBLE_DEVICES=1 python file.py
# 跑程序时，告诉程序只能看到1号GPU，其他的GPU它不可见
# 可用的形式如下：
CUDA_VISIBLE_DEVICES=1      # only device 1 will be seen
CUDA_VISIBLE_DEVICES=0,1    # devices 0 and 1 will be visible
CUDA_VISIBLE_DEVICES='0,1'  # devices 0 and 1 will be visible
CUDA_VISIBLE_DEVICES=0,2,3  # devices 0,2,3 will be visible ,device 1 is masked

CUDA_VISIBLE_DEVICES=""    #　no GPU will be visible

# 在python代码中指定gpu
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#　设置定量的gpu使用量
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用90%的显存
session = tf.Session(config = config)
# --------------------------------------------------------------------------

