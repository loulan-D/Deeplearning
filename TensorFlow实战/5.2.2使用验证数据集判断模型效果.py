# 评测神经网络模型在不同参数下的效果，一般从训练数据中抽取一部分做为验证数据。
# 使用验证数据就可以评判不同参数取值下模型的表现。

# 计算滑动平均模型在测试数据和验证数据上的正确率
validate_acc = sess.run(accuracy,feed_dict = validate_feed)
test_acc = sess.run(accuracy,feed_dict = test_feed)


# 一个模型只最小化交叉熵损失
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    cross_entropy_mean,global_step = global_step
)

# 一个模型最小化交叉熵和l2正则化损失的和
loss = cross_entropy_mean + regularaztion
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss,global_step = global_step
)
