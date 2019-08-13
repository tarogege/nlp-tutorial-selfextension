'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

# TextRNN Parameter
n_step = 2 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell

def make_batch(sentences):
    input_batch = []
    target_batch = []
    
    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

# Model
# RNN 模型分为三层：输入，隐藏层，输出层
# 隐藏层 ht = f(h*ht-1+H*x=b)
# 输出层 softmax y = W*ht+b
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, n_step, n_class]
Y = tf.placeholder(tf.float32, [None, n_class])         # [batch_size, n_class]

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 常用rnn BasicRNNCell
# RNNCell
# BasicLSTMCell
# LSTMCell
# GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
# ouputs 输出的tensor默认为[batch_size,max_time,hidden_size] state最终的状态 [batch_size,n_hidden]
# 运行结果为 隐藏层的值
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# tf.transpose(a矩阵,perm) 矩阵的转置，perm，转置后 维度的排列顺序
# 转置之前的outputs : [batch_size, n_step, n_hidden]
# outputs = tf.transpose(outputs, [1, 0, 2]) # [n_step, batch_size, n_hidden]
# outputs = outputs[-1] # [batch_size, n_hidden]
# model = tf.matmul(outputs, W) + b # model : [batch_size, n_class]

# 这里直接用states就可以了
# 运行结果为输入层的值，通过求softmax的交叉墒损失，梯度下降求最值
model = tf.matmul(states, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, target_batch = make_batch(sentences)

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        
input = [sen.split()[:2] for sen in sentences]

predict =  sess.run([prediction], feed_dict={X: input_batch})
print('predict:',predict)
print([sen.split()[:2] for sen in sentences], '->',[number_dict[n] for n in predict[0]])