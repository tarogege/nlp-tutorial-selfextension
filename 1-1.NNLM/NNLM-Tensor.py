# code by Tae Hwan Jung @graykode
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sentences = [ "i like dog", "i love coffee", "i hate milk",'i enjoy myself','you like milk','you like coffee','i enjoy dog']

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # number of steps ['i like', 'i love', 'i hate']
n_hidden = 2 # number of hidden units

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        # np.eye create one-hot code v*v 代表矩阵c的初始状态，目标函数 f（wt|wt-1..w1）=p=softmax = wx+b+U（Hx+d）
        #最大化对数似然概率，随机梯度下降，参数theta（C，omega），omega又可以扩展为（w，d，u，h,b,）,当没有直连选项时，w=0
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

# Model
# model确立阶段，确定网络结果，表达式，参数和每个参数的维度，////确定目标函数，梯度下降算法
X = tf.placeholder(tf.float32, [None, n_step, n_class]) # [batch_size, number of steps, number of Vocabulary]
Y = tf.placeholder(tf.float32, [None, n_class])

# 确定模型变量纬度 input 句子单词数量*（批处理词数*每个词的特征值纬度）对应论文中|V|(nm+h)
# 实际两个隐藏层 达到了降纬的效果，n*m -> h -> class
# 变量HdUb 的初始值 通过标准正态分布求取 n_step表达的就是 每个句子除开句尾词所剩下的词的数量，在这个案例中 固定为2，实际情况会更为复杂
# n_class 即为论文中|V|的概念
input = tf.reshape(X, shape=[-1, n_step * n_class]) # [batch_size, n_step * n_class]
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 套入模型的激活函数，获得实际的model值，model维度|V|
tanh = tf.nn.tanh(d + tf.matmul(input, H)) # [batch_size, n_hidden]
model = tf.matmul(tanh, U) + b # [batch_size, n_class]

# 目标函数 ：最大化对数似然概率的平均值/ 最小化损失函数，对model维度为|v|的向量求softmax，得到 |V|词典中每个次在当前上下文中产生的概率
# argmax，取概率最大词的为当前词的下标，此时的下标与词汇库里次词的下标对齐
# reduce——mean 最后输出结果是一个数值
# 使用adam优化随机梯度下降，求解目标函数对应的参数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction =tf.argmax(model, 1)

# Training
# 训练时期 初始化变量，创建session/session.run
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 随机梯度下降，循环5000次，每次输出当前变量所对应的损失函数
# 取最后三个句子为测试集
input_batch, target_batch = make_batch(sentences[:-3])

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
# 输出最后稳定的参数 H d U b

# Predict
# 区分训练集和测试集，当测试集包含训练集中没有的词汇时
predict_input_batch,real_target_batch = make_batch(sentences[-3:])
result,test_const,predict = sess.run([model,cost,prediction],feed_dict={X: predict_input_batch,Y:real_target_batch})

print('predict0:',predict)
predict1 = np.array(predict)
print('predict1:',predict1.size)



# Test
input = [sen.split()[:2] for sen in sentences[-3:]]
print([sen.split()[:2] for sen in sentences[-3:]], '->', [number_dict[n] for n in predict])
#print([sen.split()[:2] for sen in sentences[-2:]], '->', number_dict[predict])