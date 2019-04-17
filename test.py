import  tensorflow as tf
"""

sess = tf.Session()
z_prior=tf.constant(5,shape=[4,16])
print(sess.run(z_prior))
z_prior=tf.unstack(z_prior,4,0)#这里的4必须和上面的4对应
print(sess.run(z_prior))

"""
"""
#sess = tf.Session()

z_prior = tf.constant(5., shape=[13, 56,10])
z_prior = tf.unstack(z_prior, 10, 2)
lstm_cell = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(7)) for _ in range(2)]);
res, states = tf.contrib.rnn.static_rnn(lstm_cell, z_prior, dtype=tf.float32);

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(res.shape)
"""
def attention(inputs, attention_size, time_major=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vectorc
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 2)

    return alphas,output

# z_prior = tf.constant(5., shape=[13, 4,10])
# z_prior = tf.unstack(z_prior, 10, 2);  # shape 1,16,1
# lstm_cell = tf.contrib.rnn.MultiRNNCell(
#     [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(7)) for _ in range(2)]);
# with tf.variable_scope("gen") as gen:
#     res, states = tf.contrib.rnn.static_rnn(lstm_cell, z_prior, dtype=tf.float32);
#     weights = tf.Variable(tf.random_normal([7, 8]));
#     biases = tf.Variable(tf.random_normal([8]));
#     #print(len(res))
#
#     for i in range(len(res)):
#         res[i] = tf.nn.tanh(tf.matmul(res[i], weights) + biases);
#
#     tensor_a = tf.convert_to_tensor(res)
#     tensor_b = tf.transpose(tensor_a,perm=[1,2,0])
#     tensor_c = tf.transpose(tensor_b,perm=[0,2,1])
#     output,out=attention(tensor_c,8)

"""
    W = tf.Variable(tf.truncated_normal([4, 6], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[6]), name="b")
    res = tf.nn.xw_plus_b(out, W, b, name="scores")
"""


    #print(len(res))
    #print(len(res[0]))
    
sess = tf.Session()
sess.run(tf.initialize_all_variables())
# print(sess.run(res))
# print(type(res))
# print(sess.run(tensor_a))
# print(tensor_a.shape)
# #print(sess.run(tensor_b))
# print(tensor_b.shape)
# #print(sess.run(tensor_c))
# print(tensor_c.shape)
# print(output.shape)
# print(sess.run(output))
# print(out.shape)
# print(sess.run(out))
# print(states)
# print(sess.run(states))




'''
name_net = 'layer_1'
z_inp=tf.constant(5., shape=[4, 32])
with tf.variable_scope(name_net):
    net = tf.layers.dense(z_inp,
                          units=64,

                          name='fc')
    net = tf.nn.relu(net)

name_net = 'layer_2'
with tf.variable_scope(name_net):
    net = tf.layers.dense(net,
                          units=128,

                          name='fc')
    net = tf.nn.relu(net)

name_net = 'layer_3'
with tf.variable_scope(name_net):
    net = tf.layers.dense(net,
                          units=121,  # cong121-->10

                          name='fc')
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(net))
#print(len(net))
print(type(net))
print(net.shape)
'''

a = "max: = 0.932096 | Recall = 0.956243 | F1 = 0.944015 "
print()
print(a.strip().split()[2])
print(a.strip().split()[6])
print(a.strip().split()[10])