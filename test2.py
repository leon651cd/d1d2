# import  tensorflow as tf
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# # z = tf.constant(5., shape=[4,5,6])
# # d= tf.constant(5., shape=[4,1,6])
# # #e= tf.concat([z,d],axis=2)
# # # d = tf.constant([[1.],[2.],[3.],[4.]])
# # sess = tf.Session()
# # sess.run(tf.initialize_all_variables())
# # print(tf.squeeze(d).shape)
# # print(e.shape)
# # print(d.shape)
# # print(sess.run(z*d))
# # e=tf.constant([[[1.,2.,3.,4.],[5.,6.,7.,8.]],[[1.,2.,3.,4.],[5.,6.,7.,8.]]])
# # print(sess.run(tf.reduce_sum(e,2)))
# # x = tf.constant([[[1., 1.], [2., 2.],[5., 5.]], [[3., 3.], [4., 4.], [6., 6.]]])
# # print(sess.run(tf.reduce_mean(x, 1)))
# # print(sess.run(tf.reduce_mean(z, 1)))
# # print(tf.transpose(z).shape)
# #
# # n_data = np.loadtxt("train_data_bigan/a",delimiter=' ')
# # print(n_data.shape)
# # n_batch_num=int(n_data.shape[0]/56)
# # print(n_batch_num)
# # r=np.reshape(n_data, [-1,56,10])
# # print(r)
# # r_=tf.convert_to_tensor(r)
# # print(r_.shape)
# # print(sess.run(r_))
# # a=np.array([1,0,1,1,0])
# # b=np.ones([8,3,4])
# # # c=b[a==0]
# # d=np.zeros([8,3,4])
# # e=np.concatenate((b,d),axis=0)
# #
# # a=np.ones(3)
# # n=np.ones(4)
# # # print(np.concatenate([a,n],axis=0).shape)
# #
# # mm=n[:3]
# # print(mm)
# # characters=56
# # times=10
# # def _get_dataset():
# #     n_data = np.loadtxt("train_data_bigan/n",delimiter=' ')
# #     a_data = np.loadtxt("train_data_bigan/a",delimiter=' ')
# #     n_batch_num=int(n_data.shape[0]/characters)
# #     a_batch_num=int(a_data.shape[0]/characters)
# #     n_data=np.reshape(n_data,[-1,characters,times])
# #     a_data=np.reshape(a_data,[-1,characters,times])
# #     n_data_label=np.zeros((n_batch_num),dtype=np.int)
# #     a_data_label=np.ones((a_batch_num),dtype=np.int)
# #
# #     data=np.concatenate([n_data,a_data],axis=0)
# #     data_label=np.concatenate([n_data_label,a_data_label],axis=0)
# #
# rng = np.random.RandomState(42)
# #     inds = rng.permutation(data.shape[0])
# #     data=data[inds]
# #     data_label=data_label[inds]
# #
# #     fen=int(data.shape[0]/2)
# #     x_train=data[:fen]
# #     x_test=data[fen:]
# #
# #     print(data.shape)
# #     print(x_train.shape)
# #
# #     y_train=data_label[:fen]
# #     y_test=data_label[fen:]
# #
# #     x_train=x_train[y_train!=1]
# #     y_train=y_train[y_train!=1]
# #
# #     dataset = {}
# #     dataset['x_train'] = x_train.astype(np.float32)
# #     dataset['y_train'] = y_train.astype(np.float32)
# #     dataset['x_test'] = x_test.astype(np.float32)
# #     dataset['y_test'] = y_test.astype(np.float32)
# #
# # _get_dataset()
#
# def attention(inputs, attention_size, time_major=False):
#     """
#     inputs.shape = batch,times,hidden_n
#     """
#     if isinstance(inputs, tuple):
#         # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
#         inputs = tf.concat(inputs, 2)
#
#     if time_major:
#         # (T,B,D) => (B,T,D)
#         inputs = tf.transpose(inputs, [1, 0, 2])
#
#     inputs_shape = inputs.shape
#     sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
#     hidden_size = inputs_shape[2].value  # hidden size of the RNN layer
#
#     # Attention mechanism
#     W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
#     b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#     u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#
#     v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
#     vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
#     exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
#     alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
#
#     # Output of Bi-RNN is reduced with attention vector
#     output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
#     #只返回权值
#     return alphas
#
# # z = tf.constant(5., shape=[4,5,6])
# # out=attention(z,8)
# #
# # sess = tf.Session()
# # sess.run(tf.initialize_all_variables())
# # print(sess.run(out))
# def list_shuffle(x,inds):
#     inds=inds.tolist()
#     x_new=[]
#     for i in inds:
#         x_new+=(x[i * 10:(i + 1) * 10])
#     return x_new
# characters=10
# def _get_dataset(ratio):
#
#     print("ratio!!!!!")
#     print(ratio)
#     file_name='18_pca_bigan_more/'
#     n_data = np.loadtxt(file_name+"n_pca_transpose", delimiter=' ')
#     a_data = np.loadtxt(file_name+"a_pca_transpose", delimiter=' ')
#
#     n_ip = np.loadtxt(file_name+"n_ip", delimiter=' ',dtype=str)
#     a_ip = np.loadtxt(file_name+"a_ip", delimiter=' ',dtype=str)
#
#     n_label = np.loadtxt(file_name+"n_label", delimiter=' ')
#     a_label = np.loadtxt(file_name+"a_label", delimiter=' ')
#
#     # n_slot_attack = np.loadtxt("18_pca_bigan/n_slot_attack", dtype=bytes).astype(str)
#     # a_slot_attack = np.loadtxt("18_pca_bigan/a_slot_attack", dtype=bytes).astype(str)
#     n_slot_attack = open(file_name+"n_slot_attack").readlines()
#     # a_slot_desip = np.loadtxt("98_pca_bigan/a_slot_desip", dtype=bytes).astype(str)
#     a_slot_attack = open(file_name+"a_slot_attack").readlines()
#     # print("slot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111")
#     # print(a_slot_attack)
#     # print(a_slot_attack.shape)
#     # n_slot_desip = np.loadtxt("98_pca_bigan/n_slot_desip", dtype=bytes).astype(str)
#     n_slot_desip = open(file_name+"n_slot_desip").readlines()
#     # a_slot_desip = np.loadtxt("98_pca_bigan/a_slot_desip", dtype=bytes).astype(str)
#     a_slot_desip = open(file_name+"a_slot_desip").readlines()
#     # print("slot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # print(a_slot_desip.shape)
#     a_attack =np.loadtxt(file_name+"a_attack",dtype=int) #获取是否有某类异常的全部标签
# #delimiter="\n"
#     n_data=np.reshape(n_data,[-1,1,characters])
#     a_data=np.reshape(a_data,[-1,1,characters])
#
#     trian_ratio=0.6  #训练集所占的比例
#
#     rng = np.random.RandomState(42)
#
#     inds = rng.permutation(n_data.shape[0])
#     n_data=n_data[inds]
#     n_label=n_label[inds]
#     n_ip=n_ip[inds]
#     n_slot_attack=list_shuffle(n_slot_attack,inds)
#     n_slot_desip=list_shuffle(n_slot_desip,inds)
#     fen_n=int(n_data.shape[0]*trian_ratio)
#     n_data_train=n_data[:fen_n]
#     n_data_test=n_data[fen_n:]
#     n_label_train=n_label[:fen_n]
#     n_label_test=n_label[fen_n:]
#     n_ip_train=n_ip[:fen_n]
#     n_ip_test=n_ip[fen_n:]
#     n_slot_attack_train=n_slot_attack[:fen_n]
#     n_slot_attack_test=n_slot_attack[fen_n:]
#     n_slot_desip_train = n_slot_desip[:fen_n]
#     n_slot_desip_test = n_slot_desip[fen_n:]
#     n_data_train_label=np.zeros((n_data_train.shape[0]),dtype=np.int)
#     n_data_test_label=np.zeros((n_data_test.shape[0]),dtype=np.int)
#
#     inds = rng.permutation(a_data.shape[0])
#     a_data=a_data[inds]
#     a_label=a_label[inds]
#     a_ip=a_ip[inds]
#     a_attack=a_attack[inds]
#     # print(len(a_slot_attack))
#     a_slot_attack = list_shuffle(a_slot_attack,inds)
#     print(len(a_slot_attack))
#     print("slot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111")
#     print(a_slot_attack)
#
#     a_slot_desip =list_shuffle(a_slot_desip,inds)
#     fen_a=int(a_data.shape[0]*trian_ratio)
#     a_data_train=a_data[:fen_a]
#     a_data_test=a_data[fen_a:]
#     a_label_train=a_label[:fen_a]
#     a_label_test=a_label[fen_a:]
#     a_ip_train=a_ip[:fen_a]
#     a_ip_test=a_ip[fen_a:]
#     a_attack_train=a_attack[:fen_a]
#     a_attack_test=a_attack[fen_a:]
#     a_slot_attack_train = a_slot_attack[:fen_a]
#     a_slot_attack_test = a_slot_attack[fen_a:]
#     a_slot_desip_train = a_slot_desip[:fen_a]
#     a_slot_desip_test = a_slot_desip[fen_a:]
#     a_data_train_label = np.ones((a_data_train.shape[0]), dtype=np.int)
#     a_data_test_label = np.ones((a_data_test.shape[0]), dtype=np.int)
#
#     data_test=np.concatenate([n_data_test,a_data_test],axis=0)
#     label_test=np.concatenate([n_label_test,a_label_test],axis=0)
#     ip_test=np.concatenate([n_ip_test,a_ip_test],axis=0)
#     data_test_label=np.concatenate([n_data_test_label,a_data_test_label],axis=0)
#     slot_attack_test=n_slot_attack_test+a_slot_attack_test
#     # slot_attack_test=np.concatenate([n_slot_attack_test,a_slot_attack_test],axis=0)
#     # slot_desip_test =np.concatenate([n_slot_desip_test,a_slot_desip_test],axis=0)
#     slot_desip_test= n_slot_desip_test+a_slot_desip_test
#
#     #在这里随机的抽取一定比例的异常到正常中,dont forget put a_label_train away
#     x_train=n_data_train
#     y_train=n_data_train_label
#     x_train_a=a_data_train
#     y_train_a=a_data_train_label
#
#
#     #随机抽10%的异常到正常中
#     if(ratio==2):
#         # #将某种异常的全部标签去除
#         x_train_a_qu = x_train_a[a_attack_train == 1]
#         a_label_train_qu = a_label_train[a_attack_train == 1]
#         x_train_a = x_train_a[a_attack_train == 0]
#         a_label_train = a_label_train[a_attack_train == 0]
#         x_train = np.concatenate([x_train, x_train_a_qu], axis=0)
#         n_label_train = np.concatenate([n_label_train, a_label_train_qu], axis=0)
#
#     else:
#         num = int(x_train_a.shape[0] * ratio)
#         p_inds = np.random.choice(x_train_a.shape[0], num, replace=False)
#         x_train_a_ratio = x_train_a[p_inds]
#         a_label_train_ratio = a_label_train[p_inds]
#
#         x_train_a = np.delete(x_train_a, p_inds, axis=0)
#         a_label_train = np.delete(a_label_train, p_inds, axis=0)
#
#         x_train = np.concatenate([x_train, x_train_a_ratio], axis=0)
#         n_label_train = np.concatenate([n_label_train, a_label_train_ratio], axis=0)
#
#
#     # print("normal num:")
#     # print(x_train.shape[0])
#     # print("anomaly num:")
#     # print(x_train_a.shape[0])
#
#
#
#     x_test=data_test
#     label=label_test
#     ip=ip_test
#     y_test=data_test_label
#
#
#     dataset = {}
#     dataset['x_train'] = x_train.astype(np.float32)
#     dataset['y_train'] = y_train.astype(np.float32)
#     dataset['x_test'] = x_test.astype(np.float32)
#     dataset['y_test'] = y_test.astype(np.float32)
#     dataset['x_train_a']=x_train_a.astype(np.float32)
#     dataset['a_label_train'] = a_label_train
#     dataset['y_train_a']=y_train_a.astype(np.float32)
#     dataset['label']=label
#     dataset['ip']=ip
#     dataset['slot_attack']=slot_attack_test
#     dataset['slot_desip'] =slot_desip_test
#
#     print(len(slot_attack_test))
#     print(ip.shape[0])
#
#     return  dataset
#
# a=_get_dataset(0)
# # print(len(a['slot_attack']))
# # print(a["slot_attack"])
# # print(a['y_test'])
# # print(len(a['y_test']))
#
# # def fizzbuzz(num):
# #     if num % 3 == 0 and num % 5 == 0:
# #         print('FizzBuzz')
# #     elif num % 3 == 0:
# #         print('Fizz')
# #     elif num % 5 == 0:
# #         print('Buzz')
# #     else:
# #         print(num)
# #     return num
# #
# #
# # num = tf.placeholder(tf.int32)
# # print("hello")
# # result = fizzbuzz(num)
# # with tf.Session() as sess:
# #     for n in range(10,16):
# #         sess.run(result, feed_dict={num:n})
# #     a=1
# #     print(a)
# # l_generator=tf.constant([0.3,0.4,0.5,0.6])
# # l_generator_1=tf.constant([0.4,0.5,0.6,0.7])
# # res=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator)
# #
# # res_1=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator-1), logits=l_generator_1)
# # x_inp=tf.constant([[[-9.700762822484452386e+01,-9.497358832370281334e+01 ,-1.042463278664764630e+02, 2.517128340231348034e+02, -9.942081528212382580e+01 ,-8.532448543099381766e+01, -1.042458971013734725e+02 ,-8.896553801602645706e+01 ,-8.287322890816716381e+01, -1.026378236716922316e+02]]])
# # x_prior = tf.unstack(x_inp, 10, 2)  #
# # lstm_cell = tf.contrib.rnn.MultiRNNCell(
# #     [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(1,forget_bias=2.0)) for _ in range(2)])
# # res, states = tf.contrib.rnn.static_rnn(lstm_cell, x_prior, dtype=tf.float32)
# # tensor_a = tf.convert_to_tensor(res)
# # tensor_b = tf.transpose(tensor_a, perm=[1, 2, 0])
# # o=attention(z,12)
# # x_prior = tf.unstack(z, 56, 2);  #
# # lstm_cell = tf.contrib.rnn.MultiRNNCell(
# #     [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(1)) for _ in range(2)]);
# # res, states = tf.contrib.rnn.static_rnn(lstm_cell, x_prior, dtype=tf.float32);
# # tensor_a = tf.convert_to_tensor(z)
# # tensor_b = tf.reshape(tensor_a,[-1,1,56])
# # tensor_c= tf.squeeze(tensor_b)
# # x_prior = tf.unstack(z, 10, 2);
# # lstm_cell = tf.contrib.rnn.MultiRNNCell(
# #                 [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(36)) for _ in range(2)]);
# # res, states = tf.contrib.rnn.static_rnn(lstm_cell, x_prior, dtype=tf.float32);
# # tensor_a = tf.convert_to_tensor(res)
# # o=attention(z,25)
# # o1=attention(z,25)
# # sess = tf.Session()
# # # o2=tf.concat([o,o1],axis=1)
# # sess.run(tf.initialize_all_variables())
# # # print(sess.run(res))
# # # print(sess.run(res_1))
# # # print(o.shape)
# # # print(o1.shape)
# # # print(o2.shape)
# #
# # print(sess.run(tensor_b))
# # f=open('feature_test')
# # new_data=""
# # for line in f.readlines():
# #     attack=line.strip().split()[0]
# #     num=line.strip().split()[1]
# #     num=int(num)
# #     new_data+=attack+"        "+str(num)+"\n"
# # w=open('feature_test','w')
# # w.write(new_data)
#
# z = tf.constant(5., shape=[13,10,1])
# o=attention(z,12)
#
# # n_slot_attack = np.loadtxt("98_pca_bigan/a_slot_desip",delimiter='\n',dtype=bytes).astype(str)
# # f=open("98_pca_bigan/a_slot_desip")
# # n_slot_attack=f.readlines()
# # print(n_slot_attack)
# # # print(n_slot_attack[904])
# # # n_slot_attack=np.array(n_slot_attack)
# # # print(n_slot_attack.shape)
# # a=[1,2,3,4,5]
# # aa=np.array(a)
# # print(aa)
# # cc=aa.tolist()
# # print(cc)
# # inds=[4,0,2,1,3]
# # b=[]
# # print(a)
# # print(98232*30/149)
# inds = rng.permutation(5)
# print(inds)
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# # print(o.shape)
# # for i in range(4):
# #     print(i)
# # max_test_label=np.ones(shape=[10,5])
# # a= np.zeros(shape=[10,5])
# # max_test_label=a
# # print(max_test_label)
# # a_ip = np.loadtxt("98_pca_bigan/a_ip", delimiter=' ',dtype=str)
# # print(o.shape)
# # a=np.random.choice(5, 3,replace=False)
# # print(a)
# # print()
# # x_train_a=np.ones(shape=[120,2,4])
# # ratio = 0.1
# # num = int(x_train_a.shape[0] * ratio)
# # print(num)
# # p_inds = np.random.choice(x_train_a.shape[0], num,replace=False)
# # x_train_a_ratio = x_train_a[p_inds]
# # print(x_train_a_ratio.shape)
# #
# # x_train_a=np.delete(x_train_a,p_inds,axis=0)
# # print(x_train_a.shape)
# # file_name= '98_pca_bigan/'
# # n_slot_attack = open(file_name+"a_slot_attack").readlines()
# # f=open('slot_test','w')
# # for i in n_slot_attack:
# #     f.write(i)
# # print(n_slot_attack)
a=0.845409
b=0.845409
print(2*a*b/(a+b))





