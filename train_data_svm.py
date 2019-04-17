import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.utils import shuffle
character=56
time_windows=10
def get_num(s):
    f=open(s)
    i=0
    for line in f.readlines():
        i+=1
    return i

#实质为子矩阵的转置运算
def data_to_svm(name):#name : a or n
    num=get_num("train_data/"+name+"_label")
    print(name+"  :"+str(num))
    data=np.genfromtxt("train_data/"+name)
    label=np.genfromtxt("train_data/"+name+"_label")
    data_result=np.zeros(shape=(time_windows*num,character))
    label_result=np.zeros(shape=(time_windows*num))
    """
    w_data=open('train_data_svm/'+name,"w")
    w_label=open("train_data_svm/"+name+"_label","w")
    """
    for i in range(0,num):
        data_tem=data[character*i:character*(i+1)]
        label_tem=label[i]
        data_transpose=np.transpose(data_tem)
        label_transpose=np.transpose(label_tem)
        data_result[i*time_windows:(i+1)*time_windows]=data_transpose
        label_result[i*time_windows:(i+1)*time_windows]=label_transpose

    np.savetxt('train_data_svm/'+name,data_result,fmt="%d")
    np.savetxt('train_data_svm/' + name+"_label", label_result,fmt="%d")
#把a和n文件汇为一个
def to_xy():
    """
    x文件中的数据每一行表示一个小时间窗口的的56个特征，y文件中对应行表示对应标签
    """
    w_data = open('train_data_svm/x', "w")
    w_label = open("train_data_svm/y", "w")
    f1=open('train_data_svm/a')
    for line in f1.readlines():
        w_data.write(line)
    f2 = open('train_data_svm/n')
    for line in f2.readlines():
        w_data.write(line)
    f3 = open('train_data_svm/a_label')
    for line in f3.readlines():
        if(line.strip()=="0"):
            w_label.write("-1\n")
        else:w_label.write(line)
    f4 = open('train_data_svm/n_label')
    for line in f4.readlines():
        if (line.strip() == "0"):
            w_label.write("-1\n")
        else:
            w_label.write(line)


"""
#将数据转换为svm能接收的形式
data_to_svm("a")
data_to_svm("n")
to_xy()
"""


def svm_get_data():
    """
    对x,y数据进行svm
    """
    list1 = []
    for i in range(1, 57):
        list1.append(str(i))
    x=pd.read_table("train_data_svm/x",header=None,names=list1,sep=" ")#注意这里read_csv的默认分隔符是，。导致出错！
    y=pd.read_csv("train_data_svm/y",header=None)
    print(x.head(5))
    y.columns=["label"]
    x_n=x[y.label==-1]
    x_a=x[y.label==1]
    y_n=y[y.label==-1]
    y_a=y[y.label==1]

    a=x_a.join(y_a)
    n=x_n.join(y_n)
    print(a.columns)
    print(a.head(5))

    a_train=a.sample(frac=0.5, random_state=42)
    a_test=a.loc[~a.index.isin(a_train.index)]
    n_train=n.sample(frac=0.5, random_state=42)
    n_test=n.loc[~n.index.isin(n_train.index)]

    train=pd.concat([a_train,n_train],ignore_index=True)
    print(train.columns)
    test=pd.concat([a_test,n_test],ignore_index=True)
    print(test.columns)
    train=shuffle(train)
    test=shuffle(test)

    x_train,y_train=_to_xy(train,"label")
    y_train = y_train.flatten().astype(int)
    x_test,y_test=_to_xy(test,"label")
    y_test = y_test.flatten().astype(int)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    return dataset

def _to_xy(df, target):
    # Converts a Pandas dataframe to the x,y inputs that TensorFlow needs
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

def svm():
    data = svm_get_data()
    batch_size = 100
    x = tf.placeholder(shape=[None, character], dtype=tf.float32)
    y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[character, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    model_output = tf.subtract(tf.matmul(x, A), b)
    l2_norm = tf.reduce_sum(tf.square(A))
    alpha = tf.constant([0.01])
    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

    prediction = tf.sign(model_output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Training loop
    loss_vec = []
    train_accuracy = []
    test_accuracy = []
    f = open('train_data_svm/exactSVM1.txt', 'w')
    for i in tqdm(range(10000)):  # 进度提示
        rand_index = np.random.choice(len(data['x_train']), size=batch_size)
        rand_x = data['x_train'][rand_index]
        rand_y = np.transpose([data['y_train'][rand_index]])
        sess.run(train_step, feed_dict={x: rand_x, y: rand_y})
        temp_loss = sess.run(loss, feed_dict={x: rand_x, y: rand_y})
        loss_vec.append(temp_loss)
        train_acc_temp = sess.run(accuracy, feed_dict={x: data['x_train'], y: np.transpose([data['y_train']])})
        train_accuracy.append(train_acc_temp)
        test_acc_temp = sess.run(accuracy, feed_dict={x: data['x_test'], y: np.transpose([data['y_test']])})
        test_accuracy.append(test_acc_temp)
        pred = sess.run(prediction, feed_dict={x: data['x_test'], y: np.transpose([data['y_test']])}).tolist()
        if (i + 1) % 100 == 0:
            f.write('step # ' + str(i + 1) + 'A=' + str(sess.run(A)) + 'b=' + str(sess.run(b)) + '\n')
            f.write('Loss = ' + str(temp_loss) + '\n')
            f.write('accuracy=' + str(train_acc_temp) + ',' + str(test_acc_temp) + '\n')
        if (i + 1) % 1000 == 0:
            f.write('exact_pred:' + str(pred) + '\n')
            f.write('exact:' + str(np.transpose([data['y_test']]).tolist()) + '\n')
    f.close()

svm()

