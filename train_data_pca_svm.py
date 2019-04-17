import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
character=56
time_windows=10

def loaddata(datafile):
    list1 = []
    for i in range(1, character+1):
        list1.append(str(i))
    return np.array(pd.read_csv(datafile,sep=" ",header=None,names=list1).astype(np.float))

def pca(name,line_num):#name = a or n 及其对应的行数

    datafile1="train_data_svm/"+name
    pca = PCA(n_components=1)
    res=pca.fit_transform(loaddata(datafile1))
    print(res)
    array_res=np.array(res)
    res_line=int(line_num/time_windows)#结果的行数
    name_result=np.zeros(shape=(res_line,time_windows))
    for i in range(0,res_line):
        tem=array_res[i*time_windows:(i+1)*time_windows]
        tem_transpose=np.transpose(tem)
        name_result[i]=tem_transpose
    np.savetxt('train_data_svm/' + name+"_pca_transpose", name_result)

"""
pca("a",273090)
pca("n",82970)
"""
def to_xy():
    """
        x文件中的数据每一行表示pca提取之后的10个时间窗口的特征值（10维），y文件中对应行表示对应标签
        """
    w_data = open('train_data_svm/x_pca_transpose', "w")
    w_label = open("train_data_svm/y_pca_transpose", "w")
    f1 = open('train_data_svm/a_pca_transpose')
    for line in f1.readlines():
        w_data.write(line)
        w_label.write("1\n")
    f1 = open('train_data_svm/n_pca_transpose')
    for line in f1.readlines():
        w_data.write(line)
        w_label.write("-1\n")
#to_xy()


def svm_get_data():
    """
    对x,y数据进行svm
    """
    list1 = []
    for i in range(1, 11):
        list1.append(str(i))
    x=pd.read_table("train_data_svm/x_pca_transpose",header=None,names=list1,sep=" ")#注意这里read_csv的默认分隔符是，。导致出错！
    y=pd.read_csv("train_data_svm/y_pca_transpose",header=None)
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

    a_train=a.sample(frac=0.6, random_state=42)
    a_test=a.loc[~a.index.isin(a_train.index)]
    n_train=n.sample(frac=0.6, random_state=42)
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
    x = tf.placeholder(shape=[None, time_windows], dtype=tf.float32)
    y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[time_windows, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    model_output = tf.subtract(tf.matmul(x, A), b)
    l2_norm = tf.reduce_sum(tf.square(A))
    alpha = tf.constant([0.01])
    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

    prediction = tf.sign(model_output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))
    #计算召回率和precision
    ones=tf.ones_like(y)
    prediction_copy=tf.add(ones,prediction)
    prediction_copy=tf.scalar_mul(0.5,prediction_copy)#先要将所有的-1变回0
    y_copy=tf.add(ones,y)
    y_copy=tf.scalar_mul(0.5,y_copy)

    TP = tf.count_nonzero(prediction_copy * y_copy)
    TN = tf.count_nonzero((prediction_copy - 1) * (y_copy - 1))
    FP = tf.count_nonzero(prediction_copy * (y_copy - 1))
    FN = tf.count_nonzero((prediction_copy - 1) * y_copy)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)


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
    for i in tqdm(range(300)):  # 进度提示
        rand_index = np.random.choice(len(data['x_train']), size=batch_size)
        rand_x = data['x_train'][rand_index]
        rand_y = np.transpose([data['y_train'][rand_index]])
        sess.run(train_step, feed_dict={x: rand_x, y: rand_y})
        temp_loss = sess.run(loss, feed_dict={x: rand_x, y: rand_y})
        loss_vec.append(temp_loss)
        train_acc_temp,train_recall,train_precision,train_f1 = sess.run([accuracy,recall, precision,f1],feed_dict={x: data['x_train'], y: np.transpose([data['y_train']])})
        train_accuracy.append(train_acc_temp)
        test_acc_temp,test_recall,test_precision,test_f1 = sess.run([accuracy,recall,precision,f1],feed_dict={x: data['x_test'], y: np.transpose([data['y_test']])})
        test_accuracy.append(test_acc_temp)
        pred = sess.run(prediction, feed_dict={x: data['x_test'], y: np.transpose([data['y_test']])}).tolist()
        if (i + 1) % 1 == 0:
            f.write('step # ' + str(i + 1) + 'A=' + str(sess.run(A)) + 'b=' + str(sess.run(b)) + '\n')
            f.write('Loss = ' + str(temp_loss) + '\n')
            f.write('accuracy=' + str(train_acc_temp) + ',' + str(test_acc_temp) + '\n')
            f.write('recall=' + str(train_recall) + ',' + str(test_recall) + '\n')
            f.write('precision=' + str(train_precision) + ',' + str(test_precision) + '\n')
            f.write('f1=' + str(train_f1) + ',' + str(test_f1) + '\n')
        """
        if (i + 1) % 1000 == 0:
            f.write('exact_pred:' + str(pred) + '\n')
            f.write('exact:' + str(np.transpose([data['y_test']]).tolist()) + '\n')
        """
    f.close()

svm()

