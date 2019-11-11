from scipy.sparse import csr
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm


def vectorize_dic(dic, ix=None, p=None, n=0, g=0):
    """
    dic -- dictionary of feature lists. Keys are the name of features 数据表示：字典类型，key：特征名称；value表示值
    ix -- index generator (default None)
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    n -- 
    g -- 
    """
    if ix == None:
        ix = dict()

    nz = n * g

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k), 0) + 1
            col_ix[i+t*g] = ix[str(lis[t]) + str(k)]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)
    if p == None:
        p = len(ix)

    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix

def batcher(X, y, batch_size=-1):
    n_samples = X.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size <= 0:# not equals -1
        raise ValueError("Parameter batch_size={} isnt supported.".format(batch_size))

    for i in range(0, n_samples, batch_size):
        end = min(i+batch_size, n_samples)

        ret_X = X[i: end]
        ret_Y = y[i: end]

        yield (ret_X, ret_Y)

### 1. 数据处理
cols = ['UserID','MovieID','Rating','Timestamp']

train_path = 'data/ua.base'
test_path = 'data/ua.test'

train = pd.read_csv(train_path, sep='\t', names=cols) # dataframe
test = pd.read_csv(test_path, sep='\t', names=cols)

'''
# to numpy
x_train, ix = vectorize_dic({'Users': train['UserID'], 'Movies': train['MovieID']}, n=len(train),g=2)
# ix: 保证x_train, x_test 特征维度相同
x_test, ix = vectorize_dic({'Users': test['UserID'], "Movies": test['MovieID']}, ix, x_train.shape[1], n=len(test), g=2)

x_train = x_train.todense()
x_test = x_test.todense()

y_train = train['Rating'].values
y_test = test['Rating'].values
'''

x_train = train[['UserID','MovieID']].values
x_test = test[['UserID','MovieID']].values
y_train = train['Rating'].values
y_test = test['Rating'].values

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
### 模型:input, variables, model, loss, optimizer

# 0. some hyper-parameters & constant params
N, P = x_train.shape
K = 10

epochs = 10
batch_size = 1000

# 1.input

X = tf.placeholder(tf.float32, [None, P])
y = tf.placeholder(tf.float32, [None, 1])

# 2. variables
w0 = tf.Variable(tf.zeros([1]))
# w1 = tf.Variable(tf.truncated_normal([P],mean=0.0,stddev=0.1))
w1 = tf.Variable(tf.zeros([P]))
V = tf.Variable(tf.random_normal([P,K], mean=0.0, stddev=0.01))

# 3. model
## lr部分, return shape: [N, 1]
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w1, X), axis=1, keep_dims=True))

## FM 二阶交叉部分
### embeddings
# embeddings = tf.matmul(X, V)# N * K
# N * 1
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(tf.matmul(X, V), 2),# N * K
        tf.matmul(tf.pow(X,2), tf.pow(V, 2)),# N * K
    ),axis=1, keep_dims=True)

y_hat = tf.add(linear_terms, pair_interactions)

# 4. loss

## l2正则
lambda_w1 = tf.constant(0.001, name='lambda_w1')
lambda_V = tf.constant(0.001, name='lambda_V')

# l2_norm = tf.add(lambda_w1 * tf.reduce_sum(tf.pow(w1, 2)) , lambda_V * tf.reduce_sum(tf.pow(V, 2)))

l2_norm = tf.reduce_sum(  # 应用了广播机制
    tf.add(
        tf.multiply(lambda_w1, tf.pow(w1, 2)),
        tf.multiply(lambda_V, tf.transpose(tf.pow(V, 2)))
    )
)


error = tf.reduce_mean(tf.square(y-y_hat))#MSE
loss = tf.add(error, l2_norm)

# 5. optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#### train

# 1. 变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs)):
        perm = np.random.permutation(x_train.shape[0])
        # print(perm)
        for batch_X, batch_y in batcher(x_train[perm], y_train[perm],batch_size=batch_size):
            _, t = sess.run([train_op, loss], feed_dict={X: batch_X.reshape(-1, P), y: batch_y.reshape(-1, 1)})
            # print(t)
        
        errors = []
        for batch_X, batch_y in batcher(x_test, y_test):
            errors.append(sess.run(error, feed_dict={X: batch_X.reshape(-1, P),y: batch_y.reshape(-1, 1)}))
        rmse = np.sqrt(np.array(errors).mean())
        # print(rmse)


