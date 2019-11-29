import numpy as np
import tensorflow as tf

import math
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


class DCN(BaseEstimator, TransformerMixin):
    def __init__(self, cate_feature_size, field_size, numeric_feature_size, \
        embedding_size=8, deep_layers=[32,32], dropout_deep=[0.5,0.5],\
            deep_layers_activation=tf.nn.relu,
            cross_layer_num=3,epoch=10,batch_size=256,learning_rate=1e-3,optimizer_type='adam',
            batch_norm=0,batch_norm_decay=0.995,loss_type="logloss",l2_reg=0.0,
            eval_metric=roc_auc_score,greater_is_better=True,
            verbose=False,random_seed=2016):
        assert loss_type in ['mse', 'logloss'], "loss_type can be either 'logloss' or 'mse'"

        self.cate_feature_size = cate_feature_size # after onehot-encoding
        self.numeric_feature_size = numeric_feature_size
        self.field_size = field_size # number of sparse features
        self.embedding_size = embedding_size
        self.total_size = self.field_size * self.embedding_size + \
            self.numeric_feature_size  # 有一点问题
        
        self.deep_layers = deep_layers
        self.deep_layers_activation = deep_layers_activation
        
        self.cross_layer_num = cross_layer_num

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_deep = dropout_deep
        
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better

        
        self.verbose = verbose
        self.random_seed = random_seed

        self.train_result, self.eval_result = [], []

        self._init_graph()
    
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # 1. for input data
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
            
            self.numeric_value = tf.placeholder(tf.float32, shape=[None, None], name='numeric_value')

            self.label = tf.placeholder(tf.float32, shape=[None,1], name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            # 2. weights
            self.weights = self._initialize_weights()

            # 3. define the computing graph
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            self.x0 = tf.concat([self.numeric_value, 
                                tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])],
                                axis=1)

            with tf.name_scope('deep_part'):
                self.y_deep = self.x0 # x0需要进行dropout?

                for i in range(len(self.deep_layers)):
                    self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['deep_layer_%d' % i ]),
                                            self.weights['deep_bias_%d' % i])
                    self.y_deep = self.deep_layers_activation(self.y_deep)
                    self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i]) # 
            
            with tf.name_scope('cross_part'):
                self._x0 = tf.reshape(self.x0, (-1, self.total_size, 1))
                x_l = self._x0

                for l in range(self.cross_layer_num):
                    # cross_layer: x_l+1 = x_0 * x_l.T * W_l + b_l + x_l
                    # 默认为列向量
                    # x_l = tf.tensordot(tf.matmul(self._x0, x_l, transpose_b=True),
                                    # self.weights['cross_layer_%d' % l], 1) + self.weights['cross_bias_%d' % l] + x_l
                    # 上式可以简化运算,先算x_l.T * w_l 得到一个实数,然后再和前面的x_0进行计算
                    xlT_w = tf.tensordot(x_l, self.weights['cross_layer_%d' % l],axes=(1,0))
                    print(xlT_w)
                    print(self.weights['cross_layer_%d' % l])
                    x_l = tf.matmul(self._x0, xlT_w) + \
                        self.weights['cross_bias_%d' % l] + x_l
                
                self.cross_part_out = tf.reshape(x_l, (-1, self.total_size))
            
            ## concat_part
            concat_input = tf.concat([self.cross_part_out, self.y_deep], axis=1)
            ## last layer
            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

            # 4. loss function
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.substract(self.label, self.out))
            
            ## regularization
            if self.l2_reg > 0.0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_projection'])

                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['deep_layer_%d' % i])
                
                for i in range(self.cross_layer_num):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['cross_layer_%d' % i])
            
            # 5 optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,initial_accumulator=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=0.5).minimize(self.loss)
            

            # init
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            ## 算一下参数数量
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.shape
                temp = 1
                for dim in shape:
                    temp *= dim
                total_parameters += temp
            
            if self.verbose > 0:
                print("#params: %d" % total_parameters)
    
    def _initialize_weights(self):
        weights = dict()

        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.cate_feature_size, self.embedding_size], 0.0, 0.1),
            name='feature_embeddings'
        )
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.cate_feature_size, 1],0.0, 0.1), name='feature_bias')

        # deep_layers
        num_layer = len(self.deep_layers)
        glorot = np.sqrt(2.0/(self.total_size + self.deep_layers[0]))

        weights['deep_layer_0'] = tf.Variable(
            tf.random_normal([self.total_size, self.deep_layers[0]], 0.0, glorot)
        )
        weights['deep_bias_0'] = tf.Variable(tf.random_normal([1, self.deep_layers[0]], 0.0, glorot))

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0/(self.deep_layers[i-1] + self.deep_layers[i]))
            weights['deep_layer_%d' % i] = tf.Variable(
                tf.random_normal([self.deep_layers[i-1], self.deep_layers[i]], 0.0, glorot)
            )
            weights['deep_bias_%d' % i] = tf.Variable(
                tf.random_normal([1, self.deep_layers[i]], 0.0, glorot)
            )
        
        glorot = np.sqrt(2.0/(self.total_size*2)) # some problem 
        for i in range(self.cross_layer_num):
            weights['cross_layer_%d' % i] = tf.Variable(
                tf.random_normal([self.total_size, 1], 0.0, glorot)
            )
            weights['cross_bias_%d' % i] = tf.Variable(
                tf.random_normal([self.total_size, 1], 0.0, glorot)
            )
        
        # last layer
        input_size = self.total_size + self.deep_layers[-1]
        glorot = np.sqrt(2.0/input_size)
        weights['concat_projection'] = tf.Variable(
            tf.random_normal([input_size, 1], mean=0.0, stddev=glorot),dtype=tf.float32
        )
        weights['concat_bias'] = tf.Variable(
            tf.constant(0.01),
            dtype=tf.float32
        )

        return weights
    
    def get_batch(self, Xi, Xv, Xv2, y, batch_size, index):
        start = batch_size * index
        end = min(start + batch_size, len(y))

        return Xi[start: end], Xv[start:end], Xv2[start:end], [[y_] for y_ in y[start:end]]

    def shuffle_in_unison_scary(self, a, b, c, d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)

        np.random.set_state(rng_state)
        np.random.shuffle(b)

        np.random.set_state(rng_state)
        np.random.shuffle(c)

        np.random.set_state(rng_state)
        np.random.shuffle(d)
    
    def evaluate(self, Xi, Xv, Xv2, y):### 并不是真正的predict
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.numeric_value: Xv2,
            self.label: y,
            self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
            self.train_phase: False,
        }
        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss
    def predict(self, Xi, Xv, Xv2):
        dummy_y = [1] * len(Xi)
        batch_index = 0

        Xi_batch, Xv_batch, Xv2_batch, y_batch = self.get_batch(Xi,Xv,Xv2,dummy_y, self.batch_size, batch_index)
        
        y_pred = None
        while len(Xi_batch) > 0:
            feed_dict = {
                self.feat_index: Xi_batch,
                self.feat_value: Xv_batch,
                self.numeric_value: Xv2_batch,
                self.label: y_batch,
                self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                self.train_phase: False,
            }

            batch_out = self.sess.run([self.out], feed_dict=feed_dict)

            cur_batch_size = len(y_batch)# 因为样本数量不一定能整除batch_size

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (cur_batch_size, ))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (cur_batch_size,))))
            
            batch_index += 1
            Xi_batch, Xv_batch, Xv2_batch, y_batch = self.get_batch(Xi,Xv,xv2,dummy_y, self.batch_size, batch_index)
        
        return y_pred


    def fit_on_batch(self, Xi, Xv, Xv2, y):
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.numeric_value: Xv2,
            self.label: y,
            self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
            self.train_phase: True,
        }

        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, cate_Xi_train,cate_Xv_train, numeric_Xv2_train,y_train,
        cate_Xi_valid=None, cate_Xv_valid=None, numeric_Xv2_valid=None, y_valid=None):
        has_valid = cate_Xi_valid is not None

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(cate_Xi_train,cate_Xv_train, numeric_Xv2_train, y_train)
            total_batch = int(math.ceil(len(y_train) / self.batch_size))

            for i in range(total_batch):
                cate_Xi_batch, cate_Xv_batch, numeric_Xv2_batch, y_batch = self.get_batch(cate_Xi_train, cate_Xv_train,numeric_Xv2_train, y_train, self.batch_size, i)
                self.fit_on_batch(cate_Xi_batch,cate_Xv_batch,numeric_Xv2_batch,y_batch)

            if has_valid:
                y_valid = np.array(y_valid).reshape((-1, 1))
                loss = self.evaluate(cate_Xi_valid,cate_Xv_valid,numeric_Xv2_valid,y_valid)
                print("epoch ", epoch, "loss=", loss)



