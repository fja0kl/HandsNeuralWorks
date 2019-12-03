import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

from time import time
import math

class AFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, embedding_size,
                attention_size=10, deep_layers=[32, 32],
                dropout_deep=[0.5,0.5,0.5],
                deep_layers_activation=tf.nn.relu,
                epoch=10, batch_size=256,
                learning_rate=1e-2, optimizer_type='adam',
                batch_norm=0,batch_norm_decay=0.995,
                verbose=False,random_seed=2017,
                loss_type='logloss',eval_metric=roc_auc_score,
                greater_is_better=True):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        
        self.train_result, self.valid_result = [], []

        self._init_graph()
    
    def _init_graph(self):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # 0. for input data
            self.feat_index = tf.placeholder(
                tf.int32,
                shape=[None, None],
                name='feat_index'
            )
            self.feat_value = tf.placeholder(
                tf.float32,
                shape=[None, None],
                name='feat_value'
            )
            self.label = tf.placeholder(
                tf.float32,
                shape=[None, 1],
                name='label'
            )
            self.dropout_keep_deep = tf.placeholder(
                tf.float32,
                shape=[None],
                name='dropout_keep_deep'
            )
            self.train_phase = tf.placeholder(
                tf.bool,
                name='train_phase'
            )

            # 1. weights
            self.weights = self._initialize_weights()

            # 2.define the computing graph
            
            ## 1. embedding_layer
            # M: feature_size, K: embedding_size, F: field_size, also equals the length of sample, N: sample size; 
            # embedding_lookup: M * K, result: N * F * K 
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.expand_dims(self.feat_value, axis=2)
            # [N, F, K]
            self.embeddings = tf.multiply(self.embeddings, feat_value)# got a vector of vixi

            ## 2. wide part: the linear regression
            self.linear_part = tf.nn.embedding_lookup(self.weights['feature_linear'], self.feat_index)
            self.linear_part = tf.reduce_sum(
                tf.multiply(self.linear_part, feat_value),
                axis=2
            )# N * 1

            ## 3. deep part: the pair-wise interaction layer & attention-based pooling layer

            ### got pair-wise result
            element_pair_wise_product = []
            for i in range(self.field_size):
                for j in range(i+1, self.field_size):
                    element_pair_wise_product.append(
                        tf.multiply(self.embeddings[:,i,:], self.embeddings[:,j,:])
                    )
            
            self.element_wise_product = tf.stack(element_pair_wise_product)# [F(F-1)/2, N, K]
            # [N, F(F-1)/2, K]
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1,0,2],name='element_wise_product')

            ### attention_part: input is the element-wise product of two vectors
            # wx+b --> relu(wx+b) --> h*relu(wx+b)
            num_interactions = int(self.field_size * (self.field_size - 1) / 2)
            self.attention_wx_plus_b = tf.add(
                tf.matmul(
                    tf.reshape(self.element_wise_product,shape=[-1, self.embedding_size]),
                    self.weights['attention_W']),
                self.weights['attention_b']
            )# [N*F*(F-1)/2, attention_size]
            #[N, F(F-1)/2, attention_size]
            self.attention_wx_plus_b = tf.reshape(self.attention_wx_plus_b, shape=[-1, num_interactions, self.attention_size])
            
            # [N, F(F-1),1]
            self.attention_exp = tf.exp(
                tf.reduce_sum(
                    # [N, F*(F-1)/2, attention_size]
                    tf.multiply(tf.nn.relu(self.attention_wx_plus_b),
                                self.weights['attention_h']),
                    axis=2,
                    keepdims=True
                )
            )
            # [N,1,1]
            self.attention_sum = tf.reduce_sum(
                self.attention_exp,
                axis=1,
                keepdims=True
            )
            # [N, F(F-1)/2, 1]
            self.attention_score = tf.div(
                self.attention_exp,
                self.attention_sum,
                name='attention_score'
            )

            ## attention-based pooling:带权和
            # [N, F(F-1)/2, K], [N, F(F-1)/2, 1]
            # wrong
            self.attention_x_product = tf.multiply(self.element_wise_product, self.attention_score)
            print(self.attention_score)
            print(self.element_wise_product)
            # 和上式子等效--发生广播
            # self.attention_x_product = tf.multiply(self.attention_score, self.element_wise_product)
            
            # [N, K]
            self.attention_x_product = tf.reduce_sum(self.attention_x_product, axis=1, name='afm')
            # [N, 1]
            self.attention_part_sum = tf.matmul(self.attention_x_product, self.weights['attention_p'])

            self.y_bias = self.weights['feature_bias'] * tf.ones_like(self.label)

            # add_n:参数是一个列表,列表中每个元素shape都相同;add双元操作符,a,b的shape可以不同,发生广播
            self.out = tf.add_n(
                [tf.reduce_sum(self.linear_part, axis=1, keep_dims=True),
                self.attention_part_sum,
                self.y_bias,],
                name='out_afm'
            )

            # 3.loss function
            if self.loss_type == 'logloss':
                self.out = tf.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            
            # 4. optimizer method
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # 5. train phase
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            total_parameters = 0
            for variable in self.weights.values():
                temp = 1
                shape = variable.get_shape()
                for dim in shape:
                    temp *= dim
                total_parameters += temp
            if self.verbose:
                print("Total parameters:#{}".format(total_parameters))

    def _initialize_weights(self):
        weights = dict()

        # embedding
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings'
        )
        # linear regression part
        weights['feature_linear'] = tf.Variable(# 列向量
            tf.random_normal([self.feature_size, 1], 0.0, 0.01),
            name='feature_linear'
        )
        weights['feature_bias'] = tf.Variable(
            tf.constant(0.1),
            name='feature_bias'
        )
        # attentional layer
        glorot = np.sqrt(2.0/(self.attention_size + self.embedding_size))
        weights['attention_W'] = tf.Variable(
            tf.random_normal([self.embedding_size, self.attention_size], 0.0, glorot),
            name='attention_W'
        )
        weights['attention_b'] = tf.Variable(
            tf.random_normal([self.attention_size, ], mean=0.0, stddev=glorot),
            name='attention_b'
        )
        weights['attention_h'] = tf.Variable(
            tf.random_normal([self.attention_size, ], mean=0.0, stddev=glorot),
            name='attention_h'
        )

        # cross layer?
        weights['attention_p'] = tf.Variable(
            tf.ones([self.embedding_size, 1]),
            name='attention_p'
        )

        return weights
    
    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = min((index + 1) * batch_size, len(y))

        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)

        np.random.set_state(rng_state)
        np.random.shuffle(b)

        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def predict(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.label: y,
            self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
            self.train_phase: True
        }

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.label: y,
            self.dropout_keep_deep: self.dropout_deep,
            self.train_phase: True
        }

        loss, opt = self.sess.run(
            [self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):

        has_valid = Xv_valid is not None

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(
                    Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            if has_valid:
                y_valid = np.array(y_valid).reshape((-1, 1))
                loss = self.predict(Xi_valid, Xv_valid, y_valid)
                print("epoch", epoch, "loss", loss)
