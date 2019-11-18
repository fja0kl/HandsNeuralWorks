import numpy as np
import tensorflow as tf
from time import time
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

# 继承BaseEstimator,TransformerMixin:使用时更加高效,简洁,fit,transform

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, embedding_size=8, \
        deep_layers=[32,32], deep_layers_activation=tf.nn.relu,\
        dropout_fm=[1.0, 1.0], dropout_deep=[0.5, 0.5, 0.5],
        l2_reg=0.0, batch_norm=0, batch_norm_decay=0.995, 
        use_fm=True, use_deep=True, epoch=10, batch_size=128,
        learning_rate=1e-3, optimizer_type='adam',
        loss_type='logloss', eval_metric=roc_auc_score,
        verbose=2,greater_is_better=True,random_seed=256):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        # deep part
        self.deep_layers = deep_layers
        self.deep_layers_activation = deep_layers_activation

        # for over-fitting
        self.dropout_fm = dropout_fm
        self.dropout_deep = dropout_deep
        self.l2_reg = l2_reg

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        # for architecture
        self.use_fm = use_fm
        self.use_deep = use_deep

        # for training phase
        self.epochs = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better

        self.verbose = verbose
        self.random_seed = random_seed

        self.train_result, self.valid_result = [], []

        # init computing graph
        self._init_graph()
    
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # 0. input placeholder
            self.feat_index = tf.placeholder(tf.int32,[None,None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, [None, None], name='feat_value')
            self.label = tf.placeholder(tf.float32, [None, 1], name='label')

            # training phase hypers
            self.dropout_keep_fm = tf.placeholder(tf.float32, [None], name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32, [None], name='dropout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            # 1. weights assigin
            self.weights = self._initialize_weights()

            # 2. model[computing graph]

            ## a. embedding处理
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value) # vi*xi; None * field_size * K

            with tf.name_scope('fm_part'):
                with tf.name_scope('linear_part'):
                    # None * feature_size * 1
                    self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_linear'], self.feat_index)
                    self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), axis=2)# None * 1
                    # dropout
                    self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])
                with tf.name_scope('interaction_part'):
                    self.summed_features_embedding = tf.reduce_sum(self.embeddings, 1)
                    self.summed_features_embedding_square = tf.square(self.summed_features_embedding)

                    self.squared_features_embedding = tf.square(self.embeddings)# None * field_size * K
                    self.squared_sum_features_embedding = tf.reduce_sum(self.squared_features_embedding, 1)
                    
                    self.y_second_order = 0.5 * tf.subtract(self.summed_features_embedding_square, self.squared_sum_features_embedding)
                    self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])
            
            with tf.name_scope('deep_part'):
                self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
                # self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

                for i in range(len(self.deep_layers)):
                    self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['layer_%d' % i]), self.weights['bias_%d' % i])
                    self.y_deep = self.deep_layers_activation(self.y_deep)
                    # dropout
                    self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i])
            
            # DeepFM
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order],axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            
            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

            # 3. loss function
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.losses.mean_squared_error(self.label, self.out)
            else:
                raise ValueError('loss_type must be in [logloss, mse]')

            ## l2 regularization
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_projection'])

                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['layer_%d' % i])
                if self.use_fm:
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['feature_linear'])
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['feature_embeddings'])
                
            # 4. optimizer
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,\
                                        beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,momentum=0.95).minimize(self.loss)
            
            
            # 5. train
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # for tensorboard 
            writer = tf.summary.FileWriter("logs/", self.sess.graph)  # 用于将汇总数据写入磁盘

            ## number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim
                total_parameters += variable_parameters
            
            if self.verbose > 0:
                print("#params: {}".format(total_parameters))
            
    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size],\
            mean=0.0, stddev=0.01), name='feature_embeddings')
        # linear part of fm
        weights['feature_linear'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 0.01), name='feature_linear')

        # deep part of deepfm
        num_layers = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(tf.random_normal([input_size, self.deep_layers[0]], mean=0.0, stddev=glorot), name='layer_0')
        weights['bias_0'] = tf.Variable(tf.random_normal([1, self.deep_layers[0]]), name='bias_0')

        for i in range(1, num_layers):
            glorot = np.sqrt(2.0/(self.deep_layers[i-1] + self.deep_layers[i]))
            weights['layer_%d' % i] = tf.Variable(
                tf.random_normal([self.deep_layers[i-1], self.deep_layers[i]], mean=0.0, stddev=glorot),
                name='layer_%d' % i
            )
            weights['bias_%d' % i] = tf.Variable(
                tf.random_normal([1, self.deep_layers[i]], mean=0.0, stddev=glorot),
                name='bias_%d' % i
            )
        
        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        
        glorot = np.sqrt(2.0/(input_size + 1))
        weights['concat_projection'] = tf.Variable(
            tf.random_normal([input_size, 1], mean=0.0, stddev=glorot),
            name='concat_projection'
        )
        weights['concat_bias'] = tf.Variable(
            tf.constant(0.001), dtype=tf.float32,
            name='concat_bias'
        )

        return weights
    
    # 待改进,换成yield形式,不用将全部数据读取到memory中
    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)

        return Xi[start:end], Xv[start: end], [[y_] for y_ in y[start: end]]
    
    # shuffle
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)

        np.random.set_state(rng_state)
        np.random.shuffle(b)

        np.random.set_state(rng_state)
        np.random.shuffle(c)
    
    # evaluate
    def evaluate(self, Xi, Xv, y):
        y_pred = self.predict(Xi, Xv)
        
        return self.eval_metric(y, y_pred)
    
    def predict(self, Xi, Xv):
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        y_pred = None
        while len(Xi_batch) > 0 :
            cur_batch_size = len(y_batch)

            feed_dict = {
                self.feat_index: Xi_batch,
                self.feat_value: Xv_batch,
                self.label: y_batch,
                self.dropout_keep_fm: self.dropout_fm,
                self.dropout_keep_deep: self.dropout_deep,
                self.train_phase: False,
            }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            # 处理输出
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (cur_batch_size, ))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (cur_batch_size, ))))
            
            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        
        return y_pred

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False

    # fit_on_batch
    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.label: y,
            self.dropout_keep_fm: self.dropout_fm,
            self.dropout_keep_deep: self.dropout_deep,
            self.train_phase: True,
        }

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss
    
    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        has_valid = Xi_valid is not None

        for epoch in range(self.epochs):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(math.ceil(len(y_train) / float(self.batch_size)))
            
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            
            # evaluate training 
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)

            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train result=%0.4f, valid result=%0.4f [%0.1f s]" % (
                        epoch+1, train_result, valid_result, time()-t1))
                else:
                    print("[%d] train result=%0.4f [%0.1f s]" % (epoch+1, train_result, time()-t1))
            
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break
        
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]

            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid

            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(math.ceil(len(Xi_train) / float(self.batch_size)))

                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break






