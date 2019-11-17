import tensorflow as tf
import numpy as np
import pandas as pd
import os

INPUT_X_SIZE = 20
FIELD_SIZE = 2
VECTOR_DIMENSION = 3

TOTAL_TRAIN_STEPS = 100

BATCH_SIZE = 1
ALL_DATA_SIZE = 1000

LR = 0.01


def gen_data():
    labels = [-1, 1]
    y = [np.random.choice(labels, 1)[0] for _ in range(ALL_DATA_SIZE)]  # 生成对应样本的标签

    x = np.random.randint(0,2, size=(ALL_DATA_SIZE, INPUT_X_SIZE))# 生成样本
    x_field = [i//10 for i in range(INPUT_X_SIZE)] #对应输入维度的对应field编号
    
    return x, x_field, y

def createZeroDimensionWeight():
    weights = tf.truncated_normal([1])
    tf_weights = tf.Variable(weights)

    return tf_weights

def createOneDimensionWeight(INPUT_X_SIZE):
    weights = tf.truncated_normal([INPUT_X_SIZE])
    tf_weights = tf.Variable(weights)

    return tf_weights

def createTwoDimensionWeight(INPUT_X_SIZE,FIELD_SIZE, VECTOR_DIMENSION):
    weights = tf.truncated_normal([INPUT_X_SIZE, FIELD_SIZE, VECTOR_DIMENSION])
    tf_weights = tf.Variable(weights)

    return tf_weights

def inference(INPUT_X, INPUT_FIELD_SIZE, zeroWeights, oneWeights, twoWeights):
    """
    计算模型输出值: lr + FFM
    """
    second_value = tf.reduce_sum(tf.multiply(oneWeights, INPUT_X, name='secondValue'))
    linear_part = tf.add(zeroWeights, second_value, name='linearRegression')

    third_value = tf.Variable(0.0, dtype=tf.float32)
    
    input_shape = INPUT_X_SIZE

    # 计算交叉项部分
    for i in range(INPUT_X_SIZE):
        feature_index1 = i
        field_index1 = int(INPUT_FIELD_SIZE[i])
        for j in range(i+1, INPUT_X_SIZE):
            feature_index2 = j
            field_index2 = int(INPUT_FIELD_SIZE[j])

            # 取对应交叉项的隐向量
            # vector_left = tf.convert_to_tensor([[feature_index1, field_index2, i] for i in range(VECTOR_DIMENSION)])
            # weight_left = tf.gather_nd(twoWeights, vector_left)
            # weight_left_afer_cut = tf.squeeze(weight_left)

            # vector_right = tf.convert_to_tensor([[feature_index2, field_index1, i] for i in range(VECTOR_DIMENSION)])
            # weight_right = tf.gather_nd(twoWeights, vector_right)
            # weight_right_after_cut = tf.squeeze(weight_right)
            # weight_left_after_cut = 

            # 计算隐向量的内积,以及对应特征值的乘积,最后再相乘,累加到third_value上
            # temp_value = tf.reduce_sum(tf.multiply(weight_left_afer_cut, weight_right_after_cut))
            
            # 计算隐向量的内积,以及对应特征值的乘积,最后再相乘,累加到third_value上
            temp_value = tf.reduce_sum(tf.multiply(
                two_weights[feature_index1, field_index2], twoWeights[feature_index2, field_index1]))

            idx1 = [i]
            idx2 = [j]
            xi = tf.squeeze(tf.gather_nd(INPUT_X, idx1))
            xj = tf.squeeze(tf.gather_nd(INPUT_X, idx2))
            
            product = tf.reduce_sum(tf.multiply(xi, xj))

            cross_value_item = tf.multiply(temp_value, product)

            # 累加到third_value上
            tf.assign(third_value, tf.add(third_value, cross_value_item))
        
        return tf.add(linear_part, third_value)
    

if __name__ == "__main__":
    global_step = tf.Variable(0, trainable=False)

    # 0. 数据准备
    x, x_field, y = gen_data()

    # 1. 输入处理
    input_X = tf.placeholder(tf.float32, [INPUT_X_SIZE])
    input_y = tf.placeholder(tf.float32)

    # 2. 权重声明
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    zero_weights = createZeroDimensionWeight()
    one_weights = createOneDimensionWeight(INPUT_X_SIZE)
    two_weights = createTwoDimensionWeight(INPUT_X_SIZE, FIELD_SIZE, VECTOR_DIMENSION)

    # 3. 创建运算图graph(模型)
    y_ = inference(input_X, x_field, zero_weights, one_weights, two_weights)

    # 4. loss函数定义
    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(one_weights, 2)),
            tf.reduce_sum(tf.multiply(lambda_v, tf.pow(two_weights, 2)), axis=[1,2])
        )
    )

    loss = tf.log(1 + tf.exp(-input_y * y_)) + l2_norm # 指数损失

    # 4. 训练
    train_step = tf.train.GradientDescentOptimizer(LR).minimize(loss, global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TOTAL_TRAIN_STEPS):
            errors = []
            for t in range(ALL_DATA_SIZE):
                input_x_batch = x[t]
                input_y_batch = y[t]

                predict_loss, _, steps = sess.run([loss, train_step, global_step],
                                    feed_dict={input_X:input_x_batch, input_y: input_y_batch})
                
                print("After {step} training step(s),loss on training batch is {predict_loss}"
                      .format(step=steps, predict_loss=predict_loss))
                errors.append(predict_loss)
            print("current epoch {}, mean loss is {}".format(i, np.array(errors).mean()))
