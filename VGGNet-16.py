from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel =tf.get_variable(scope+"w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val,trainable=True,name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p+=[kernel,biases]
        return activation

def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name = scope)
        p+=[kernel,biases]
        return activation

def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize =[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name =name)


def inference_op(input_op,keep_prob):
    p = []
    conv1_1 = conv_op(input_op,name = "conv1_1",kh = 3,kw = 3, n_out = 64 ,dh = 1 ,dw = 1 ,p = p)
    conv1_2 = conv_op(conv1_1,name = "conv1_2",kh = 3,kw = 3,n_out = 64,dh =1,dw =1,p = p)
    pool1 = mpool_op(conv1_2,name = "pool1",kh=2,kw,2,dw =2,dh=2)

    conv2_1 = conv_op(pool1,name = "conv2_1",kh = 3,kw = 3,n_out = 128,dh=1,dw =1,p = p)
    conv2_2 = conv_op(conv2_1,name = "conv2_2",kh = 3,kw =3 ,n_out = 128,dh =1,dw =1,p = p)
    pool2 = mpool_op(conv2_2,name = "pool2",kh = 2,kw = 2,dh = 2,dw = 2 )

    conv3_1 = conv_op(pool2,name = "conv3_1",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p = p )
    conv3_2 = conv_op(conv3_1,name = "conv3_2",kh =3, kw = 3,n_out = 256,dh = 1,dw = 1,p = p )
    conv3_3 = conv_op(conv3_2,name = "conv3_3",kh = 3,kw = 3,n_out = 256,dh = 1,dw = 1,p= p )
    pool3 = mpool_op(conv3_3,name ="pool3",kh = 2,kw = 2 dh = 2,dw = 2)
