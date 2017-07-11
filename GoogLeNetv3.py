import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncatemod_normal_initializer(0.0,stddev)

 def inception_v3_arg_scope(weight_decay = 0.00004,stddev = 0.1,batch_norm_var_collection = 'moving_vars'):
     batch_norm_params = {
         'decay' : 0.9997,
         'epsilon' : 0.001,
         'updates_collections' : tf.GraphKeys.UPDATE_OPS,
         'variables_collections' : {
             'beta' : None,
             'gamma' : None,
             'moving_mean' : [batch_norm_var_collection],
             'moving_variance' : [batch_norm_var_collection]
         }
     }
    with slim.arg_scope([slim.conv2d,slim.fully_connected],weight_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params) as sc:
        return sc

def inception_v3_base(inputs,scope = None):
    end_points = {}
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.arg_pool2d],stride =1,padding = 'VALID'):
            net = slim.conv2d(inputs,32,[3,3],stride = 2,scope = 'Conv2d_1a_3x3')
            net = slim.conv2d(net,32,[3,3],scope = 'Conv2d_2a_3x3')
            net = slim.conv2d(net,64,[3,3],padding = 'SAME')
            net = slim.max_pool2d(net,[3,3],stride = 2,scope = 'MaxPool_3a_3x3')
            net = slim.conv2d(net,80,[1,1],scope = 'Conv2d_3b_1x1')
            net = slim.conv2d(net,192,[3,3],scope = 'Conv2d_4a_3x3')
            net = slim.max_pool2d(net,[3,3],stride = 2,scope = 'MaxPool_5a_3x3')

        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride = 1,padding = 'SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,64,[1,1],scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,48,[1,1],scope = 'Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1,64,[5,5],scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,64,[1,1],scope = 'Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2,96,[3,3],scope = 'Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2,96,[3,3],scope = 'Conv2d_2c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3,32,[1,1],scope = 'Conv2d_3b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,64,[1,1],scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,48,[1,1],scope = 'Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1,64,[5,5],scope = 'Conv2d_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,64,[1,1],scope = 'Conv2d_1a_1x1')
                    branch_2 = slim.conv2d(branch_2,96,[3,3],scope = 'Conv2d_1b_3x3')
                    branch_2 = slim.conv2d(branch_2,96,[3,3],scope = 'Conv2d_1c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,64,[1,1],scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,64,[1,1],scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(branch_0,48,[1,1],scope = 'Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1,64,[5,5],scope = 'Conv2d_1b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,64,[1,1],scope = 'Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2,96,[3,3],scope = 'Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2,96,[3,3],scope = 'Conv2d_2c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3,64,[1,1],scope = 'Conv2d_3b_1x1')
                net = tf.concat([branh_0,branch_1,branch_2,branch_3],3)

            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,384,[3,3],stride = 2,padding = 'VALID',scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,64,[1,1],scope = 'Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1,96,[3,3],scope = 'Conv2d_1b_3x3')
                    branch_1 = slim.conv2d(branch_1,96,[3,3],stride = 2,padding = 'VALID',scope = 'Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net,[3,3],stride = 2,padding = 'VALID',scope = 'MaxPool_1a_3x3')
                net = tf.concat([branch_0,branch_1,branch_2],3)

            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,192,[1,1],scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,128,[1,1],scope = 'Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1,128,[1,7],scope = 'Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1,192,[7,1],scope = 'Conv2d_1c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,128,[1,1],scope = 'Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2,128,[7,1],scope = 'Conv2d_2b_7x1')
                    branch_2 = slim.conv2d(branch_2,128,[1,7],scope = 'Conv2d_2c_1x7')
                    branch_2 = slim.conv2d(branch_2,128,[7,1],scope = 'Conv2d_2d_7x1')
                    branch_2 = slim.conv2d(branch_2,192,[1,7],scope = 'Conv2d_2e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3,192,[1,1],scope = 'Conv2d_3b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,192,[1,1],scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net,160,[1,1],scope = 'Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1,160,[1,7],scope = 'Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1,192,[7,1],scope = 'Conv2d_1c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,160,[1,1],scope = 'Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2,160,[7,1],scope = 'Conv2d_2b_7x1')
                    branch_2 = slim.conv2d(branch_2,160,[1,7],scope = 'Conv2d_2c_1x7')
                    branch_2 = slim.conv2d(branch_2,160,[7,1],scope = 'Conv2d_2d_7x1')
                    branch_2 = slim.conv2d(branch_2,192,[1,7],scope = 'Conv2d_2d_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3,192,[1,1],scope = 'Conv2d_3b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2],3)

            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net,192,[1,1],scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(branch_0,160,[1,1],scope = 'Con2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1,160,[1,7],scope = 'Con2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1,192,[7,1],scope = 'Con2d_1c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net,160,[1,1],scope = 'Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2,160,[7,1],scoep = 'Conv2d_2b_7x1')
                    branch_2 = slim.conv2d(branch_2,160,[1,7],scope = 'Conv2d_2c_1x7')
                    branch_2 = slim.conv2d(branch_2,160,[7,1],scope = 'Conv2d_2d_7x1')
                    branch_2 = slim.conv2d(branch_2,192,[1,7],scope = 'Conv2d_2e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3,192,[1,1],scope = 'Conv2d_3b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

                with tf.variable_scope('Mixed_6e'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope = 'Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(branch_0,160,[1,1],scope = 'Con2d_1a_1x1')
                        branch_1 = slim.conv2d(branch_1,160,[1,7],scope = 'Con2d_1b_1x7')
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope = 'Con2d_1c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,160,[1,1],scope = 'Conv2d_2a_1x1')
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scoep = 'Conv2d_2b_7x1')
                        branch_2 = slim.conv2d(branch_2,160,[1,7],scope = 'Conv2d_2c_1x7')
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope = 'Conv2d_2d_7x1')
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope = 'Conv2d_2e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_3a_3x3')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope = 'Conv2d_3b_1x1')
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
                end_points['Mixed_6e'] = net

                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope = 'Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0,320,[3,3],stride = 2,padding = 'VALID',scope = 'Conv2d_0a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,192,[1,1],scope = 'Conv2d_1a_1x1')
                        branch_1 = slim.conv2d(branch_1,192,[1,7],scope = 'Conv2d_1b_1x7')
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope = 'Conv2d_1c_7x1')
                        branch_1 = slim.conv2d(branch_1,192,[3,3],stride = 2,padding = 'VALID',scope = 'Conv2d_1d_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net,[3,3],stride = 2,padding = 'VALID',scope = 'MaxPool_2a_3x3')
                    net = tf.concat([branch_0,branch_1,branch_2],3)

                with tf.variable_scope('Mixed_7b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,320,[1,1],scope = 'Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(branch_0,384,[1,1],scope = 'Con2d_1a_1x1')
                        branch_1 = tf.concat([slim.conv2d(branch_1,384,[1,3],scope  = 'Conv2d_1b_1x3'),slim.conv2d(branch_1,384,[3,1],scope ='Conv2d_1b_3x1')],3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,448,[1,1],scope = 'Conv2d_2a_1x1')
                        branch_2 = slim.conv2d(branch_2,384,[3,3],scoep = 'Conv2d_2b_3x3')
                        branch_2 = tf.concat([slim.conv2d(branch_2,384,[1,3],scope  = 'Conv2d_2c_1x3'),slim.conv2d(branch_2,384,[3,1],scope ='Conv2d_2d_3x1')],3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_3a_3x3')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope = 'Conv2d_3b_1x1')
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

                with tf.variable_scope('Mixed_7c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,320,[1,1],scope = 'Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(branch_0,384,[1,1],scope = 'Con2d_1a_1x1')
                        branch_1 = tf.concat([slim.conv2d(branch_1,384,[1,3],scope  = 'Conv2d_1b_1x3'),slim.conv2d(branch_1,384,[3,1],scope ='Conv2d_1b_3x1')],3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,448,[1,1],scope = 'Conv2d_2a_1x1')
                        branch_2 = slim.conv2d(branch_2,384,[3,3],scoep = 'Conv2d_2b_3x3')
                        branch_2 = tf.concat([slim.conv2d(branch_2,384,[1,3],scope  = 'Conv2d_2c_1x3'),slim.conv2d(branch_2,384,[3,1],scope ='Conv2d_2d_3x1')],3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_3a_3x3')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope = 'Conv2d_3b_1x1')
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
                return net,end_points

def inception_v3(inputs,num_classes = 1000,is_training =True,dropout_keep_prob = 0.8,prediction_fn=slim.softmax,spatial_squeeze = True,reuse = None,scope = 'InceptionV3'):
    with tf.variable_scope(scope,'InceptionV3',[inputs,num_classes],reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training = is_training):
            net,end_points = inception_v3_base(inputs,scope = scope)

        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride = 1,padding = 'SAME'):
            aux_logits =end_points['Mixed_6e']
            with tf.variable_scope('AuxLogits'):
                aux_logits = slim.avg_pool2d(aux_logits,[5,5],stride = 3,padding = 'VALID',scope = 'AvgPool_3a_5x5')
                aux_logits = slim.conv2d(aux_logits,128,[1,1],scope = 'Conv2d_1b_1x1')

                aux_logits = slim.conv2d(aux_logits,768,[5,5],weights_initializer = trunc_normal(0.01),padding = 'VALID',scope ='Conv2d_2a_5x5')
                aux_logits = slim.conv2d(aux_logits,num_classes,[1,1],activation_fn =None,normalizer_fn=None,weights_initializer=trunc_normal(0.001),scope='Conv2d2b_1x1')
            if spatial_squeeze:
                aux_logits = tf.squeeze(aux_logits,[1,2],name='SpatialSqueeze')
            end_points['AuxLogits'] = aux_logits

    with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net,[8,8],padding ='VALID',scope ='AvgPool_1a_8x8')
        net = slim.dropout(net,keep_prob =dropout_keep_prob,scope ='Dropout_1b')
        end_points['prediction_fn'] = net
        Logits = slim.conv2d(net,num_classes,[1,1],activation_fn = None,normalizer_fn=None,scope ='Conv2d_1c_1x1')

        if spatial_squeeze:
            Logits = tf.squeeze:
            Logits = tf.squeeze(Logits,[1,2],name = 'SpatialSqueeze')
        end_points['Logits']= Logits
        end_points['prediction_fn'] = prediction_fn(Logits,scope ='Predictions')
    return logits,end_points

batch_size = 32
height,width =299,299
inputs = tf.random_uniform((batch_size,height,width,3))
with slim.arg_scope(inception_v3_arg_scope()):
    logits,end_points = inception_v3(inputs,is_training=False)

init = tf.global_variables_initializer()
sess = tf.Session
sess.run(init)
num_batches = 100
time_tensorflow_run(sess,logits,"Forward")
