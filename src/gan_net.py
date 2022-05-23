import numpy as np
import tensorflow as tf

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

# based on progressive_growing_of_gans
# but remove dense which is the same as patchGAN
def PGGAN_D(inputs,name,ngf,max_ngf,levels,
            reuse=False,
            use_wscale = True,
            fused_scale = True
            ):

    x = tf.transpose(inputs,[0,3,1,2]) #->NCHW
    dims = ngf

    with tf.variable_scope(name,reuse=reuse):

        # first layer
        with tf.variable_scope("FirstLayer"):
            x = leaky_relu(apply_bias(conv2d(x,fmaps=dims,kernel=3,use_wscale=use_wscale)))

        for i_level in range(levels):
            with tf.variable_scope("Level%d" % i_level):
                with tf.variable_scope("Conv0"):
                    x = leaky_relu(apply_bias(conv2d(x,fmaps=dims,kernel=3,use_wscale=use_wscale)))
                dims = min(dims * 2, max_ngf)
                if fused_scale:
                    with tf.variable_scope("Conv1_down"):
                        x = leaky_relu(apply_bias(conv2d_downscale2d(x,fmaps=dims,kernel=3,use_wscale=use_wscale)))
                else:
                    with tf.variable_scope("Conv1"):
                        x = leaky_relu(apply_bias(conv2d(x,fmaps=dims,kernel=3,use_wscale=use_wscale)))
                    x = downscale2d(x)

        # last layer
        with tf.variable_scope("LastLayer"):
            x = apply_bias(conv2d(x,fmaps=1,kernel=1,use_wscale=use_wscale))

    return tf.transpose(x,[0,2,3,1]) #->NHWC

#---------------------------------------------------
#
#  SRGAN  https://github.com/tensorlayer/srgan
#
#---------------------------------------------------

def instance_norm(x,act=None):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        if act == 'relu':
            out = tf.nn.relu(out)
        elif act == 'lrelu':
            out = leaky_relu(out)
        return out

def Bias(x):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    return x + b

def Conv2d(inputs,dims,kernel,stride,padding='SAME',act=None,W_init=None,b_init=None):
    in_dims = inputs.shape[-1]
    initializer = tf.random_normal_initializer(stddev=0.02)

    w = tf.get_variable('weight', shape=[kernel[0],kernel[1],in_dims,dims], initializer=initializer)
    x = Bias(tf.nn.conv2d(inputs,w,strides=stride,padding=padding))
    if act == 'relu':
        x = tf.nn.relu(x)
    elif act == 'lrelu':
        x = leaky_relu(x)
    return x

def Conv2d_Interpolate(inputs,dims,kernel,stride,padding='SAME',act=None,W_init=None,b_init=None):
    in_dims = inputs.shape[-1]
    initializer = tf.random_normal_initializer(stddev=0.02)

    new_shape = tf.cast(tf.shape(inputs)[1:3],tf.int32) * 2
    #upscale_x = tf.image.resize(inputs,new_shape,method='bilinear')
    upscale_x = tf.image.resize(inputs, new_shape, method='nearest')
    w = tf.get_variable('weight',shape=[kernel[0], kernel[1], in_dims, dims], initializer=initializer)
    x = Bias(tf.nn.conv2d(upscale_x, w, strides=stride, padding=padding))
    if act == 'relu':
        x = tf.nn.relu(x)
    elif act == 'lrelu':
        x = leaky_relu(x)
    return x

def Conv2d_Transpose(inputs,dims,kernel,stride,padding='SAME',act=None,W_init=None,b_init=None):
    in_dims = inputs.shape[-1]
    initializer = tf.random_normal_initializer(stddev=0.02)

    shapes = tf.cast(tf.shape(inputs),tf.int32) * 2
    shapes = [inputs.shape[0],shapes[1],shapes[2],dims]

    w = tf.get_variable('weight', shape=[kernel[0], kernel[1], dims, in_dims], initializer=initializer)
    x = Bias(tf.nn.conv2d_transpose(inputs, w, output_shape=shapes, strides=stride, padding=padding))
    if act == 'relu':
        x = tf.nn.relu(x)
    elif act == 'lrelu':
        x = leaky_relu(x)
    return x

def Dense(x, fmaps, act=None, W_init=None):

    initializer = tf.random_normal_initializer(stddev=0.02)

    if len(x.shape) > 2:
        #x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        #n = [-1] + [tf.shape(x)[ii] for ii in range(1,len(x.shape))]
        n = [-1, x.shape[-1]]
        x = tf.reshape(x,n)

    w = tf.get_variable('weight',shape=[x.shape[1], fmaps], initializer=initializer)
    w = tf.cast(w, x.dtype)
    x = Bias(tf.matmul(x, w))
    if act == 'relu':
        x = tf.nn.relu(x)
    elif act == 'lrelu':
        x = leaky_relu(x)
    return x

# SRGAN_like structure
def SRGAN_G(inputs,output_channels,activation,prefix,args):

    ngf = args.ngf # original 64
    n_downsampling = args.resnet_conv_count
    n_resblock = args.resnet_res_count

    with tf.variable_scope("SRGAN_G_%s" % prefix):
        with tf.variable_scope("FirstLayer"):
            n = Conv2d(inputs,ngf,[3,3],[1,1],act="relu")

        id = 0
        ngf_ = ngf
        for _ in range(n_downsampling):
            with tf.variable_scope("DownLayer_%d" % id):
                ngf_ = ngf_ * 2
                n = Conv2d(n,ngf_,[4,4],[2,2],act="relu")
                id = id + 1

        # B residual blocks
        for ii in range(n_resblock):
            with tf.variable_scope("Res%d" % ii):
                with tf.variable_scope("Conv0"):
                    nn = Conv2d(n,ngf_,[3,3],[1,1],act=None)
                    nn = instance_norm(nn,act="lrelu")
                    nn = Conv2d(nn,ngf_,[3,3],[1,1],act=None)
                    nn = instance_norm(nn,act=None)
                    n = n + nn

        for _ in range(n_downsampling):
            with tf.variable_scope("UpLayer_%d", id-1):
                ngf_ = ngf_ // 2
                n = Conv2d_Transpose(n,ngf_,[4,4],[2,2],act="relu")
                #n = Conv2d_Interpolate(n,ngf_,[3,3],[1,1],act="relu")
                id = id - 1

        with tf.variable_scope("LastLayer_%d" % id):
            o_c6 = Conv2d(n,output_channels,[1,1],[1,1],act=None)

        if activation == "tanh":
            out_gen = tf.tanh(o_c6, "t1") # [-1,1]
        elif activation == 'sigmoid':
            out_gen = tf.sigmoid(o_c6)
        elif activation == 'none':
            out_gen = o_c6

        return out_gen

def SRGAN_D(inputs,name,ngf,max_ngf,levels,reuse):

    # ngf recommend:64

    with tf.variable_scope(name, reuse=reuse):

        with tf.variable_scope('layer0'): # +1
            n = Conv2d(inputs,ngf,[4,4],[2,2],act="lrelu")

        with tf.variable_scope('layer1'): # +2
            n = Conv2d(n,ngf * 2,[4,4],[2,2],act=None)
            n = instance_norm(n,act="lrelu")
        with tf.variable_scope("layer2"): # +3
            n = Conv2d(n,ngf * 4,[4,4],[2,2],act=None)
            n = instance_norm(n,act="lrelu")
        with tf.variable_scope("layer3"): # +4
            n = Conv2d(n,ngf * 8,[4,4],[2,2],act=None)
            n = instance_norm(n,act="lrelu")
        with tf.variable_scope("layer4"): # +5
            n = Conv2d(n, ngf * 16,[4,4],[2,2],act=None)
            n = instance_norm(n,act="lrelu")
        with tf.variable_scope("layer5"): # +6
            n = Conv2d(n, ngf * 32,[4,4],[2,2],act=None)
            n = instance_norm(n,act="lrelu")

        with tf.variable_scope("layer6"):
            n = Conv2d(n, ngf * 16,[1,1],[1,1],act=None)
            n = instance_norm(n,act="lrelu")
        with tf.variable_scope("layer7"):
            n = Conv2d(n, ngf * 8,[1,1],[1,1],act=None)
            nn = instance_norm(n,act=None)

        with tf.variable_scope("res1"):
            n = Conv2d(nn, ngf * 2,[1,1],[1,1],act=None)
            n = instance_norm(n,act="lrelu")
        with tf.variable_scope("res2"):
            n = Conv2d(n, ngf * 2,[3,3],[1,1],act=None)
            n = instance_norm(n,act="lrelu")
        with tf.variable_scope("res3"):
            n = Conv2d(n, ngf * 8,[3,3],[1,1],act=None)
            n = instance_norm(n,act=None)
        n = n + nn # element-wise add

        with tf.variable_scope("Dense"):
            no = Dense(n,1,act=None)
        return no

#---------------------------------------------------
#
#  simpleNet
#
#---------------------------------------------------

def simpleNet(inputs,output_channels,activation,prefix,args):

    ngf = args.ngf
    n_conv = args.resnet_conv_count

    shape = tf.shape(inputs)
    with tf.variable_scope("simpleNet_%s" % prefix):

        x = inputs
        for ii in range(0,n_conv-1):
            with tf.variable_scope("layer%d" % ii):
                x = Dense(x,ngf,act="relu")
        with tf.variable_scope("last_layer"):
            x = Dense(x,output_channels,act=None)

        o_c6 = tf.reshape(x,[inputs.shape[0],shape[1],shape[2],output_channels])

        if activation == "tanh":
            out_gen = tf.tanh(o_c6, "t1")  # [-1,1]
        elif activation == 'sigmoid':
            out_gen = tf.sigmoid(o_c6)
        elif activation == 'none':
            out_gen = o_c6

    return out_gen

def simpleNetConv(inputs,output_channels,activation,prefix,args):

    ngf = args.ngf
    n_conv = args.resnet_conv_count

    with tf.variable_scope("simpleNetConv_%s" % prefix):
        x = inputs
        for ii in range(0,n_conv-1):
            with tf.variable_scope("layer%d" % ii):
                x = Conv2d(x,ngf,[3,3],[1,1],act="relu")
        with tf.variable_scope("last_layer"):
            o_c6 = Conv2d(x,output_channels,[3,3],[1,1],act=None)

        if activation == "tanh":
            out_gen = tf.tanh(o_c6, "t1")  # [-1,1]
        elif activation == 'sigmoid':
            out_gen = tf.sigmoid(o_c6)
        elif activation == 'none':
            out_gen = o_c6

    return out_gen

def simpleED(inputs,output_channels,activation,prefix,args):

    ngf = args.ngf
    n_conv = 2
    upsample_type = 0

    with tf.variable_scope("simpleED_%s" % prefix):
        x = inputs
        ngf_ = ngf

        idx = 0
        for ii in range(0,n_conv):
            with tf.variable_scope("down_layer%d" % idx):
                x = Conv2d(x,ngf_,[4,4],[2,2],act="relu")
                idx = idx + 1
                ngf_ = ngf_ * 2

        for ii in range(0,n_conv):
            idx = idx - 1
            ngf_ = ngf_ // 2
            with tf.variable_scope("up_layer%d" % idx):
                if upsample_type == 0:
                    x = Conv2d_Transpose(
                        x,
                        output_channels if idx == 0 else ngf_,
                        [4,4],
                        [2,2],
                        act="relu" if idx > 0 else None
                    )
                elif upsample_type == 1:
                    x = Conv2d_Interpolate(
                        x,
                        output_channels if idx == 0 else ngf_,
                        [3,3],
                        [1,1],
                        act="relu" if idx > 0 else None
                    )

        o_c6 = x
        if activation == "tanh":
            out_gen = tf.tanh(o_c6, "t1")  # [-1,1]
        elif activation == 'sigmoid':
            out_gen = tf.sigmoid(o_c6)
        elif activation == 'none':
            out_gen = o_c6

    return out_gen











