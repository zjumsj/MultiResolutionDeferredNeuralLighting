#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

#---------------------------------------------
#
# decoder like architecture
#
#---------------------------------------------

class PGArchitecture:
    def __init__(self,upscale_style=0,do_norm=True,init_flag=-1,gain = 1.0):
        # 0 transpose
        # 1 resize conv
        # 2 pixel shuffle
        self.upscale_style = upscale_style
        self.init_flag = init_flag
        self.gain = gain
        self.gb_do_norm = do_norm

    #--------
    # simple ops
    #--------

    def lrelu(self, x, leak=0.2, name="lrelu", alt_relu_impl=False):
        with tf.variable_scope(name):
            if alt_relu_impl:
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
                return f1 * x + f2 * abs(x)
            else:
                return tf.maximum(x, leak * x)

    # from PGGAN
    # To disallow the scenario where the magnitudes in the generator and discriminator spiral out of control as a result of competition,
    # we normalize the feature vector in each pixel to unit length in the generator after each convolutional layer
    def pixel_norm(self,x,epsilon=1e-8): # also known as LRN
        with tf.variable_scope("pixel_norm"):
            return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)

    def instance_norm(self,x):
        with tf.variable_scope("instance_norm"):
            epsilon = 1e-5
            mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
            scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                    initializer=tf.truncated_normal_initializer(
                                        mean=1.0, stddev=0.02
            ))
            offset = tf.get_variable(
                'offset', [x.get_shape()[-1]],
                initializer=tf.constant_initializer(0.0)
            )
            out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

            return out

    def apply_bias(self,x,initializer=None):
        if initializer is None:
            initializer = tf.initializers.zeros()
        b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=initializer)
        b = tf.cast(b, x.dtype)
        return x + b

    def flatten(self,x):
        if len(x.shape) > 2:
            n = [-1,x.shape[-1]]
            x = tf.reshape(x,n)
        return x

    #--------
    # level 1 (sub layer)
    #-------

    def conv2d_interpolate(self,inputs,dims,kernel,stride,dilations=None,use_bias=True,method='nearest',padding='SAME',initializer1=None,initializer2=None):
        in_dims = inputs.shape[-1]
        new_shape = tf.cast(tf.shape(inputs)[1:3], tf.int32) * 2
        upscale_x = tf.image.resize(inputs, new_shape, method=method)
        assert(initializer1 is not None)
        w = tf.get_variable("weight",shape=[kernel[0],kernel[1],in_dims,dims],initializer=initializer1)
        x = tf.nn.conv2d(upscale_x,w,strides=stride,padding=padding,dilations=dilations)
        if use_bias:
            x = self.apply_bias(x,initializer=initializer2)
        return x

    def conv2d(self,inputs,dims,kernel,stride,dilations=None,use_bias=True,padding='SAME',initializer1=None,initializer2=None):
        in_dims = inputs.shape[-1]
        assert (initializer1 is not None)
        w = tf.get_variable("weight",shape=[kernel[0],kernel[1],in_dims,dims],initializer=initializer1)
        x = tf.nn.conv2d(inputs,w,strides=stride,padding=padding,dilations=dilations)
        if use_bias:
            x = self.apply_bias(x,initializer=initializer2)
        return x

    def conv2d_transpose(self,inputs,dims,kernel,stride,dilations=None,use_bias=True,padding='SAME',initializer1=None,initializer2=None):
        in_dims = inputs.shape[-1]

        shapes = tf.cast(tf.shape(inputs), tf.int32) * 2
        shapes = [inputs.shape[0], shapes[1], shapes[2], dims]

        assert (initializer1 is not None)
        w = tf.get_variable("weight",shape=[kernel[0],kernel[1],dims,in_dims],initializer=initializer1)
        x = tf.nn.conv2d_transpose(inputs,w,output_shape=shapes,strides=stride,padding=padding,dilations=dilations)
        if use_bias:
            x = self.apply_bias(x,initializer=initializer2)
        return x

    def conv2d_pixelshuffle(self,inputs,dims,kernel,stride,dilations=None,use_bias=True,padding='SAME',initializer1=None,initializer2=None):
        in_dims = inputs.shape[-1]
        assert(initializer1 is not None)
        w = tf.get_variable("weight",shape=[kernel[0],kernel[1],in_dims,dims * 4],initializer=initializer1)
        x = tf.nn.conv2d(inputs,w,strides=stride,padding=padding,dilations=dilations)
        if use_bias:
            x = self.apply_bias(x,initializer=initializer2)
        x_shuffle = tf.nn.depth_to_space(x,2)
        return x_shuffle

    def dense(self,inputs,dims,use_bias=True,initializer1=None,initializer2=None):
        x = inputs
        assert(initializer1 is not None)
        w = tf.get_variable('weight', shape=[x.shape[-1], dims], initializer=initializer1)
        w = tf.cast(w, x.dtype)
        x = tf.matmul(x,w)
        if use_bias:
            x = self.apply_bias(x,initializer=initializer2)
        return x

    #--------
    # level 2 (Layer)
    #--------

    def general_conv2d(self,inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):

        with tf.variable_scope(name):
            if self.init_flag == 0:  # torch default
                k1 = np.sqrt(1. / (inputconv.get_shape().as_list()[-1] * f_w * f_w))  # sqrt(k),k=1/(cin*kernelx*kernely)
                initializer1 = tf.random_uniform_initializer(minval=-k1,maxval=k1)
                initializer2 = initializer1
            elif self.init_flag == 1:  # xavier_uniform, also glorot initialization
                k1 = self.gain * np.sqrt(6 / ((inputconv.get_shape().as_list()[-1] + o_d) * f_w * f_w))  # k=gain * sqrt(6/(fan_in+fan_out))
                initializer1 = tf.random_uniform_initializer(minval=-k1, maxval=k1)
                initializer2 = tf.constant_initializer(0.0)
            elif self.init_flag == 2: # # xavier_normal
                k1 = self.gain * np.sqrt(2/((inputconv.get_shape().as_list()[-1] + o_d) * f_w * f_w)) # k = gain *sqrt(2/(fan_in+fan_out))
                initializer1 = tf.random_normal_initializer(mean=0.0,stddev=k1)
                initializer2 = tf.constant_initializer(0.0)
            else: # original
                initializer1 = tf.truncated_normal_initializer(stddev=stddev)
                initializer2 = tf.constant_initializer(0.0)

            conv = self.conv2d(inputconv,o_d,[f_h,f_w],[s_h,s_w],use_bias=True,padding=padding,initializer1=initializer1,initializer2=initializer2)

            if do_norm:
                conv = self.instance_norm(conv)

            if do_relu:
                if (relufactor == 0):
                    conv = tf.nn.relu(conv, "relu")
                else:
                    conv = self.lrelu(conv, relufactor, "lrelu")
            return conv

    def general_deconv2d(self,inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
        with tf.variable_scope(name):
            if self.init_flag == 0: # torch default
                k1 = np.sqrt(1./(inputconv.get_shape().as_list()[-1] * f_w * f_w)) # sqrt(k),k=1/(cin*kernelx*kernely)
                initializer1 = tf.random_uniform_initializer(minval=-k1,maxval=k1)
                initializer2 = initializer1
            elif self.init_flag == 1: # xavier_uniform, also glorot initialization
                k1 = self.gain *np.sqrt(6/((inputconv.get_shape().as_list()[-1] + o_d) * f_w * f_w)) # k=gain * sqrt(6/(fan_in+fan_out))
                initializer1 = tf.random_uniform_initializer(minval=-k1,maxval=k1)
                initializer2 = tf.constant_initializer(0.0)
            elif self.init_flag == 2: # xavier_normal
                k1 = self.gain * np.sqrt(2/((inputconv.get_shape().as_list()[-1] + o_d) * f_w * f_w)) # k = gain *sqrt(2/(fan_in+fan_out))
                initializer1 = tf.random_normal_initializer(mean=0.0,stddev=k1)
                initializer2 = tf.constant_initializer(0.0)
            else: # original
                initializer1 = tf.truncated_normal_initializer(stddev=stddev)
                initializer2 = tf.constant_initializer(0.0)

            if self.upscale_style == 0:
                conv = self.conv2d_transpose(inputconv,o_d,[f_h,f_w],[s_h,s_w],use_bias=True,padding=padding,initializer1=initializer1,initializer2=initializer2)
            elif self.upscale_style == 1:
                conv = self.conv2d_interpolate(inputconv,o_d,[f_h,f_w],[s_h,s_w],use_bias=True,padding=padding,initializer1=initializer1,initializer2=initializer2)
            elif self.upscale_style == 2:
                conv = self.conv2d_pixelshuffle(inputconv,o_d,[f_h,f_w],[s_h,s_w],use_bias=True,padding=padding,initializer1=initializer1,initializer2=initializer2)

            if do_norm:
                conv = self.instance_norm(conv)

            if do_relu:
                if (relufactor == 0):
                    conv = tf.nn.relu(conv, "relu")
                else:
                    conv = self.lrelu(conv, relufactor, "lrelu")
            return conv

    def general_dense(self,input,o_d=64,stddev=0.02,name="dense",do_norm=False,do_relu=True,relufactor=0):
        with tf.variable_scope(name):
            if self.init_flag == 0: # torch default
                k1 = np.sqrt(1./input.get_shape().as_list()[-1])
                initializer1 = tf.random_uniform_initializer(minval=-k1,maxval=k1)
                initializer2 = initializer1
            elif self.init_flag == 1: # xavier_uniform, also glorot initialization
                k1 = self.gain * np.sqrt(6. / (input.get_shape().as_list()[-1] + o_d) ) # k=gain * sqrt(6/(fan_in+fan_out))
                initializer1 = tf.random_uniform_initializer(minval=-k1,maxval=k1)
                initializer2 = tf.constant_initializer(0.0)
            elif self.init_flag == 2: # xavier_normal
                k1 = self.gain * np.sqrt( 2./(input.get_shape().as_list()[-1] + o_d)) # k = gain *sqrt(2/(fan_in+fan_out))
                initializer1 = tf.random_normal_initializer(mean=0.0, stddev=k1)
                initializer2 = tf.constant_initializer(0.0)
            else: # original
                initializer1 = tf.truncated_normal_initializer(stddev=stddev)
                initializer2 = tf.constant_initializer(0.0)

            net = self.dense(input,o_d,use_bias=True,initializer1=initializer1,initializer2=initializer2)

            if do_norm:
                net = self.instance_norm(net)

            if do_relu:
                if (relufactor == 0):
                    net = tf.nn.relu(net, "relu")
                else:
                    net = self.lrelu(net, relufactor, "lrelu")
            return net

    #--------
    # level 3 (Block)
    #--------

    def build_resnet_block(self,inputres,dim,do_final_relu=True,name='resnet',padding="REFLECT",do_norm=True):
        with tf.variable_scope(name):
            out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
            out_res = self.general_conv2d(out_res,dim,3,3,1,1,0.02,"VALID","c1",do_norm)
            out_res = tf.pad(out_res,[[0,0],[1,1],[1,1],[0,0]],padding)
            out_res = self.general_conv2d(out_res,dim,3,3,1,1,0.02,"VALID","c2",do_norm,do_relu = False)
            if do_final_relu:
                return tf.nn.relu(out_res + inputres)
            return out_res + inputres

    def gated_conv(self,inputres,dim,f_h=3, f_w=3, s_h=1, s_w=1, name='gated_conv',padding="REFLECT",do_norm = True,do_relu=True,relufactor=0):
        with tf.variable_scope(name):
            k_pad = f_h // 2
            x_pad = tf.pad(inputres,[[0,0],[k_pad,k_pad],[k_pad,k_pad],[0,0]], padding)
            gate = self.general_conv2d(x_pad,dim,f_h,f_w,s_h,s_w,0.02,"VALID","gate",do_norm=False,do_relu=False)
            content = self.general_conv2d(x_pad,dim,f_h,f_w,s_h,s_w,0.02,"VALID","content",do_norm=False,do_relu=False)
            out = tf.sigmoid(gate) * content
            if do_norm:
                out = self.instance_norm(out)
            if do_relu:
                if (relufactor == 0):
                    out = tf.nn.relu(out,"relu")
                else:
                    out = self.lrelu(out,relufactor,"lrelu")
            return out

    #--------
    # build net
    #--------

    # the proposed architecture in our paper
    def texture_decoder(self,inputs,output_channels,activation,prefix,args):

        ngf = args.ngf
        n_upsampling = args.module_count
        n_conv = args.conv_count

        use_gated_conv = False
        up_stride = 2 if self.upscale_style == 0 else 1

        with tf.variable_scope("texture_decoder_%s" % prefix):
            x = None
            for idx in range(n_upsampling):
                if idx == 0:
                    x = inputs[idx]
                else:
                    x = tf.concat([x,inputs[idx]],axis=-1)

                for jj in range(n_conv-1):
                    if use_gated_conv:
                        x = self.gated_conv(x,ngf,3,3,1,1,"gated_conv%d_%d" % (idx,jj),do_norm=self.gb_do_norm)
                    else:
                        x_pad = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')
                        x = self.general_conv2d(x_pad,ngf,3,3,1,1,0.02,"VALID","conv%d_%d" % (idx,jj),do_norm=self.gb_do_norm)

                if idx < n_upsampling - 1:
                    x = self.general_deconv2d(x,ngf,3,3,up_stride,up_stride,0.02,"SAME",'conv%d_up' % idx,do_norm=self.gb_do_norm)
                else:
                    # final layer
                    o_c6 = self.general_conv2d(x,output_channels,3,3,1,1,0.02,"SAME",'final',do_norm=False,do_relu=False)

            if activation == "tanh":
                out_gen = tf.tanh(o_c6, "t1")  # [-1,1]
            elif activation == 'sigmoid':
                out_gen = tf.sigmoid(o_c6)
            elif activation == 'none':
                out_gen = o_c6

            return out_gen

    # the proposed architecture with progressive training & multi-level loss support
    def texture_decoder_progressive(self, inputs, output_channels, activation, prefix, args):

        ngf = args.ngf
        n_upsampling = args.module_count
        n_conv = args.conv_count

        use_gated_conv = False
        up_stride = 2 if self.upscale_style == 0 else 1
        output_list = []

        with tf.variable_scope("texture_decoder_%s" % prefix):
            x = None
            for idx in range(n_upsampling):
                if idx == 0:
                    x = inputs[idx]
                else:
                    x = tf.concat([x, inputs[idx]], axis=-1)

                for jj in range(n_conv - 1):
                    if use_gated_conv:
                        x = self.gated_conv(x, ngf,3,3,1,1, "gated_conv%d_%d" % (idx, jj), do_norm=self.gb_do_norm)
                    else:
                        x_pad = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')
                        x = self.general_conv2d(x_pad, ngf, 3, 3, 1, 1, 0.02, "VALID", "conv%d_%d" % (idx, jj), do_norm=self.gb_do_norm)

                o_c6 = self.general_conv2d(x, output_channels,3,3,1,1,0.02,"SAME",'output%d' % idx,do_norm=False,do_relu=False)
                if activation == "tanh":
                    out_gen = tf.tanh(o_c6, "t1")  # [-1,1]
                elif activation == 'sigmoid':
                    out_gen = tf.sigmoid(o_c6)
                elif activation == 'none':
                    out_gen = o_c6
                output_list.append(out_gen)

                if idx < n_upsampling - 1:
                    x = self.general_deconv2d(x, ngf, 3, 3, up_stride, up_stride, 0.02, "SAME", 'conv%d_up' % idx, do_norm=self.gb_do_norm)

            return output_list

    # other design of feature transform module shown in evaluation
    def texture_decoder_v2(self,inputs,output_channels,activation,prefix,args):

        setting = 0 # {0,1,2,3,4}

        # params definition
        # m downsampling number
        # m+1 upsampling number
        # n normal convolution number
        # q resblock number

        m = 0 ## ours default setting
        n = 1
        q = 0

        if setting == 1: # no downsample, resblock=1
            m = 0
            n = 0
            q = 1
        elif setting == 2: # no downsample, resblock=2
            m = 0
            n = 0
            q = 2
        elif setting == 3: # one downsample, resblock=1
            m = 1
            n = 0
            q = 1
        elif setting == 4: # one downsample, resblock=2
            m = 1
            n = 0
            q = 2

        ngf = args.ngf
        n_upsampling = args.module_count
        padding = args.resnet_padding

        do_final_relu = True
        use_gated_conv = False
        up_stride = 2 if self.upscale_style == 0 else 1

        with tf.variable_scope("texture_decoder_v2_%s" % prefix):
            x = None
            for idx in range(n_upsampling): # LEVEL OUTER LOOP
                if idx == 0:
                    x = inputs[idx]
                else:
                    x = tf.concat([x,inputs[idx]],axis=-1)

                for jj in range(m): # downsample
                    x = self.general_conv2d(x,ngf,3,3,2,2,0.02,"SAME","down%d_%d" % (idx,jj),do_norm=self.gb_do_norm)

                for jj in range(n): # normal conv
                    if use_gated_conv:
                        x = self.gated_conv(x,ngf,3,3,1,1,"gated_conv%d_%d" % (idx,jj), padding=padding,do_norm=self.gb_do_norm)
                    else:
                        x_pad = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
                        x = self.general_conv2d(x_pad,ngf,3,3,1,1,0.02,"VALID","conv%d_%d" % (idx,jj),do_norm=self.gb_do_norm)

                if x.get_shape().as_list()[-1] != ngf: # transfer channel number to ngf
                    print('extra layer added!')
                    x = self.general_conv2d(x,ngf,1,1,1,1,0.02,"VALID","conv%d_0" % idx,do_norm=False,do_relu=False)

                for jj in range(q): # resblock
                    x = self.build_resnet_block(x, ngf, do_final_relu, "resblock%d_%d" % (idx,jj), padding, do_norm=self.gb_do_norm)

                for jj in range(m + 1): # upsampling
                    if jj == m and idx == n_upsampling - 1:
                        # final layer
                        o_c6 = self.general_conv2d(x,output_channels,3,3,1,1,0.02,"SAME",'final',do_norm=False,do_relu=False)
                    else:
                        x = self.general_deconv2d(x,ngf,3,3,up_stride,up_stride,0.02,"SAME",'up%d_%d' % (idx,jj),do_norm=self.gb_do_norm)

            if activation == "tanh":
                out_gen = tf.tanh(o_c6, "t1")  # [-1,1]
            elif activation == 'sigmoid':
                out_gen = tf.sigmoid(o_c6)
            elif activation == 'none':
                out_gen = o_c6

            return out_gen

    # Unet of Deferred Neural Rendering, Image Synthesis using Neural Textures
    def NT_unet(self,inputs,output_channels,activation,prefix,args):

        up_stride = 2 if self.upscale_style == 0 else 1

        with tf.variable_scope("NTUnet_%s" % prefix):
            x1 = self.general_conv2d(inputs,64,4,4,2,2,0.02,"SAME",name="down1",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            x2 = self.general_conv2d(x1,128,4,4,2,2,0.02,"SAME",name="down2",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            x3 = self.general_conv2d(x2,256,4,4,2,2,0.02,"SAME",name="down3",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            x4 = self.general_conv2d(x3,512,4,4,2,2,0.02,"SAME",name="down4",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            x5 = self.general_conv2d(x4,512,4,4,2,2,0.02,"SAME",name="down5",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)

            x = self.general_deconv2d(x5,512,4,4,up_stride,up_stride,0.02,"SAME",name="up1",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            x = self.general_deconv2d(tf.concat([x,x4],axis=-1),512,4,4,up_stride,up_stride,0.02,"SAME",name="up2",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            x = self.general_deconv2d(tf.concat([x,x3],axis=-1),256,4,4,up_stride,up_stride,0.02,"SAME",name="up3",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            x = self.general_deconv2d(tf.concat([x,x2],axis=-1),128,4,4,up_stride,up_stride,0.02,"SAME",name="up4",do_norm=self.gb_do_norm,do_relu=True,relufactor=0.2)
            o_c6 = self.general_deconv2d(tf.concat([x,x1],axis=-1),output_channels,4,4,up_stride,up_stride,0.02,"SAME",name="up5",do_norm=False,do_relu=False)

            if activation == "tanh":
                out_gen = tf.tanh(o_c6, "t1")  # [-1,1]
            elif activation == 'sigmoid':
                out_gen = tf.sigmoid(o_c6)
            elif activation == 'none':
                out_gen = o_c6

            return out_gen

