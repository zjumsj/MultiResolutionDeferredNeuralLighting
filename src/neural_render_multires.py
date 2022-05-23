#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("./src")

from neural_texture import create_neural_texture, sample_texture

import tensorflow as tf
import collections

# TODO, remove xxx
from src.ops_experimental import PGArchitecture
from src.ops import avg_downsample
from src.ops import GDSummaryWriter
from src.ops import quantize_gamma
from src.loss import compute_loss
from src.ops import compute_number_of_parameters
from src.ops import create_train_op
from src.vgg.perceptual_loss import perceptual_loss_vgg16
#from vgg.vgg19 import perceptual_loss_vgg19
from functools import partial

from src.neural_render import select_k_basis_from_5, sample_texture_lod_bias

# progressive_flag
# enable to use progressive training
# 0 disable
# 1 classical progressive training
# 2 gather loss in all levels

#progressive_flag = 2 # 0
#progressive_flag = 0

def neural_render(neural_texture, uv, lod, index, _basis, args):

    # flip uv, y axis
    x_ = uv[:,:,:,0]
    y_ = 1 - uv[:,:,:,1]
    uv = tf.stack([x_,y_],axis=-1)

    # 0. select k basis from 5
    basis_count = len(args.basis_lists)
    basis = select_k_basis_from_5(_basis, args.basis_lists)

    # 1. neural render part
    with tf.variable_scope("neural_renderer"):
        with tf.variable_scope("sample_texture"):
            sampled_texture = sample_texture_lod_bias(neural_texture, uv, lod, args)

        with tf.variable_scope("multiply_basis_and_sample_texture"):

            #n_upsampling = args.resnet_res_count
            n_upsampling = args.module_count

            input_list = []
            #sampled_texture_list = [] # visualize sampled texture
            mapped_basis = args.mapper.map_input(basis)

            merged_feature = tf.concat([mapped_basis, sampled_texture],axis=-1)
            input_list.append(merged_feature)
            for ii in range(1, n_upsampling):
                merged_feature = avg_downsample(merged_feature)
                input_list.append(merged_feature)

            input_list.reverse() # coarse -> fine

    # 2. neural renderer
    with tf.variable_scope("unet"):
        c_ = PGArchitecture(
            upscale_style=1,
            do_norm=True
        )
        if args.progressive_flag == 1:
            # classical progressive training
            output = c_.texture_decoder_progressive(input_list,3,args.activation,'render',args)

            if args.stage > 2 * args.module_count - 2 or args.stage < 0:  # default all layers in
                output = output[-1]
            elif args.stage % 2 == 0:  # single
                output = output[args.stage // 2]
            else:
                output_low = output[(args.stage - 1) // 2]
                output_high = output[(args.stage + 1) // 2]
                output_low = tf.image.resize(output_low, tf.shape(output_high)[1:3], method='bilinear')
                output = output_low * (1 - args.lod) + output_high * args.lod
        elif args.progressive_flag == 2:
            # count all loss
            output = c_.texture_decoder_progressive(input_list,3,args.activation,'render',args)
        else:
            ## texture_decoder is our default network architecture
            ## TODO, write something here

            #output = c_.texture_decoder_v2(input_list,3,args.activation,'render',args)
            #output = c_.NT_unet(input_list,3,args.activation,'render',args)
            output = c_.texture_decoder(input_list,3,args.activation,'render',args)

        if isinstance(output,list):
            reconstruct = [args.mapper.map_output(o) for o in output]
        else:
            reconstruct = args.mapper.map_output(output)

    return sampled_texture,  basis, output, reconstruct

Model = collections.namedtuple("Model", "train_op, vgg_op, summary_op, loss, vars, output, real, fake")

def create_model(dataset, args, first_call=True):

    #n_upsampling = args.resnet_res_count
    n_upsampling = args.module_count

    #define network
    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)

        lod = None
        if "lod" in dataset._fields:
            lod = dataset.lod

        sampled_texture, basis, output, reconstruct = neural_render(neural_texture, dataset.uv, lod, dataset.index, dataset.basis, args)

    # loss and train_op
    loss = tf.zeros(shape=(), dtype=tf.float32)
    # loss_aug = tf.zeros(shape=(), dtype=tf.float32)

    if "color_gt" in dataset._fields: # for compatibility with downgraded edition
        target_image = dataset.color_gt * dataset.mask
    else:
        target_image = dataset.color * dataset.mask

    if args.progressive_flag == 1:
        lod = args.stage
        if lod < 0:
            lod = 10000
        lod = min((lod + 1) // 2, args.module_count-1)

        for _ in range(args.module_count-1-lod):
            target_image = avg_downsample(target_image)

    if args.LDR:
        _gamma = 1.0 if args.linear else 2.2
        target_image = quantize_gamma(target_image, args.keep_max_val, _gamma)
    else:
        target_image = tf.minimum(target_image, args.keep_max_val)

    target = args.mapper.map_input(target_image)

    if args.progressive_flag == 2: # create pyramid

        output.reverse() # fine -> coarse
        reconstruct.reverse() # fine -> coarse

        target = [target]
        target_image = [target_image]
        for ii in range(1,n_upsampling):
            target.append(avg_downsample(target[-1]))
            target_image.append(avg_downsample(target_image[-1]))

    # original loss
    common_loss = None
    if args.loss_scale != 0:
        if args.progressive_flag == 2:
            # TODO ...
            #W = [1.] * n_upsampling # each layer is equally evaluated
            W = [1./float(ii * ii) for ii in range(1,n_upsampling + 1)] # each pixel is equally evaluated
            W[0] = W[0] * 100. # put more weight on the finest level

            common_loss = compute_loss(output,target, args.loss, weights= W )
        else:
            common_loss = compute_loss(output, target, args.loss)  # !
        loss += common_loss * args.loss_scale

    # VGG loss
    vgg_model = None
    vgg_loss = None
    if args.vgg_loss_scale != 0:
        target_ = (target + 1.) * 0.5 * 255.  # mapping to [0,255]
        output_ = (output + 1.) * 0.5 * 255.

        vgg_loss, vgg_model = perceptual_loss_vgg16(partial(compute_loss, method=args.vgg_loss), output_, target_,
                                                    args.vgg_layer_list,
                                                    args.vgg_blend_weights,
                                                    first_call=first_call)

        '''
        vgg_loss,vgg_model = perceptual_loss_vgg19(partial(compute_loss,method=args.vgg_loss),output_,target_,
                                      args.vgg_layer_list,
                                      args.vgg_blend_weights,
                                      first_call= first_call )
        '''

        loss += vgg_loss * args.vgg_loss_scale

    if args.adv_loss_scale != 0:
        tf_vars = None
        train_op = None
    else:
        tf_vars = tf.trainable_variables(scope="OffsetNetwork")
        print("[info] Pameters: #%d, Variables: #%d" % (compute_number_of_parameters(tf_vars), len(tf_vars)))

        train_op = create_train_op(args.lr, 0.9, 0.999, loss, tf_vars, "G", args)

    # visualize
    summary_writer = GDSummaryWriter(args.batch_size)
    with tf.name_scope("tensorboard_visualize"):
        # scalar
        summary_writer.add_scalar("loss", loss)
        if common_loss is not None:
            summary_writer.add_scalar("common_loss", common_loss)
        if vgg_loss is not None:
            summary_writer.add_scalar("vgg_loss", vgg_loss)

        # image
        if isinstance(target_image,list):
            for i in range(len(target_image)):
                merged_image = tf.concat([target_image[i],reconstruct[i]],axis=2) # on W ?
                summary_writer.add_image("image_level_%d" % i, merged_image, rescale_factor=args.rescale_output,linear=args.linear)
        else:
            summary_writer.add_image("image(GT)", target_image, rescale_factor=args.rescale_output, linear=args.linear)
            summary_writer.add_image("image(recon)", reconstruct, rescale_factor=args.rescale_output, linear=args.linear)

        summary_writer.add_image("basis", basis)
        summary_writer.add_image("sampled_texture", sampled_texture, channels=args.texture_channels)

        for i in range(args.texture_levels):  # one image per level
            summary_writer.add_image('neural_texture_level_%d' % i, tf.clip_by_value(neural_texture[i][..., :3][tf.newaxis, ...], 0, 1), channels=3, batch=1)

    summary_op = tf.summary.merge(summary_writer.lists)

    return Model(train_op=train_op,
                 vgg_op=vgg_model,
                 summary_op=summary_op,
                 loss=loss,
                 vars=tf_vars,
                 output=reconstruct,
                 real=target,
                 fake=output)

def create_test_model(dataset,args):

    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)

        lod = None
        if "lod" in dataset._fields:
            lod = dataset.lod

        basis = dataset.basis * args.rescale_input

        sampled_texture, basis, output, reconstruct = neural_render(neural_texture, dataset.uv, lod, dataset.index, basis, args)

    return reconstruct
