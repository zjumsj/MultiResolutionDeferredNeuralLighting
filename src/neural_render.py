#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("./src")

from neural_texture import create_neural_texture, sample_texture
from neural_texture import sample_texture_lod

import tensorflow as tf
import collections

from ops import resnet_cyclegan
from ops import GDSummaryWriter
from ops import quantize_gamma
from loss import compute_loss
from ops import compute_number_of_parameters
from ops import create_train_op

def select_k_basis_from_5(basis, choices):
    total_number = int (basis.get_shape().as_list()[-1] / 3)
    with tf.name_scope("select_k_basis_from_5"):
        basis_lst = tf.split(basis, total_number, axis = -1) # [B, H, W, 15] => 5 x [B, H, W, 3]
        res_basis_lists = []
        for idx in choices:
            res_basis_lists.append(basis_lst[idx])
        return tf.concat(res_basis_lists, axis = -1)

def sample_texture_lod_bias(neural_texture,uv,lod,args):
    if lod is None:
        return sample_texture(neural_texture, uv, args)

    length = tf.shape(neural_texture[0])[0]  # H
    # as reference texture is 1024x1024
    lod = lod + tf.log(tf.cast(length, tf.float32) / 1024.) / tf.log(2.0)
    return sample_texture_lod(neural_texture,uv,lod,args)

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

        with tf.variable_scope("multipy_basis_and_sampled_texture"):
            assert (args.texture_channels % int(basis_count * 3) == 0)

            n_times = int(args.texture_channels / (basis_count * 3))
            reduced_basis = tf.concat([basis] * n_times, axis=-1, name="concat_more_times")

            mapped_reduced_basis = args.mapper.map_input(reduced_basis)
            mapped_sampled_texture = args.mapper.map_texture(sampled_texture)

            multiplied_texture = mapped_reduced_basis + mapped_sampled_texture
            net_input = multiplied_texture

    # 2. neural renderer
    with tf.variable_scope("unet"):
        output = resnet_cyclegan(net_input, 3, args.activation, 'render', args)
        reconstruct = args.mapper.map_output(output)

    return sampled_texture, reduced_basis, multiplied_texture, output, reconstruct

Model = collections.namedtuple("Model", "train_op, summary_op, loss, vars, output")

def create_model(dataset, args):
    # define network
    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)

        lod = None
        if "lod" in dataset._fields:
            lod = dataset.lod

        sampled_texture, reduced_basis, multiplied_texture, output, reconstruct = neural_render(neural_texture, dataset.uv, lod, dataset.index, dataset.basis, args)

    # loss and train_op
    loss = tf.zeros(shape=(), dtype=tf.float32)

    if "color_gt" in dataset._fields: # for compatibility
        target_image = dataset.color_gt * dataset.mask
    else:
        target_image = dataset.color * dataset.mask

    if args.LDR:
        _gamma = 1.0 if args.linear else 2.2
        target_image = quantize_gamma(target_image,args.keep_max_val,_gamma)
    else:
        target_image = tf.minimum(target_image,args.keep_max_val)

    target = args.mapper.map_input(target_image)
    loss += compute_loss(output, target, args.loss)

    tf_vars = tf.trainable_variables()
    print("[info] Pameters: #%d, Variables: #%d" % (compute_number_of_parameters(tf_vars), len(tf_vars)))

    train_op = create_train_op(args.lr, 0.9, 0.999, loss, tf_vars, "all", args)

    # visualize
    summary_writer = GDSummaryWriter(args.batch_size)
    with tf.name_scope("tensorboard_visualize"):
        # scalar
        summary_writer.add_scalar("loss", loss)

        # image
        summary_writer.add_image("image(GT)", target_image, rescale_factor=args.rescale_output, linear=args.linear)
        summary_writer.add_image("image(recon)", reconstruct, rescale_factor=args.rescale_output,linear=args.linear)

        summary_writer.add_image("sampled_texture", sampled_texture, channels=args.texture_channels)
        summary_writer.add_image("basis", reduced_basis, channels=args.texture_channels)
        summary_writer.add_image("multiplied_texture", multiplied_texture, channels=args.texture_channels)

        for i in range(args.texture_levels):
            summary_writer.add_image('neural_texture_level_%d(0-2)' % i, tf.clip_by_value(neural_texture[i][..., :3][tf.newaxis, ...], 0, 1), channels=3, batch=1)

    summary_op = tf.summary.merge(summary_writer.lists)

    return Model(train_op=train_op,
        summary_op=summary_op,
        loss=loss,
        vars=tf_vars,
        output=reconstruct)

def create_test_model(dataset, args):

    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)

        lod = None
        if "lod" in dataset._fields:
            lod = dataset.lod

        basis = dataset.basis * args.rescale_input

        sampled_texture, reduced_basis, multiplied_texture, output, reconstruct = neural_render(neural_texture, dataset.uv, lod, dataset.index, basis, args)

    return reconstruct

