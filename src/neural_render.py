#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("./src")

# TODO, remove xxx
from src.neural_texture import create_neural_texture, sample_texture
from src.neural_texture import sample_texture_lod
#from src.neural_texture2 import create_neural_texture_from_value, sample_texture_lod


import tensorflow as tf
import collections

from src.ops import resnet_cyclegan
from src.ops import GDSummaryWriter
#from ops import quantize
from src.ops import quantize_gamma
from src.loss import compute_loss
from src.ops import compute_number_of_parameters
from src.ops import create_train_op

#import os
#import numpy as np

#import exr_loader


def select_k_basis_from_5(basis, choices):
    total_number = int (basis.get_shape().as_list()[-1] / 3)
    with tf.name_scope("select_k_basis_from_5"):
        basis_lst = tf.split(basis, total_number, axis = -1) # [B, H, W, 15] => 5 x [B, H, W, 3]
        res_basis_lists = []
        for idx in choices:
            res_basis_lists.append(basis_lst[idx])
        return tf.concat(res_basis_lists, axis = -1)

'''
def load_noise_texture(args):

    #noise_texture_channel = 6
    shape = (args.texture_size,args.texture_size,args.noise_texture_channels)
    #noise = tf.random.uniform(shape,minval=-1,maxval=1,dtype=tf.float32)
    noise_ = np.asarray(np.random.uniform(-1,1,size=shape),dtype=np.float32)
    noise = tf.get_variable(name="neural_texture_noise_%d" % 1, dtype=tf.float32, trainable=False,initializer=noise_)
    mask = tf.get_variable(name="neural_texture_noise_%d" % 0, shape=shape,dtype=tf.float32,trainable=True,initializer=tf.initializers.constant(1))
    #mask = tf.get_variable(name="neural_texture_noise_%d" % 0, shape=shape, dtype=tf.float32, trainable=True)
    noise_mipmap = create_neural_texture_from_value(args, noise)
    texture_mipmap = create_neural_texture_from_value(args, mask)

    # hack texture channels
    #tmp = args.texture_channels
    #args.texture_channels = args.noise_texture_channel
    #texture_mipmap = create_neural_texture(args)
    #args.texture_channels = tmp

    return noise_mipmap,texture_mipmap
'''

'''
def load_other_texture(args):

    texture_path = args.other_texture

    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / (np.expand_dims(l2, axis) if len(a.shape) > 1 else l2)

    axay = exr_loader.EXRLoaderRGB(os.path.join(texture_path,"axay_fitted.exr"))
    axay = np.asarray(axay[:2],dtype=np.float32)
    axay = np.transpose(axay,[1,2,0]) # ax,ay
    #axay = axay - 0.5

    normal = exr_loader.EXRLoaderRGB(os.path.join(texture_path,"normal_fitted_global.exr"))
    normal = np.asarray(normal,dtype=np.float32)
    normal = np.transpose(normal,[1,2,0]) - 0.5
    normal = normalized(normal,axis=-1)

    tangent = exr_loader.EXRLoaderRGB(os.path.join(texture_path,"tangent_fitted_global.exr"))
    tangent = np.asarray(tangent,dtype=np.float32)
    tangent = np.transpose(tangent,[1,2,0]) - 0.5
    tangent = normalized(tangent, axis=-1)

    ax_ay_n_t = np.concatenate([axay,normal,tangent],axis=-1)
    ax_ay_n_t = tf.convert_to_tensor(ax_ay_n_t,dtype=tf.float32)

    pd = exr_loader.EXRLoaderRGB(os.path.join(texture_path,"pd_fitted_cc.exr"))
    ps = exr_loader.EXRLoaderRGB(os.path.join(texture_path,"ps_fitted_cc.exr"))
    pd = np.asarray(pd,dtype=np.float32)
    ps = np.asarray(ps,dtype=np.float32)
    pd = np.transpose(pd,[1,2,0])
    ps = np.transpose(ps,[1,2,0])
    pd_ps = np.concatenate([pd,ps],axis=-1)

    pd_ps = tf.convert_to_tensor(pd_ps,dtype=tf.float32)
    mapped_pd_ps = args.mapper.map_texture(pd_ps)

    args.logger.info('use other texture from %s' % args.other_texture)

    other_texture = tf.concat([ax_ay_n_t,mapped_pd_ps],axis=-1)
    return create_neural_texture_from_value(args, other_texture)
'''

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

            sampled_texture_other = None
            #noise_input = None
            #noise_mask = None
            #multiplied_noise = None

            # if other_texture is not None:
            #     sampled_texture_other = sample_texture_lod_bias(other_texture,uv, lod, args)
            #
            # if noise_texture is not None:
            #     noise_input = sample_texture_lod_bias(noise_texture[0],uv,lod,args)
            #     noise_mask = sample_texture_lod_bias(noise_texture[1],uv,lod,args)

        with tf.variable_scope("multipy_basis_and_sampled_texture"):
            assert (args.texture_channels % int(basis_count * 3) == 0)

            n_times = int(args.texture_channels / (basis_count * 3))
            reduced_basis = tf.concat([basis] * n_times, axis=-1, name="concat_more_times")

            mapped_reduced_basis = args.mapper.map_input(reduced_basis)
            mapped_sampled_texture = args.mapper.map_texture(sampled_texture)

            multiplied_texture = mapped_reduced_basis + mapped_sampled_texture
            net_input = multiplied_texture

            # if other_texture is not None:
            #     net_input = tf.concat([net_input,sampled_texture_other],axis=-1)
            #
            # if noise_texture is not None:
            #     multiplied_noise = noise_input * noise_mask
            #     net_input = tf.concat([net_input, multiplied_noise],axis=-1)


            # if ref_cg is not None:
            #     mapped_ref_cg = args.mapper.map_input(ref_cg)
            #     net_input = tf.concat([net_input,mapped_ref_cg],axis=-1)


    # 2. neural renderer
    with tf.variable_scope("unet"):
        output = resnet_cyclegan(net_input, 3, args.activation, 'render', args)
        reconstruct = args.mapper.map_output(output)

    #return sampled_texture, sampled_texture_other , multiplied_noise, reduced_basis, multiplied_texture, output, reconstruct
    return sampled_texture, reduced_basis, multiplied_texture, output, reconstruct

#Model = collections.namedtuple("Model", "train_op, aug_train_op, summary_op, loss, vars, output")
Model = collections.namedtuple("Model", "train_op, summary_op, loss, vars, output")

def create_model(dataset, args):
    # define network
    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)

        # other_texture = None
        # if args.other_texture :
        #     other_texture = load_other_texture( args)

        # noise_texture = None
        # if args.use_noise_texture:
        #     with tf.variable_scope("create_noise_texture"):
        #         noise_texture = load_noise_texture( args)

        # ref_cg_image = None
        # if args.use_cg_image:
        #     ref_cg_image = dataset.color_cg

        lod = None
        if "lod" in dataset._fields:
            lod = dataset.lod

        #sampled_texture, sampled_texture_other, multiplied_noise, reduced_basis, multiplied_texture, output, reconstruct = neural_render(neural_texture, dataset.uv, lod, dataset.index, dataset.basis, args)
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
        #if args.use_cg_image:
        #    summary_writer.add_image("image(Ref)",dataset.color_cg * dataset.mask, rescale_factor=args.rescale_output,linear=args.linear)
        summary_writer.add_image("image(recon)", reconstruct, rescale_factor=args.rescale_output,linear=args.linear)

        summary_writer.add_image("sampled_texture", sampled_texture, channels=args.texture_channels)
        # if sampled_texture_other is not None:
        #     summary_writer.add_image("sampled_texture_other", sampled_texture_other[...,-6:],channels=6)
        # if multiplied_noise is not None:
        #     summary_writer.add_image("multiplied_noise", multiplied_noise)
        summary_writer.add_image("basis", reduced_basis, channels=args.texture_channels)
        summary_writer.add_image("multiplied_texture", multiplied_texture, channels=args.texture_channels)

        for i in range(args.texture_levels):
            summary_writer.add_image('neural_texture_level_%d(0-2)' % i, tf.clip_by_value(neural_texture[i][..., :3][tf.newaxis, ...], 0, 1), channels=3, batch=1)

    summary_op = tf.summary.merge(summary_writer.lists)

    return Model(train_op=train_op,
        #aug_train_op=aug_train_op,
        summary_op=summary_op,
        loss=loss,
        vars=tf_vars,
        output=reconstruct)



def create_test_model(dataset, args):

    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)

        # other_texture = None
        # if args.other_texture:
        #     other_texture = load_other_texture(args)

        # noise_texture = None
        # if args.use_noise_texture:
        #     with tf.variable_scope("create_noise_texture"):
        #         noise_texture = load_noise_texture( args)

        # ref_cg_image = None
        # if args.use_cg_image:
        #     ref_cg_image = dataset.color_cg * args.rescale_input

        lod = None
        if "lod" in dataset._fields:
            lod = dataset.lod

        basis = dataset.basis * args.rescale_input

        #sampled_texture, sampled_texture_other, multiplied_noise, reduced_basis, multiplied_texture, output, reconstruct = neural_render(
        #    neural_texture, ref_cg_image, other_texture, noise_texture, dataset.uv, lod, dataset.index, basis, args)

        sampled_texture, reduced_basis, multiplied_texture, output, reconstruct = neural_render(neural_texture, dataset.uv, lod, dataset.index, basis, args)

    return reconstruct

