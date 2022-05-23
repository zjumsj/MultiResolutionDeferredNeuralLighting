import numpy as np
import glob
import os
import tensorflow as tf
import collections

import sys
sys.path.append("./src")
import exr_loader

Dataset = collections.namedtuple("Dataset", "iterator, color, uv, mask, basis, index")

def parse_tfrecord_np_shape(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value 
    data = ex.features.feature['color_basis'].bytes_list.value[0] 
    return np.fromstring(data, np.float16).reshape(shape)

def parse_tfrecord_tf(record, args):
    # probably data is encoded as bytes(string?)
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'color_basis': tf.FixedLenFeature([], tf.string),
        'uv': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string),
        'index': tf.FixedLenFeature([], tf.string)
        })
    
    shape_color_basis = features["shape"]
    shape_uv = tf.convert_to_tensor([shape_color_basis[0],shape_color_basis[1], 2], dtype=tf.int64)
    shape_mask = tf.convert_to_tensor([shape_color_basis[0],shape_color_basis[1], 1], dtype=tf.int64)
   
    color_basis = tf.decode_raw(features['color_basis'], tf.float16)
    uv = tf.decode_raw(features['uv'], tf.float32)
    mask = tf.decode_raw(features['mask'], tf.uint8)

    idx = tf.decode_raw(features['index'], tf.int32)
    
    return tf.reshape(color_basis, shape_color_basis), tf.reshape(uv, shape_uv), tf.reshape(mask, shape_mask), idx


def load_data_iterator(args, data_dir=None, seed = None):
    if data_dir is None:
        data_dir = args.dataDir

    batch_size = args.batch_size
    
    assert os.path.isdir(data_dir)
    tfr_files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecords')))
    # TODO: msj modified, read all tfrecords
    if len(tfr_files) == 0:
        for i_part in range(13):
            tfr_files = tfr_files + glob.glob(os.path.join(data_dir,'Cluster_%d/*.tfrecords' % i_part))
        tfr_files = sorted(tfr_files)
    assert len(tfr_files) >= 1

    args.logger.info("Load data from: " + str(tfr_files))

    tfr_shapes = []
    for tfr_file in tfr_files: # go through all found tfrecords
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
            tfr_shapes.append(parse_tfrecord_np_shape(record).shape) # return an iterator of serialized TFRecords
            break
    args.logger.info(tfr_shapes)
    
    args.logger.info("Loading and mapping data")
    dataset = tf.data.TFRecordDataset(filenames=tfr_files)
    dataset = dataset.map(lambda x : parse_tfrecord_tf(x, args), num_parallel_calls=16)

    if args.shuffle:
        args.logger.info("shuffling data")
        if 'shuffle_buffer_size' in args:
            shuffle_buffer_size = args.shuffle_buffer_size
        else:
            shuffle_buffer_size = 256
        if seed is None:
            seed = args.seed
        dataset = dataset.shuffle(buffer_size = shuffle_buffer_size, reshuffle_each_iteration=True, seed=seed)

    args.logger.info("Repeat and batch data")
    dataset = dataset.repeat() # make examples seen multiple times
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size = batch_size) # prepare while the current element is being processed
    # improve latency and throughput
    
    args.logger.info("make_one_shot_iterator")
    iterator = dataset.make_one_shot_iterator()

    return iterator, tfr_shapes


def postProcessData(iterator,tfr_shapes, args):
    batch_size = args.batch_size
    batch_color_basis, batch_uv, batch_mask, batch_indices = iterator.get_next() # get dataset

    batch_color_basis = tf.cast(batch_color_basis, tf.float32) 
    batch_color_basis = tf.maximum(batch_color_basis, 0)

    batch_mask = tf.cast(batch_mask, tf.float32) / 255.0
    batch_mask = tf.clip_by_value(batch_mask, 0.0, 1.0)

    resolution = tfr_shapes[0][0]

    # color + basis
    batch_color = batch_color_basis[...,0:3]
    batch_basis = batch_color_basis[...,3:]
   
    batch_color.set_shape([batch_size, resolution, resolution, 3])
    batch_uv.set_shape([batch_size, resolution, resolution, 2])
    batch_basis.set_shape([batch_size, resolution, resolution, tfr_shapes[0][-1] - 3 ])
    batch_mask.set_shape([batch_size, resolution, resolution, 1])
    batch_indices.set_shape([batch_size, 1])
    
    return batch_color, batch_uv, batch_mask, batch_basis, batch_indices

def load_data(args, path=None, seed= None, rescale=None):
    iterator, tfr_shapes = load_data_iterator(args, path, seed)
    batch_color, batch_uv,  batch_mask, batch_basis, batch_indices = postProcessData(iterator,tfr_shapes, args)
    
    if rescale is not None:
        batch_color *= rescale
        batch_basis *= rescale
        
    return Dataset (iterator = iterator, 
        color = batch_color, 
        uv = batch_uv,
        mask = batch_mask,
        basis = batch_basis,
        index = batch_indices, 
      )

##############################################

def load_test_data(uv_name_list,basis_name_list,idx_list):

    uv_data = []
    basis_data = []

    basis_label = []
    for ii in range(5):
        basis_label = basis_label + ['basis%d.R' % ii, 'basis%d.G' % ii, 'basis%d.B' % ii]

    for idx in idx_list:
        tmp = exr_loader.EXRLoaderRGBX(uv_name_list[idx],["uv.X","uv.Y"])
        tmp = np.asarray(tmp,dtype=np.float32)
        tmp = np.transpose(tmp,[1,2,0])
        uv_data.append(tmp)

        tmp = exr_loader.EXRLoaderRGBX(basis_name_list[idx],basis_label)
        tmp = np.asarray(tmp,dtype=np.float32)
        tmp = np.transpose(tmp,[1,2,0])
        basis_data.append(tmp)

    # make batch
    uv_data = np.stack(uv_data,axis=0)
    basis_data = np.stack(basis_data,axis=0)

    # no crop
    return uv_data,basis_data


