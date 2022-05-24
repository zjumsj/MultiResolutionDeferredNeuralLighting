import sys
sys.path.append('./src')
import configargparse
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from IO_mapper import LogIOMapper
from util import str2bool, set_random_seed, initial_logger, restore_model
from neural_render import create_model
from data_min_size import load_data,load_test_data

parser = configargparse.ArgumentParser()
parser.add_argument('--config', is_config_file = True,
                    help = 'config file path')
parser.add_argument('--dataDir', type=str,  default = None)
parser.add_argument('--logDir', type=str,  default = None)
parser.add_argument('--summaryDir', type=str,  default= None)
parser.add_argument('--checkpoint', type=str,  default= None,
                    help = 'path for loading pre-trained model if assigned')
parser.add_argument('--checkpoint_step',type=int,default = None,
                    help = 'step for pre-trained model, None for latest model')

######### test config
parser.add_argument('--test_only', type=str2bool, default = False,
                    help = 'enabled for test')
parser.add_argument('--test_scale', type=float, default = 1.)
parser.add_argument('--test_savedir', type=str, default = None)
parser.add_argument('--test_dataset', type=str, default = None)
########

parser.add_argument('--rescale_input', type=float, default = 1.) # scale input
parser.add_argument('--rescale_output', type=float, default = 1.) # scale output, only affect display

parser.add_argument('--start_step', type=int, default = 0)
parser.add_argument('--max_steps', type=int, default = 200000) # 200k
parser.add_argument('--save_freq', type=int, default = 20000) # 20k
parser.add_argument('--display_freq', type=int, default = 400)
parser.add_argument('--batch_size', type=int, default = 1)
parser.add_argument('--seed', type=int, default = 195829826)
parser.add_argument('--lr', type=float, default = 0.0002 )
parser.add_argument('--max_to_keep', type=int, default = 100)

parser.add_argument("--shuffle", type=str2bool, default = True)
parser.add_argument("--loss", type=str, default="l1", choices=["l1", "l2", "l1_l1grad", "l2_l1grad", "l1_l2grad", "l2_l2grad"])

parser.add_argument('--texture_size', type=int, default = 512,
                    help = 'width & height of neural texture')
parser.add_argument('--texture_channels', type=int, default = 30,
                    help = 'channel number of neural texture')
parser.add_argument('--texture_levels', type=int, default = 4,
                    help = 'mipmap level for neural texture')
parser.add_argument("--texture_init", type=str, default = "glorot_uniform", choices=["normal", "glorot_uniform", "zeros", "ones", "uniform"])
parser.add_argument("--mipmap", type=str2bool, default = True,
                    help = 'enable neural texture mipmap')

parser.add_argument("--LDR",type=str2bool, default = False,
                    help = 'TODO')
parser.add_argument("--linear",type=str2bool, default = False,
                    help = 'TODO')

parser.add_argument("--crop_type",type=int, default = 0)
parser.add_argument("--crop_w",type=int, default = 512,
                    help = 'width of random crop window')
parser.add_argument("--crop_h",type=int, default = 512,
                    help = 'height of random crop window')

parser.add_argument("--basis_lists", type=str, default="0,1,2,3,4")
parser.add_argument('--data_max_val', type=float, default = 1.,
                    help = "max value used in log-mapping")
parser.add_argument('--keep_max_val', type=float, default = 1.,
                    help = "max value used in quantization") # used in quantize output

parser.add_argument("--activation", type=str, default = "tanh", choices=["none", "tanh"])

parser.add_argument('--ngf', type=int,  default = 64,
                    help = 'channel number of the first conv_layer')
parser.add_argument('--resnet_res_count', type=int,  default = 9,
                    help = 'number of res-block')
parser.add_argument('--resnet_conv_count', type=int,  default = 2,
                    help = 'number of downsample conv_layer')
parser.add_argument('--resnet_padding', type=str,  default = 'SYMMETRIC', choices=['REFLECT', 'CONSTANT', 'SYMMETRIC'])

parser.add_argument('--clip_grad', type=str2bool, default = False)
parser.add_argument('--device', default=None,
                    help = 'select a device if you have multiple GPUs')
args,unknown = parser.parse_known_args()

if len(unknown) != 0:
    print(unknown)
    exit(-1)
if args.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def main():

    set_random_seed(args)
    logger = initial_logger(args)
    args.logger = logger

    args.basis_lists = [int(i) for i in args.basis_lists.split(',')]
    args.mapper = LogIOMapper(args.data_max_val) # map data to log space

    dataset = load_data(args,rescale=args.rescale_input)

    # build network
    logger.info('------Build Network Structure------')
    model = create_model(dataset, args)

    # session initial
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep, var_list=[var for var in tf.global_variables() if 'Adam' not in var.name and 'train_op' not in var.name], save_relative_paths=True)
    if args.summaryDir is None:
        args.summaryDir = args.logDir
    train_writer = tf.summary.FileWriter(args.summaryDir, sess.graph)

    # initial variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if args.checkpoint is not None:
        restore_model(sess, saver, args.checkpoint, args)

    logger.info('---Start training...---')
    end_step = args.start_step + args.max_steps
    for step in tqdm(range(args.start_step, end_step), file=sys.stdout):
        def should(freq):
            return freq > 0 and (step % freq == 0 or step == end_step - 1)

        fetches = {
            'loss': model.loss,
            'output': model.output
        }

        if should(args.display_freq):  # generate summary
            fetches["summary"] = model.summary_op

        # update
        fetches["train_op"] = model.train_op
        results = sess.run(fetches)

        # display
        if should(args.display_freq):
            summary = results["summary"]
            _loss = results["loss"]

            train_writer.add_summary(summary, step)
            train_writer.flush()
            logger.info("Iter {:06d}, loss = {:04f}".format(step, _loss))

        if should(args.save_freq):
            saver.save(sess, args.logDir + r'/{}.tfmodel'.format(step))
            saver.save(sess, args.logDir + r'/latest_model.tfmodel')

    logger.info('---Finished Training.---')
    saver.save(sess, args.logDir + r'/{}.tfmodel'.format(step))
    saver.save(sess, args.logDir + r'/latest_model.tfmodel')

def test():

    import collections
    import cv2
    import glob
    import exr_loader
    from neural_render import create_test_model

    use_lod = False
    save_exr = False

    gamma = 1.0/2.2

    scale = args.test_scale
    args.logDir = args.test_savedir
    test_dataset = args.test_dataset

    uv_name_list = glob.glob(os.path.join(test_dataset,"*.uv.exr"))
    lod_name_list = glob.glob(os.path.join(test_dataset,"*.lod.exr"))
    basis_name_list = glob.glob(os.path.join(test_dataset,"*.basis.exr"))
    uv_name_list.sort()
    lod_name_list.sort()
    basis_name_list.sort()
    assert(len(uv_name_list) == len(basis_name_list))

    # test logDir should change
    set_random_seed(args)

    logger = initial_logger(args, dump_code=False)
    args.logger = logger

    args.basis_lists = [int(i) for i in args.basis_lists.split(',')]
    args.mapper = LogIOMapper(args.data_max_val)

    # define graph
    uv = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 2])
    lod = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 1])
    basis = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 15])

    if use_lod:
        TestDataset = collections.namedtuple("Dataset", "iterator, color, uv, lod, mask, basis,index")
        dataset = TestDataset(uv=uv,lod=lod,basis=basis,iterator=None,color=None,mask=None,index=None)
    else:
        TestDataset = collections.namedtuple("Dataset", "iterator, color, uv, mask, basis, index")
        dataset = TestDataset(uv=uv,basis=basis,iterator=None,color=None,mask=None,index=None)
    output = create_test_model(dataset,args)

    saver = tf.train.Saver(var_list=[var for var in tf.global_variables()])
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=config)

    restore_model(sess,saver,args.checkpoint,args)

    for idx in tqdm(range(len(uv_name_list))):

        fetches = {
            "output" : output
        }

        if use_lod:
            _uv, _lod, _basis = load_test_data(
                uv_name_list, lod_name_list, basis_name_list, [idx]
            )
            img = sess.run(fetches, feed_dict={uv: _uv, lod: _lod, basis: _basis})["output"]
        else:
            _uv,_basis = load_test_data(
                uv_name_list,
                basis_name_list,
                [idx]
            )
            img = sess.run(fetches,feed_dict={uv:_uv,basis:_basis})["output"]

        img = img[0]
        if save_exr:
            h,w,c = img.shape
            channels = [img[:,:,i].flatten() for i in range(c)]
            exr_loader.EXRWriter(os.path.join(args.test_savedir, "%05d.exr") % idx,w,h,channels,channel=c)
        else:
            final_img = img[...,::-1] * scale
            final_img = np.clip(final_img, 0, 1)
            final_img = final_img ** gamma
            final_img = np.clip(final_img * 255 + 0.5, 0, 255.99)
            final_img = np.array(final_img, dtype='uint8')
            cv2.imwrite(os.path.join(args.test_savedir, "%05d.png" % idx), final_img)


if __name__ == "__main__":
    if args.test_only:
        test()
    else:
        main()

