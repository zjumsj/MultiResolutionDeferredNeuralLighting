import sys
sys.path.append('./src')
import argparse
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np

# NOTICE !!
#import config.config_exp2_progressive as config

# TODO, remove src
from src.IO_mapper import LogIOMapper
from src.util import str2bool, set_random_seed, initial_logger
#from src.neural_render_multires_progressive import create_model
from src.neural_render_multires import  create_model
from src.gan_net import PGGAN_D,SRGAN_D
from src.adversarial_loss import G_gan,D_gan,G_wgan,D_wgan,G_lsgan,D_lsgan
from src.data_min_size import load_data, load_test_data
from src.ops import create_train_op , GDSummaryWriter, compute_number_of_parameters
import collections

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type = int, default = -1,
                    help = "training stage, used in progressive training")
parser.add_argument('--dataDir', type=str,  default = None)
parser.add_argument('--logDir', type=str,  default = None)
parser.add_argument('--summaryDir', type=str,  default = None)
parser.add_argument('--checkpoint', type=str,  default = None,
                    help = 'path for loading pre-trained model if assigned')
parser.add_argument('--checkpoint_step',type=int,default = None,
                    help = 'step for pre-trained model, None for latest model')

######### test config
parser.add_argument('--test_only', type=str2bool, default = False,
                    help = 'enabled for test')
parser.add_argument('--test_scale', type=float, default = 1.)
parser.add_argument('--test_savedir', type=str, default = None)
parser.add_argument('--test_dataset', type=str, defautt = None)
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

# common loss
parser.add_argument("--loss", type=str, default="l1", choices=["l1", "l2", "l1_l1grad", "l2_l1grad", "l1_l2grad", "l2_l2grad"])
parser.add_argument("--loss_scale",type=float,default=1.,
                    help = "loss scale for common l1 or l2 loss")
# vgg loss
parser.add_argument("--vgg_loss",type=str,default="l2",choices=["l1", "l2", "l1_l1grad", "l2_l1grad", "l1_l2grad", "l2_l2grad"],
                    help = "metric used in VGG feature space")
parser.add_argument("--vgg_loss_scale",type=float,default=0., # 1.224e-5
                    help = "loss scale for VGG loss")
parser.add_argument("--vgg_layer_list",type=str,default="conv1_2",
                    help = "assign layers to compute loss")
parser.add_argument("--vgg_blend_weights",type=str,default="1",
                    help = "assign weights on different VGG layers")
# adv loss
parser.add_argument("--lr_D",type=float,default=0.,
                    help = "learning rate for discriminator") # TODO
parser.add_argument("--adv_loss_scale",type=float,default=config.adv_loss_scale)
parser.add_argument('--adv_loss',type=str,default=config.adv_loss,choices=["gan","wgan-gp","lsgan"])

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
                    help = "max value used in quantization")

parser.add_argument("--activation", type=str, default = "tanh", choices=["none", "tanh"])

# G
parser.add_argument('--ngf', type=int,  default = 256,
                    help = 'channel number of Feature Transform Modules')
parser.add_argument('--module_count', type=int,  default = 5,
                    help = 'number of Feature Transform Modules')
parser.add_argument('--conv_count', type=int,  default = 2,
                    help = 'layer number in each Feature Transform Module')
parser.add_argument('--resnet_padding', type=str,  default = 'SYMMETRIC', choices=['REFLECT', 'CONSTANT', 'SYMMETRIC'])
# D
parser.add_argument('--ngf_D',type=int,default=config.ngf_D) # TODO
parser.add_argument('--ngf_D_max',type=int,default=config.ngf_D_max)
parser.add_argument('--D_levels',type=int,default=config.D_levels)
parser.add_argument('--D_repeats',type=int,default=config.D_repeats)

parser.add_argument('--clip_grad', type=str2bool, default = False)
parser.add_argument('--device',default=None)
args,unknown = parser.parse_known_args()

if len(unknown) != 0:
    print(unknown)
    exit(-1)
if args.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def restore_model(sess, prefix, saver, checkpoint, args):
    if checkpoint is not None:
        args.logger.info('Restore {}'.format(checkpoint))
        if args.checkpoint_step is not None:
            ckpt = os.path.join(checkpoint, prefix + "%s.tfmodel" % args.checkpoint_step)
        else:
            ckpt = os.path.join(checkpoint, prefix + "latest_model.tfmodel")
        print('Restore {}.'.format(ckpt))
        saver.restore(sess, ckpt)
        return True
    else:
        args.logger.info('[ERROR] Restore failed {}.'.format(checkpoint))
        return False

def create_summary(scalar,images,args):
    summary_writer = GDSummaryWriter(args.batch_size)
    with tf.name_scope("tensorboard_visualize2"):
        # scalar
        for (key, value) in scalar.items():
            summary_writer.add_scalar(key,value)
        # images
        for (key, value) in images.items():
            summary_writer.add_image(key,value,linear=True)

    summary_op = tf.summary.merge(summary_writer.lists)
    return summary_op

# TODO, are you sure you have lod ?
Dataset = collections.namedtuple(
    "Dataset",
    "iterator,color,uv,lod,mask,basis,index"
)


# TODO, are you sure you support lod ?
def stupid_create_placeholder(args):
    batch_size = args.batch_size
    resolution_h = args.crop_h
    resolution_w = args.crop_w

    dataset = Dataset(
        iterator = None,
        color = tf.placeholder(tf.float32,[batch_size,resolution_h,resolution_w,3]),
        uv = tf.placeholder(tf.float32,[batch_size,resolution_h,resolution_w,2]),
        lod = tf.placeholder(tf.float32,[batch_size,resolution_h,resolution_w,1]),
        mask = tf.placeholder(tf.float32,[batch_size,resolution_h,resolution_w,1]),
        basis = tf.placeholder(tf.float32,[batch_size,resolution_h,resolution_w,15]),
        index = tf.placeholder(tf.int32,[batch_size,1]) # not sure
    )
    return dataset

def stupid_get_data(sess,dataset,dataset_in):

    dataset_ = sess.run([
        dataset.color,
        dataset.uv,
        dataset.lod,
        dataset.mask,
        dataset.basis,
        dataset.index
    ])

    # feed this
    dataset_out = {
        dataset_in.color: dataset_[0],
        dataset_in.uv: dataset_[1],
        dataset_in.lod: dataset_[2],
        dataset_in.mask: dataset_[3],
        dataset_in.basis: dataset_[4],
        dataset_in.index: dataset_[5]
    }
    return dataset_out

def main():

    set_random_seed(args)
    logger = initial_logger(args)
    args.logger = logger

    args.vgg_layer_list = [s for s in args.vgg_layer_list.split(",")]
    args.vgg_blend_weights = [float(s) for s in args.vgg_blend_weights.split(",")]
    args.basis_lists = [int(i) for i in args.basis_lists.split(',')]
    args.mapper = LogIOMapper(args.data_max_val) # map data to log space

    dataset = load_data(args,rescale=args.rescale_input)

    dataset_in = dataset
    if args.adv_loss_scale != 0:
        dataset_in = stupid_create_placeholder(args)

    # TODO
    lod = tf.placeholder(tf.float32,[]) # create lod
    args.lod = lod

    G_model = create_model(dataset_in,args,first_call=True)
    g_loss_total = G_model.loss

    if args.adv_loss_scale != 0:

        logger.info('---You are using adv training---')

        real_D_params={
            "inputs": G_model.real,
            "name": "PGGAN_D",
            "ngf": args.ngf_D,
            "max_ngf": args.ngf_D_max,
            "levels": args.D_levels,
            "reuse": False
        }
        fake_D_params={
            "inputs": G_model.fake,
            "name": "PGGAN_D",
            "ngf": args.ngf_D,
            "max_ngf": args.ngf_D_max,
            "levels": args.D_levels,
            "reuse": True
        }
        if args.adv_loss == "gan":
            d_loss,output_summary,output_summary_image = D_gan(SRGAN_D,real_D_params,fake_D_params,
                                                               patch_gan=False
                                                               )
            g_loss = G_gan(SRGAN_D,fake_D_params,patch_gan=False)
        elif args.adv_loss == "wgan-gp":
            d_loss,output_summary,output_summary_image = D_wgan(PGGAN_D,real_D_params,fake_D_params,
                                           patch_gan=True,
                                           use_reduce_mean= True
                                           )
            g_loss = G_wgan(PGGAN_D,fake_D_params,patch_gan=True)
        elif args.adv_loss == "lsgan":
            d_loss,output_summary,output_summary_image = D_lsgan(PGGAN_D,real_D_params,fake_D_params,
                                            patch_gan=True,
                                            style = 1
                                            )
            g_loss = G_lsgan(PGGAN_D,fake_D_params,patch_gan=True,style=1)

        d_loss = tf.reduce_mean(d_loss)
        g_loss = tf.reduce_mean(g_loss)
        g_loss_total = g_loss_total + g_loss * args.adv_loss_scale

        tf_vars_D = tf.trainable_variables(scope = 'PGGAN_D')
        print("[info] D Pameters: #%d, Variables: #%d" % (compute_number_of_parameters(tf_vars_D), len(tf_vars_D)))
        train_op_D = create_train_op(args.lr_D,0.9,0.999,d_loss,tf_vars_D,"D",args)

        tf_vars_G = tf.trainable_variables(scope = "OffsetNetwork")
        print("[info] G Pameters: #%d, Variables: #%d" % (compute_number_of_parameters(tf_vars_G), len(tf_vars_G)))
        train_op_G = create_train_op(args.lr,0.9,0.999,g_loss_total,tf_vars_G,"G",args)
        #G_model.train_op = train_op_G

        summary2 = create_summary(output_summary,output_summary_image,args)

        ## create CPU session to read data
        config2 = tf.ConfigProto(device_count = {'GPU':0})
        sess2 = tf.Session(config=config2)

    # session initial
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #saver = tf.train.Saver(max_to_keep=args.max_to_keep, var_list=[var for var in tf.global_variables() if 'Adam' not in var.name and 'train_op' not in var.name], save_relative_paths=True)
    saver_G = tf.train.Saver(max_to_keep = args.max_to_keep,
                             var_list = [var for var in tf.global_variables() if ("OffsetNetwork" in var.name and "Adam" not in var.name and "train_op" not in var.name)])
    saver_D = None
    if args.adv_loss_scale != 0:
        saver_D = tf.train.Saver(max_to_keep = args.max_to_keep,
                                 var_list = [var for var in tf.global_variables() if ("PGGAN_D" in var.name and "Adam" not in var.name and "train_op" not in var.name)])
    if args.summaryDir is None:
        args.summaryDir = args.logDir
    train_writer = tf.summary.FileWriter(args.summaryDir, sess.graph)

    # initial variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    if G_model.aug_train_op is not None:
        logger.info('Load Vgg16 weights')
        G_model.aug_train_op.load_weights("vgg/vgg16_weights.zip",sess)

    if args.checkpoint is not None and args.checkpoint != "None":
        restore_model(sess, "G_", saver_G, args.checkpoint, args)
        if args.adv_loss_scale != 0:
            restore_model(sess,"D_", saver_D, args.checkpoint, args)

    ################################################################
    # descriminator warm up
    if args.start_step > 0 and args.adv_loss_scale != 0:
        logger.info('--Warm up Descriminator...---')
        for step in tqdm(range(0,args.start_step),file=sys.stdout):
            def should(freq):
                return freq > 0 and (step % freq == 0 or step == args.start_step - 1)

            data_feed_in = stupid_get_data(sess2, dataset, dataset_in)

            fetches = {
                'loss': d_loss,
                'train_op': train_op_D
            }
            if should(args.display_freq) :
                fetches["summary"] = summary2

            results = sess.run(fetches, feed_dict=data_feed_in)

            if should(args.display_freq):
                summary = results["summary"]
                train_writer.add_summary(summary, step)

            if should(args.save_freq):
                saver_D.save(sess, args.logDir + r'/D_{}.tfmodel'.format(step))
                saver_D.save(sess, args.logDir + r'/D_latest_model.tfmodel')

        logger.info('---Finished Warm up.---')
        saver_D.save(sess, args.logDir + r'/D_{}.tfmodel'.format(step))
        saver_D.save(sess, args.logDir + r'/D_latest_model.tfmodel')

    ##########################################

    logger.info('---Start training...---')
    end_step = args.start_step + args.max_steps

    for step in tqdm(range(args.start_step, end_step), file=sys.stdout):
        def should(freq):
            return freq > 0 and (step % freq == 0 or step == end_step - 1)
        lod_ = float(step - args.start_step) / float(end_step - args.start_step)

        if args.adv_loss_scale != 0:

            # FIXME: quite stupid and inefficient, but TFRecordDataset does not provide proper API
            data_feed_in = stupid_get_data(sess2, dataset, dataset_in)
            data_feed_in[lod] = lod_

            for iD in range(args.D_repeats):
                fetches = {
                    'loss': d_loss,
                    'train_op': train_op_D
                }
                if should(args.display_freq) and iD == args.D_repeats - 1:
                    fetches["summary"] = summary2

                results = sess.run(fetches, feed_dict=data_feed_in)

                if should(args.display_freq) and iD == args.D_repeats - 1:
                    summary = results["summary"]
                    train_writer.add_summary(summary, step)

        fetches = {
            'loss': g_loss_total,
            'output': G_model.output
        }

        if should(args.display_freq):  # generate summary
            fetches["summary"] = G_model.summary_op

        # update
        fetches["train_op"] = train_op_G if args.adv_loss_scale != 0 else G_model.train_op
        if args.with_aug:
            fetches["aug_train_op"] = G_model.aug_train_op

        if args.adv_loss_scale != 0:
            results = sess.run(fetches, feed_dict=data_feed_in)
        else:
            results = sess.run(fetches, feed_dict={lod:lod_})

        # display
        if should(args.display_freq):
            summary = results["summary"]
            _loss = results["loss"]

            train_writer.add_summary(summary, step)
            train_writer.flush()
            logger.info("Iter {:06d}, loss = {:04f}".format(step, _loss))

        if should(args.save_freq):
            saver_G.save(sess, args.logDir + r'/G_{}.tfmodel'.format(step))
            saver_G.save(sess, args.logDir + r'/G_latest_model.tfmodel')
            if saver_D is not None:
                saver_D.save(sess, args.logDir + r'/D_{}.tfmodel'.format(step))
                saver_D.save(sess, args.logDir + r'/D_latest_model.tfmodel')

    logger.info('---Finished Training.---')
    saver_G.save(sess, args.logDir + r'/G_{}.tfmodel'.format(step))
    saver_G.save(sess, args.logDir + r'/G_latest_model.tfmodel')
    if saver_D is not None:
        saver_D.save(sess, args.logDir + r'/D_{}.tfmodel'.format(step))
        saver_D.save(sess, args.logDir + r'/D_latest_model.tfmodel')

def test():
    import collections
    import cv2
    import sys
    sys.path.append("..")
    import exr_loader
    #from neural_render_capturedGGX_vgg import create_test_model
    from neural_render_exp2 import create_test_model

    TestDataset = collections.namedtuple("TestDataset", "iterator,color_gt,color_cg,uv,lod,mask,depth,basis,index")

    save_exr = False
    #scale = 3.0 # train
    scale = 1.0 # egg and etc
    gamma = 1.0/2.2
    #gamma = 1.0
    save_dir = "../val/empty"
    args.logDir = save_dir
    #test_dataset = "E:/generate_2x_lod"
    test_dataset = "F:/Kang_tvcg_data/egg_generate_2x"
    #test_dataset = "F:/Kang_tvcg_data/tower_generate_2x"
    #test_dataset = "F:/Kang_tvcg_data/golden_cloth_generate_2x"
    #test_dataset = "F:/Kang_tvcg_data/chips_generate_2x"
    #type = 2
    type = 3

    ## train
    #center_xy = [1020,1017]
    #window_xy = [832,640]
    ## egg
    center_xy = [1014,850]
    window_xy = [512,640]
    ## tower
    #center_xy = [1015,695]
    #window_xy = [512,832]
    ## cloth
    #center_xy = [1024,845]
    #window_xy = [1216,960]
    ## chips
    # center_xy_dir = {
    #     "0":(1020, 810),
    #     "2": (1000, 810),
    #     "4": (890, 850),
    #     "6": (805, 870),
    #     "8": (730, 890),
    #     "10": (750, 920),
    #     "12": (800, 955),
    #     "14": (800, 955),
    #     "16": (840, 955),
    #     "18": (950, 955),
    #     "20": (1180, 955),
    #     "22": (1250, 955)
    # }
    # window_xy = [1408,1088]

    uv_name_list = []
    lod_name_list = []
    basis_name_list = []
    cg_name_list = []
    #center_xy_list =[]# optional
    if type == 0:
        light_idx = 1
        # rotate object
        for idx in range(0,360):
            uv_name_list.append(os.path.join(test_dataset,"%d/UV_%05d.exr" % (idx,light_idx)))
            lod_name_list.append(os.path.join(test_dataset,"%d/LOD_%05d.exr" % (idx,light_idx)))
            basis_name_list.append(
                [os.path.join(test_dataset,"%d/mat%03d_%05d.exr" % (idx,mat_id,light_idx)) for mat_id in range(5)]
            )
            cg_name_list.append(os.path.join(test_dataset,"%d/img_%05d.exr" % (idx,light_idx)))

    elif type == 1:
        # rotate light
        for idx in range(0,360):
            uv_name_list.append(os.path.join(test_dataset,"UV_%05d.exr" % idx))
            lod_name_list.append(os.path.join(test_dataset,"LOD_%05d.exr" % idx))
            basis_name_list.append(
                [os.path.join(test_dataset,"mat%03d_%05d.exr" % (mat_id,idx)) for mat_id in range(5) ]
            )
            cg_name_list.append(os.path.join(test_dataset,"img_%05d.exr" % idx))
    elif type == 2:
        # get gt
        for rot_idx in range(0,24,2):
        #for rot_idx in range(0, 72, 2): # NOTICE!!
            for light_idx in range(0,16):
                if (rot_idx // 2 + light_idx) % 2 == 0:
                    continue
                uv_name_list.append(os.path.join(test_dataset,"%d/UV_%05d.exr") % (rot_idx,light_idx))
                lod_name_list.append(os.path.join(test_dataset, "%d/LOD_%05d.exr") % (rot_idx, light_idx))
                basis_name_list.append(
                    [os.path.join(test_dataset,'%d/mat%03d_%05d.exr' % (rot_idx,mat_id,light_idx)) for mat_id in range(5)]
                )
                cg_name_list.append(os.path.join(test_dataset,'%d/img_%05d.exr' % (rot_idx,light_idx)))
                # optional
                #center_xy_list.append(center_xy_dir['%d' % rot_idx])

    elif type == 3:
        idx_list = [(12,3)]
        for idx_group in idx_list:
            uv_name_list.append(os.path.join(test_dataset, "%d/UV_%05d.exr") % (idx_group[0], idx_group[1]))
            lod_name_list.append(os.path.join(test_dataset, "%d/LOD_%05d.exr") % (idx_group[0], idx_group[1]))
            basis_name_list.append(
                [os.path.join(test_dataset, '%d/mat%03d_%05d.exr' % (idx_group[0], mat_id, idx_group[1])) for mat_id in range(5)]
            )
            cg_name_list.append(os.path.join(test_dataset, '%d/img_%05d.exr' % (idx_group[0], idx_group[1])))

    # test logDir should change
    set_random_seed(args)

    logger = initial_logger(args, dump_code=False)
    args.logger = logger

    args.basis_lists = [int(i) for i in args.basis_lists.split(',')]
    args.mapper = LogIOMapper(args.data_max_val)

    uv = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,window_xy[1],window_xy[0],2])
    lod = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, window_xy[1], window_xy[0], 1])
    basis = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,window_xy[1],window_xy[0],15])
    color_cg = tf.placeholder(dtype=tf.float32,shape=[args.batch_size,window_xy[1],window_xy[0],3])

    # uv = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 2])
    # lod = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 1])
    # basis = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 15])
    # color_cg = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 3])
    # y_start2 = 20
    # x_start2 = 95

    dataset = TestDataset(uv=uv,lod = lod,basis=basis,iterator=None,color_cg=color_cg,color_gt=None,mask=None,depth=None,index=None)

    output = create_test_model(dataset,args)
    output *= args.rescale_output

    saver = tf.train.Saver(var_list=[var for var in tf.global_variables()])
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=config)

    restore_model(sess,"G_",saver,args.checkpoint,args)

    #for idx in tqdm(len(uv_name_list)):
    for idx in tqdm(range(len(uv_name_list))):

        if False: # optional
            center_xy = center_xy_list[idx]

        # suppose here load data
        _uv,_lod,_basis,_color_cg = load_test_data(
            uv_name_list,
            lod_name_list,
            basis_name_list,
            cg_name_list,
            center_xy,
            window_xy,
            [idx]
        )

        # _uv = _uv[:,y_start2:y_start2+512,x_start2:x_start2+512]
        # _lod = _lod[:,y_start2:y_start2+512,x_start2:x_start2+512]
        # _basis = _basis[:, y_start2:y_start2 + 512, x_start2:x_start2 + 512]

        fetches = {
            "output" : output
        }

        if args.use_cg_image:
            img = sess.run(fetches,feed_dict={uv:_uv,lod:_lod,basis:_basis,color_cg:_color_cg})["output"]
        else:
            img = sess.run(fetches,feed_dict={uv:_uv,lod:_lod,basis:_basis})["output"]
        img = img[0]

        if save_exr:
            h,w,c = img.shape
            channels = [img[:,:,i].flatten() for i in range(c)]
            exr_loader.EXRWriter(os.path.join(save_dir,"%05d.exr") % idx,w,h,channels,channel=c)

        else:
            final_img = img[...,::-1] * scale
            final_img = np.clip(final_img,0,1)

            final_img = final_img ** gamma
            final_img = np.clip(final_img * 255.0, 0, 255)

            cv2.imwrite(os.path.join(save_dir,"%05d.png" % idx),final_img)

if __name__ == "__main__":
    main()
    #test()
