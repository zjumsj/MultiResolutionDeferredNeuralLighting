import tensorflow as tf
import numpy as np
#from ops import GDSummaryWriter

# Reference progressive_growing_of_gans
# https://github.com/tkarras/progressive_growing_of_gans


#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

#----------------------------------------------------------------------------
# cross entropy

def G_gan(model,params_fake, patch_gan = True):

    fake_scores_out = model(**params_fake)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_scores_out),logits=fake_scores_out)
    if patch_gan: # -> inf
        loss = tf.reduce_mean(loss, axis=[1,2,3])

    return loss

def D_gan(model,params_real,params_fake,patch_gan=True):

    real_scores_out = model(**params_real)
    fake_scores_out = model(**params_fake)
    real_input = params_real["inputs"]
    fake_input = params_fake["inputs"]

    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_scores_out),logits=real_scores_out)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_scores_out),logits=fake_scores_out)

    output_summary = {}
    output_summary_image = {}

    if patch_gan:
        rd_real_loss = tf.reduce_mean(real_loss, axis=[1, 2, 3])
        rd_fake_loss = tf.reduce_mean(fake_loss, axis=[1, 2, 3])
        output_summary["real_scores_out"] = tf.reduce_mean(rd_real_loss)
        output_summary["fake_scores_out"] = tf.reduce_mean(rd_fake_loss)
        output_summary_image["real_scores_out"] = real_scores_out * 0.5 + 0.5 # notice here
        output_summary_image["fake_scores_out"] = fake_scores_out * 0.5 + 0.5
        loss = rd_fake_loss + rd_real_loss
    else:
        output_summary["real_scores_out"] = tf.reduce_mean(real_loss)
        output_summary["fake_scores_out"] = tf.reduce_mean(fake_loss)
        loss = real_loss + fake_loss

    if True:
        # add gradients info
        output_summary['gradients'] = tf.reduce_mean(
            #tf.abs(tf.gradients(loss,fake_input))
            tf.abs(tf.gradients(fake_scores_out,fake_input))
        )

    return loss,output_summary,output_summary_image


#----------------------------------------------------------------------------
# WGAN-GP

def G_wgan(model,params_fake, patch_gan = True):

    fake_scores_out = model(**params_fake)
    if patch_gan: # -> inf
        loss = -tf.reduce_mean(fake_scores_out, axis=[1,2,3])
    else:
        loss = -fake_scores_out
    return loss

def D_wgan(model,params_real,params_fake, patch_gan = True,
           use_reduce_mean = False,
           wgan_lambda=10.0,  # Weight for the gradient penalty term.
           wgan_epsilon=0.001,  # Weight for the epsilon term, \epsilon_{drift}.
           wgan_target=1.0):  # Target value for gradient magnitudes.

    real_scores_out = model(**params_real)
    fake_scores_out = model(**params_fake)
    real_input = params_real["inputs"]
    fake_input = params_fake["inputs"]

    batch_size = tf.shape(real_input)[0]

    output_summary = {}
    output_summary_image = {}

    if patch_gan:
        rd_real_scores_out = tf.reduce_mean(real_scores_out, axis=[1, 2, 3])
        rd_fake_scores_out = tf.reduce_mean(fake_scores_out, axis=[1, 2, 3])
        output_summary["real_scores_out"] = tf.reduce_mean(rd_real_scores_out)
        output_summary["fake_scores_out"] = tf.reduce_mean(rd_fake_scores_out)
        output_summary_image["real_scores_out"] = real_scores_out * 0.5 + 0.5
        output_summary_image["fake_scores_out"] = fake_scores_out * 0.5 + 0.5
        loss = rd_fake_scores_out - rd_real_scores_out  # fake-> -inf, real->inf
    else:
        output_summary["real_scores_out"] = tf.reduce_mean(real_scores_out)
        output_summary["fake_scores_out"] = tf.reduce_mean(fake_scores_out)
        loss = fake_scores_out - real_scores_out

    with tf.name_scope("GradientPenalty"):
        params_mixed = params_fake.copy()

        mixing_factors = tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=real_input.dtype)
        mixed_images_out = lerp(real_input, fake_input, mixing_factors)  # random sample
        params_mixed["inputs"] = mixed_images_out
        mixed_scores_out = model(**params_mixed) # run mixed result

        if patch_gan:

            rd_mixed_scores_out = tf.reduce_mean(mixed_scores_out, axis=[1, 2, 3])
            output_summary["mixed_scores"] = tf.reduce_mean(rd_mixed_scores_out)
            output_summary_image["mixed_scores"] = mixed_scores_out * 0.5 + 0.5
        else:

            output_summary["mixed_scores"] = tf.reduce_mean(mixed_scores_out)

        #mixed_loss = rd_mixed_scores_out
        mixed_loss = mixed_scores_out
        mixed_grads = tf.gradients(mixed_loss,[mixed_images_out])[0] # compute gradients


        if use_reduce_mean:
            mixed_norms = tf.sqrt(tf.reduce_mean(tf.square(mixed_grads), axis=[1, 2, 3]))  # remain batch dim
        else:
            mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1, 2, 3]))  # remain batch dim

        gradient_penalty = tf.square(mixed_norms - wgan_target)
        output_summary["gradient_penalty"] = tf.reduce_mean(gradient_penalty)

    loss += gradient_penalty * (wgan_lambda / (wgan_target ** 2))

    with tf.name_scope("EpsilonPenalty"):
        epsilon_penalty = tf.square(real_scores_out)
        if patch_gan:
            epsilon_penalty = tf.reduce_mean(epsilon_penalty, axis=[1, 2, 3])
        output_summary["epsilon_penalty"] = tf.reduce_mean(epsilon_penalty)
    loss += epsilon_penalty * wgan_epsilon

    if True:
        # add gradients info
        output_summary['gradients'] = tf.reduce_mean(
            tf.abs(tf.gradients(fake_scores_out,fake_input))
        )

    return loss, output_summary, output_summary_image

#------------------------------------------------------------------------
# LSGAN

def D_lsgan(model,params_real,params_fake, patch_gan = True, style = 1):

    real_scores_out = model(**params_real)
    fake_scores_out = model(**params_fake)


    D_loss_real = tf.square(real_scores_out - 1.) #-> 1
    if style == 0:
        D_loss_fake = tf.square(fake_scores_out + 1.) #-> -1
    else:
        D_loss_fake = tf.square(fake_scores_out) #->0

    output_summary = {}
    if patch_gan:
        rd_loss_real = tf.reduce_mean(D_loss_real,axis=[1,2,3])
        rd_loss_fake = tf.reduce_mean(D_loss_fake,axis=[1,2,3])
        output_summary["real_scores_out"] = rd_loss_real
        output_summary["fake_scores_out"] = rd_loss_fake
        loss = rd_loss_fake + rd_loss_real
    else:
        output_summary["real_scores_out"] = D_loss_real
        output_summary["fake_scores_out"] = D_loss_fake
        loss = D_loss_fake + D_loss_real
    return loss, output_summary

def G_lsgan(model,params_fake, patch_gan = True, style = 1):

    fake_scores_out = model(**params_fake)
    if style == 0:
        D_loss_fake = tf.square(fake_scores_out) # ->0
    else:
        D_loss_fake = tf.square(fake_scores_out - 1.) # ->1

    if patch_gan:
        loss = tf.reduce_mean(D_loss_fake,axis=[1,2,3])
    else:
        loss = D_loss_fake
    return loss

