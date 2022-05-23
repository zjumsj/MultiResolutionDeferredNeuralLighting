# see vgg16.py

import tensorflow as tf
import numpy as np

# input is RGB in [0,255]

class vgg16:
    def __init__(self, imgs, reuse=False, trainable=False, layer_list = None):
        self.imgs = imgs
        self.layer_list = layer_list # should be a str list
        self.convlayers(reuse,trainable)

        #self.fc_layers()
        #self.probs = tf.nn.softmax(self.fc3l)

        #if weights is not None and sess is not None:
        #    self.load_weights(weights, sess)

    def convlayers(self,reuse,trainable):
        self.parameters=[]
        self.output = []

        #trainable = False

        with tf.variable_scope("vgg16",reuse=reuse):

            # zero-mean input
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

            # conv1_1
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), trainable=trainable,name='conv1_1/weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=trainable, name='conv1_1/biases')
            conv1_1 = tf.nn.bias_add(conv, biases)
            relu1_1 = tf.nn.relu(conv1_1, name='conv1_1/relu')
            self.parameters += [kernel, biases]
            if "conv1_1" in self.layer_list:
                self.output.append(conv1_1)
            if "relu1_1" in self.layer_list:
                self.output.append(relu1_1)

            # conv1_2
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), trainable=trainable, name='conv1_2/weights')
            conv = tf.nn.conv2d(relu1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=trainable, name='conv1_2/biases')
            conv1_2 = tf.nn.bias_add(conv, biases)
            relu1_2 = tf.nn.relu(conv1_2, name='conv1_2/relu')
            self.parameters += [kernel, biases]
            if "conv1_2" in self.layer_list:
                self.output.append(conv1_2)
            if "relu1_2" in self.layer_list:
                self.output.append(relu1_2)

            # pool1
            pool1 = tf.nn.max_pool(relu1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')
            if "pool1" in self.layer_list:
                self.output.append(pool1)

            # conv2_1
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), trainable=trainable, name='conv2_1/weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=trainable, name='conv2_1/biases')
            conv2_1 = tf.nn.bias_add(conv, biases)
            relu2_1 = tf.nn.relu(conv2_1, name='conv2_1/relu')
            self.parameters += [kernel, biases]
            if "conv2_1" in self.layer_list:
                self.output.append(conv2_1)
            if "relu2_1" in self.layer_list:
                self.output.append(relu2_1)

            # conv2_2
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1),trainable=trainable, name='conv2_2/weights')
            conv = tf.nn.conv2d(relu2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=trainable, name='conv2_2/biases')
            conv2_2 = tf.nn.bias_add(conv, biases)
            relu2_2 = tf.nn.relu(conv2_2, name='conv2_2/relu')
            self.parameters += [kernel, biases]
            if "conv2_2" in self.layer_list:
                self.output.append(conv2_2)
            if "relu2_2" in self.layer_list:
                self.output.append(relu2_2)

            # pool2
            pool2 = tf.nn.max_pool(relu2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')
            if "pool2" in self.layer_list:
                self.output.append(pool2)

            # conv3_1
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), trainable = trainable,name='conv3_1/weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=trainable, name='conv3_1/biases')
            conv3_1 = tf.nn.bias_add(conv, biases)
            relu3_1 = tf.nn.relu(conv3_1, name='conv3_1/relu')
            self.parameters += [kernel, biases]
            if "conv3_1" in self.layer_list:
                self.output.append(conv3_1)
            if "relu3_1" in self.layer_list:
                self.output.append(relu3_1)

            # conv3_2
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=trainable,name='conv3_2/weights')
            conv = tf.nn.conv2d(relu3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=trainable, name='conv3_2/biases')
            conv3_2 = tf.nn.bias_add(conv, biases)
            relu3_2 = tf.nn.relu(conv3_2, name='conv3_2/relu')
            self.parameters += [kernel, biases]
            if "conv3_2" in self.layer_list:
                self.output.append(conv3_2)
            if "relu3_2" in self.layer_list:
                self.output.append(relu3_2)

            # conv3_3
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), trainable=trainable,name='conv3_3/weights')
            conv = tf.nn.conv2d(relu3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=trainable, name='conv3_3/biases')
            conv3_3 = tf.nn.bias_add(conv, biases)
            relu3_3 = tf.nn.relu(conv3_3, name='conv3_3/relu')
            self.parameters += [kernel, biases]
            if "conv3_3" in self.layer_list:
                self.output.append(conv3_3)
            if "relu3_3" in self.layer_list:
                self.output.append(relu3_3)

            # pool3
            pool3 = tf.nn.max_pool(relu3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')
            if "pool3" in self.layer_list:
                self.output.append(pool3)

            # conv4_1
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),trainable=trainable, name='conv4_1/weights')
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=trainable, name='conv4_1/biases')
            conv4_1 = tf.nn.bias_add(conv, biases)
            relu4_1 = tf.nn.relu(conv4_1, name='conv4_1/relu')
            self.parameters += [kernel, biases]
            if "conv4_1" in self.layer_list:
                self.output.append(conv4_1)
            if "relu4_1" in self.layer_list:
                self.output.append(relu4_1)

            # conv4_2
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),trainable=trainable, name='conv4_2/weights')
            conv = tf.nn.conv2d(relu4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=trainable, name='conv4_2/biases')
            conv4_2 = tf.nn.bias_add(conv, biases)
            relu4_2 = tf.nn.relu(conv4_2, name='conv4_2/relu')
            self.parameters += [kernel, biases]
            if "conv4_2" in self.layer_list:
                self.output.append(conv4_2)
            if "relu4_2" in self.layer_list:
                self.output.append(relu4_2)

            # conv4_3
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),trainable=trainable, name='conv4_3/weights')
            conv = tf.nn.conv2d(relu4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=trainable, name='conv4_3/biases')
            conv4_3 = tf.nn.bias_add(conv, biases)
            relu4_3 = tf.nn.relu(conv4_3, name='conv4_3/relu')
            self.parameters += [kernel, biases]
            if "conv4_3" in self.layer_list:
                self.output.append(conv4_3)
            if "relu4_3" in self.layer_list:
                self.output.append(relu4_3)

            # pool4
            pool4 = tf.nn.max_pool(relu4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
            if "pool4" in self.layer_list:
                self.output.append(pool4)

            # conv5_1
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), trainable=trainable,name='conv5_1/weights')
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=trainable, name='conv5_1/biases')
            conv5_1 = tf.nn.bias_add(conv, biases)
            relu5_1 = tf.nn.relu(conv5_1, name='conv5_1/relu')
            self.parameters += [kernel, biases]
            if "conv5_1" in self.layer_list:
                self.output.append(conv5_1)
            if "relu5_1" in self.layer_list:
                self.output.append(relu5_1)

            # conv5_2
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),trainable=trainable, name='conv5_2/weights')
            conv = tf.nn.conv2d(relu5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=trainable, name='conv5_2/biases')
            conv5_2 = tf.nn.bias_add(conv, biases)
            relu5_2 = tf.nn.relu(conv5_2, name='conv5_2/relu')
            self.parameters += [kernel, biases]
            if "conv5_2" in self.layer_list:
                self.output.append(conv5_2)
            if "relu5_2" in self.layer_list:
                self.output.append(relu5_2)

            # conv5_3
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),trainable=trainable, name='conv5_3/weights')
            conv = tf.nn.conv2d(relu5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=trainable, name='conv5_3/biases')
            conv5_3 = tf.nn.bias_add(conv, biases)
            relu5_3 = tf.nn.relu(conv5_3, name='conv5_3/relu')
            self.parameters += [kernel, biases]
            if "conv5_3" in self.layer_list:
                self.output.append(conv5_3)
            if "relu5_3" in self.layer_list:
                self.output.append(relu5_3)

            # pool5
            pool5 = tf.nn.max_pool(relu5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool5')
            if "pool5" in self.layer_list:
                self.output.append(pool5)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))
            if i>=25: # skip dense params
                break


#def perceptual_loss_vgg16(sess,criterion,img1,img2,layer_list,weight_list,first_call=True):
def perceptual_loss_vgg16(criterion,img1,img2,layer_list,weight_list,first_call=True):

    assert(len(weight_list) == len(layer_list))

    weight_ = np.asarray(weight_list,dtype=np.float32)
    weight_ = weight_ / np.sum(weight_) # normalize it

    if first_call:
        #model1 = vgg16(img1,reuse=False,trainable=False,weights='vgg/vgg16_weights.zip',sess=sess,layer_list=layer_list)
        model1 = vgg16(img1,reuse=False,trainable=False,layer_list=layer_list)
    else:
        model1 = vgg16(img1,reuse=True,trainable=False,layer_list=layer_list)
    #model2 = vgg16(img2,reuse=True,trainable=False,sess=sess,layer_list=layer_list)
    model2 = vgg16(img2,reuse=True,trainable=False,layer_list=layer_list)

    out1 = model1.output
    out2 = model2.output

    loss = tf.zeros(shape=(), dtype=tf.float32)
    for idx in range(len(out1)):
        loss = loss + criterion(out1[idx],out2[idx]) * weight_[idx]
    return loss , model1

# model validation
if __name__ == "__main__":
    import cv2
    imgs = tf.placeholder(tf.float32,[1,224,224,3])

    model = vgg16(imgs,False,False,"pool5")
    sess = tf.Session()
    model.load_weights('vgg16_weights.zip',sess)

    img1 = cv2.imread('D:\\PycharmProjects\\test_vgg\\cat_224.jpg')
    img1 = np.asarray(img1, dtype=np.float32)
    img1 = img1[np.newaxis, :, :, ::-1]  # to RGB

    output_ = sess.run(model.output, feed_dict={imgs: img1})
    print(output_[0].shape)

    ref = np.load('vgg16_pool5.npy')
    this = output_[0]
    diff = np.abs(ref - this)
    print(np.mean(diff),diff.max())
    print(this.max(),this.min())
    print(ref.max(),ref.min())