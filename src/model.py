# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import logging
import math
import numpy as np
import tensorflow as tf

import utils as utils
import tensorflow_utils as tf_utils
from reader import Reader


class Model(object):
    def __init__(self, decode_img_shape=(320, 400, 1), output_shape=(320, 200, 1), num_classes=4,
                 data_path=(None, None), batch_size=1, lr=1e-3, total_iters=2e5, is_train=True,
                 log_dir=None, method=None, multi_test=True, resize_factor=0.5, use_dice_loss=False,
                 lambda_one=1.0, lambda_two=0.1, name='UNet'):
        self.decode_img_shape = decode_img_shape
        self.input_shape = output_shape
        self.output_shape = output_shape
        self.num_classes = num_classes
        self.method = method
        self.use_batch_norm = False
        self.is_train = is_train
        self.resize_factor = resize_factor
        self.use_dice_loss = use_dice_loss
        self.lambda_one = lambda_one
        self.lambda_two = lambda_two

        self.multi_test = False if self.is_train else multi_test
        self.degree = 10
        self.num_try = len(range(-self.degree, self.degree+1, 2))  # multi_tes: from -10 degree to 11 degrees
        self.gen_c = [32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16]
        self.dis_c = [32, 64, 128, 256, 512, 1]

        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.25)
        self.decay_steps = int(self.total_steps * 0.5)
        self.log_dir = log_dir
        self.name = name

        self.mIoU_metric, self.mIoU_metric_update = None, None
        self.tb_lr = None
        self.gen_ops = list()
        self.dis_ops = list()

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=self.log_dir, is_train=self.is_train, name=self.name)

        self._build_graph()             # main graph
        self._init_eval_graph()         # evaluation for validation data
        self._init_test_graph()         # for test data
        self._best_metrics_record()     # metrics
        self._init_tensorboard()        # tensorboard
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None, scope='Gen')
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None, scope=None)

    def _build_graph(self):
        # Input placeholders
        self.input_img_tfph = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.input_shape], name='input_tfph')
        self.rate_tfph = tf.compat.v1.placeholder(tf.float32, name='keep_prob_tfph')
        # self.train_mode_tfph = tf.compat.v1.placeholder(tf.bool, name='train_mode_ph')

        # Initialize TFRecoder reader
        train_reader = Reader(tfrecords_file=self.data_path[0],
                              decode_img_shape=self.decode_img_shape,
                              img_shape=self.input_shape,
                              batch_size=self.batch_size,
                              name='train')

        # Random batch for training
        self.img_train, self.seg_img_train = train_reader.shuffle_batch()

        # Initialize generator and discriminator
        self.gen_obj = Generator(name='Gen', conv_dims=self.gen_c, padding='SAME', logger=self.logger,
                                 resize_factor=self.resize_factor, norm='instance',
                                 num_classes=self.num_classes, _ops=self.gen_ops)
        self.dis_obj = Discriminator(name='Dis', dis_c=self.dis_c, norm='instance', logger=self.logger,
                                     _ops=self.dis_ops)

        # Network forward for training
        self.pred_train = self.gen_obj(input_img=self.normalize(self.img_train), keep_rate=self.rate_tfph)
        # self.pred_cls_train = tf.math.argmax(self.pred_train, axis=-1)
        self.pred_cls_train = self.soft_argmax(self.pred_train)

        # Concatenation
        self.real_img = self.transform_seg(self.seg_img_train, expand_dims=False)
        self.fake_img = self.transform_seg(self.pred_cls_train, expand_dims=True)
        self.real_pair = tf.concat([self.normalize(self.img_train), self.real_img], axis=3)
        self.fake_pair = tf.concat([self.normalize(self.img_train), self.fake_img], axis=3)
        # self.fake_pair_2 = tf.concat([tf.random.shuffle(self.normalize(self.img_train)),
        #                               tf.random.shuffle(self.transform_seg(self.seg_img_train))], axis=3)
        # self.fake_pair = tf.concat([self.fake_pair_1, self.fake_pair_2], axis=0)

        # Define generator loss
        # Data loss
        self.data_loss = self.lambda_one * tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.pred_train,
            labels=self.convert_one_hot(self.seg_img_train)))

        # Adversarial loss
        self.gen_loss = self.generator_loss(self.dis_obj, self.fake_pair)

        # Additional loss function
        # Dice coefficient loss term
        self.dice_loss = tf.constant(0.)
        if self.use_dice_loss:
            self.dice_loss = self.generalized_dice_loss(labels=self.seg_img_train, logits=self.pred_train,
                                                        hyper_parameter=self.lambda_two)

        # Total loss = Data loss + Generative adversarial loss + Dice coefficient loss
        self.total_loss = self.data_loss + self.gen_loss + self.dice_loss

        # Define discriminator loss
        self.dis_loss = self.discriminator_loss(self.dis_obj, self.real_pair, self.fake_pair)

        # Optimizers
        gen_train_op = self.init_optimizer(loss=self.gen_loss, variables=self.gen_obj.variables, name='Adam_gen')
        gen_train_ops = [gen_train_op] + self.gen_ops
        self.gen_optim = tf.group(*gen_train_ops)

        dis_train_op = self.init_optimizer(loss=self.dis_loss, variables=self.dis_obj.variables, name='Adam_dis')
        dis_train_ops = [dis_train_op] + self.dis_ops
        self.dis_optim = tf.group(*dis_train_ops)

    def transform_seg(self, img, expand_dims=False):
        # # label 0~3
        img = img * 255. / (self.num_classes - 1)
        img = img / 127.5 - 1.
        # # img  = img * 2. - 1.
        # # img = 1. - 2. * tf.cast(tf.math.equal(img, tf.zeros_like(img)), dtype=tf.float32)
        # # img = self.convert_one_hot(img) * 2. - 1.
        if expand_dims:
            img = tf.expand_dims(img, axis=-1)
        return img

    @staticmethod
    def soft_argmax(x, beta=1e2):
        # Psedu-math for the below
        # y = sum( i * exp(beta * x[i])  ) / sum( exp(beta * x[i]) )
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        y = tf.math.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)
        return y

    @staticmethod
    def generator_loss(dis_obj, fake_img):
        d_logit_fake = dis_obj(fake_img)
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake ,
                                                                           labels=tf.ones_like(d_logit_fake)))
        return loss

    @staticmethod
    def discriminator_loss(dis_obj, real_img, fake_img):
        d_logit_real = dis_obj(real_img)
        d_logit_fake = dis_obj(fake_img)

        error_real = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        error_fake = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))

        loss = 0.5 * (error_real + error_fake)
        return loss

    def generalized_dice_loss(self, labels, logits, hyper_parameter=1.0):
        # This implementation refers to srcolinas's dice_loss.py
        # (https://gist.github.com/srcolinas/6df2e5e21c11227a04f826322081addf)

        smooth = 1e-17
        labels = self.convert_one_hot(labels)
        logits = tf.nn.softmax(logits)

        # weights = 1.0 / (tf.reduce_sum(labels, axis=[0, 1, 2])**2)
        weights = tf.math.divide(1.0, (tf.math.square(tf.math.reduce_sum(labels, axis=[0, 1, 2])) + smooth))

        # Numerator part
        numerator = tf.math.reduce_sum(labels * logits, axis=[0, 1, 2])
        numerator = tf.reduce_sum(weights * numerator)

        # Denominator part
        denominator = tf.math.reduce_sum(labels + logits, axis=[0, 1, 2])
        denominator = tf.math.reduce_sum(weights * denominator)

        # Dice coeeficient loss
        loss = hyper_parameter * (1.0 - 2.0 * (numerator + smooth) / (denominator + smooth))

        return loss

    def _init_eval_graph(self):
        # Initialize TFRecoder reader
        val_reader = Reader(tfrecords_file=self.data_path[1],
                            decode_img_shape=self.decode_img_shape,
                            img_shape=self.input_shape,
                            batch_size=1,
                            name='validation')

        # Batch for validation data
        img_val, seg_img_val, self.img_name_val, self.user_id_val = val_reader.batch(
            multi_test= False if self.is_train else self.multi_test)

        # tf.train.batch() returns [None, H, M, D]
        # For tf.metrics.mean_iou we need [batch_size, H, M, D]
        if self.multi_test:
            shape = [2*self.num_try, *self.output_shape]
        else:
            shape = [1, *self.output_shape]

        # Roateds img and segImg are step 1
        img_val = tf.reshape(img_val, shape=shape)
        seg_img_val = tf.reshape(seg_img_val, shape=shape)

        if self.multi_test:
            # Because of GPU memory we need to split 22 images into two groups and forward independetly
            # Split test image to two groups
            n, h, w, c = img_val.get_shape().as_list()
            img_val_1 = tf.slice(img_val, begin=[0, 0, 0, 0], size=[self.num_try, h, w, c])
            img_val_2 = tf.slice(img_val, begin=[self.num_try, 0, 0, 0], size=[self.num_try, h, w, c])

            # Network forward for validation data
            pred_val_1 = self.gen_obj(input_img=self.normalize(img_val_1),
                                         keep_rate=self.rate_tfph)
            pred_val_2 = self.gen_obj(input_img=self.normalize(img_val_2),
                                         keep_rate=self.rate_tfph)
            pred_val = tf.concat(values=[pred_val_1, pred_val_2], axis=0)
        else:
            pred_val = self.gen_obj(input_img=self.normalize(img_val),
                                       keep_rate=self.rate_tfph)

        # Since multi_test, we need inversely rotate back to the original segImg
        if self.multi_test:
            # Step 1: original rotated images
            self.img_val_s1, self.pred_val_s1, self.seg_img_val_s1 = img_val, pred_val, seg_img_val

            # Step 2: inverse-rotated images
            self.img_val_s2, self.pred_val_s2, self.seg_img_val_s2 = self.roate_independently(
                img_val, pred_val, seg_img_val)

            # Step 3: combine all results to estimate the final result
            sum_all = tf.math.reduce_sum(self.pred_val_s2, axis=0)   # [N, H, W, num_actions] -> [H, W, num_actions]
            sum_all = tf.expand_dims(sum_all, axis=0)               # [H, W, num_actions] -> [1, H, W, num_actions]
            pred_val_s3 = tf.math.argmax(sum_all, axis=-1)           # [1, H, W]

            _, h, w, c = img_val.get_shape().as_list()
            base_id = int(np.floor(self.num_try / 2.))
            self.img_val = tf.slice(img_val, begin=[base_id, 0, 0, 0], size=[1, h, w, c])
            self.seg_img_val = tf.slice(seg_img_val, begin=[base_id, 0, 0, 0], size=[1, h, w, c])
            self.pred_cls_val = pred_val_s3
        else:
            self.img_val = img_val
            self.seg_img_val = seg_img_val
            self.pred_cls_val = tf.math.argmax(pred_val, axis=-1)

        with tf.compat.v1.name_scope('Metrics'):
            # Calculate mean IoU using TensorFlow
            self.mIoU_metric, self.mIoU_metric_update = tf.compat.v1.metrics.mean_iou(
                labels=tf.squeeze(self.seg_img_val, axis=-1),
                predictions=self.pred_cls_val,
                num_classes=self.num_classes)

            # Calculate accuracy using TensorFlow
            self.accuracy_metric, self.accuracy_metric_update = tf.compat.v1.metrics.accuracy(
                labels=tf.squeeze(self.seg_img_val, axis=-1),
                predictions=self.pred_cls_val)

            # Calculate precision using TensorFlow
            self.precision_metric, self.precision_metric_update = tf.compat.v1.metrics.precision(
                labels=tf.squeeze(self.seg_img_val, axis=-1),
                predictions=self.pred_cls_val)

            # Calculate recall using TensorFlow
            self.recall_metric, self.recall_metric_update = tf.compat.v1.metrics.recall(
                labels=tf.squeeze(self.seg_img_val, axis=-1),
                predictions=self.pred_cls_val)

            # Calculate F1 score
            self.f1_score_metric = tf.math.divide(2 * self.precision_metric * self.recall_metric ,
                                            (self.precision_metric + self.recall_metric))

            # Calculate per-class accuracy
            _, self.per_class_accuracy_metric_update = tf.compat.v1.metrics.mean_per_class_accuracy(
                    labels=tf.squeeze(self.seg_img_val),
                    predictions=self.pred_cls_val,
                    num_classes=self.num_classes)

        # Isolate the variables stored behind the scens by the metric operation
        running_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope='Metrics')

        # Define initializer to initialie/reset running variables
        self.running_vars_initializer = tf.compat.v1.variables_initializer(var_list=running_vars)

    def roate_independently(self, img_val, pred_val, seg_img_val=None, is_test=False):
        imgs, preds = list(), list()
        seg_imgs, seg_img = None, None
        if not is_test:
            seg_imgs = list()

        n, h, w, c = img_val.get_shape().as_list()
        n = int(0.5 * n)

        for i in range(2):
            for idx, degree in enumerate(range(self.degree, -self.degree - 1, -2)):
                # Extract spectific tensor
                img = tf.slice(img_val, begin=[idx + i * n, 0, 0, 0], size=[1, h, w, c])                    # [1, H, W, 1]
                pred = tf.slice(pred_val, begin=[idx + i * n, 0, 0, 0], size=[1, h, w, self.num_classes])   # [1, H, W, num_classes]
                if not is_test:
                    seg_img = tf.slice(seg_img_val, begin=[idx + i * n, 0, 0, 0], size=[1, h, w, c])        # [1, H, W, 1]

                # From degree to radian
                radian = degree * math.pi / 180.

                # Roate img and segImgs
                img = tf.contrib.image.rotate(images=img, angles=radian, interpolation='BILINEAR')
                pred = tf.contrib.image.rotate(images=pred, angles=radian, interpolation='BILINEAR')
                if not is_test:
                    seg_img = tf.contrib.image.rotate(images=seg_img, angles=radian, interpolation='NEAREST')

                if i == 1:
                    # Flipping flipped images
                    img = tf.image.flip_left_right(img)
                    pred = tf.image.flip_left_right(pred)
                    if not is_test:
                        seg_img = tf.image.flip_left_right(seg_img)

                imgs.append(img)
                preds.append(pred)
                if not is_test:
                    seg_imgs.append(seg_img)

        if not is_test:
            return tf.concat(imgs, axis=0), tf.concat(preds, axis=0), tf.concat(seg_imgs, axis=0)
        else:
            return tf.concat(imgs, axis=0), tf.concat(preds, axis=0)

    def _init_test_graph(self):
        # Initialize TFRecoder reader
        test_reader = Reader(tfrecords_file=self.data_path[0],
                             decode_img_shape=self.decode_img_shape,
                             img_shape=self.input_shape,
                             batch_size=1,
                             name='test')

        # Batch for test data
        img_test, _, self.img_name_test, self.user_id_test = test_reader.batch(multi_test=self.multi_test)

        # Convert the shape [?, self.num_try, H, W, 1] to [self.num_try, H, W, 1] for multi-test
        if self.multi_test:
            shape = [2*self.num_try, *self.output_shape]
        else:
            shape = [1, *self.output_shape]
        img_test = tf.reshape(img_test, shape=shape)

        if self.multi_test:
            # Because of GPU memory we need to split 22 images into two groups and forward independetly
            # Split test image to two groups
            _, h, w, c = img_test.get_shape().as_list()
            img_test_1 = tf.slice(img_test, begin=[0, 0, 0, 0], size=[self.num_try, h, w, c])
            img_test_2 = tf.slice(img_test, begin=[self.num_try, 0, 0, 0], size=[self.num_try, h, w, c])

            # Network forward for test data
            pred_test_1 = self.gen_obj(input_img=self.normalize(img_test_1),
                                          keep_rate=self.rate_tfph)
            pred_test_2 = self.gen_obj(input_img=self.normalize(img_test_2),
                                          keep_rate=self.rate_tfph)
            pred_test = tf.concat(values=[pred_test_1, pred_test_2], axis=0)
        else:
            pred_test = self.gen_obj(input_img=self.normalize(img_test),
                                        keep_rate=self.rate_tfph)

        # Since multi_test, we need inversely rotate back to the original segImg
        if self.multi_test:
            # Step 1: original rotated images
            self.img_test_s1, self.pred_test_s1 = img_test, pred_test

            # Step 2: inverse-rotated images
            self.img_test_s2, self.pred_test_s2 = self.roate_independently(self.img_test_s1, self.pred_test_s1, is_test=True)

            # Step 3: combine all results to estimate the final result
            sum_all = tf.math.reduce_sum(self.pred_test_s2, axis=0)   # [N, H, W, num_actions] -> [H, W, num_actions]
            sum_all = tf.expand_dims(sum_all, axis=0)                # [H, W, num_actions] -> [1, H, W, num_actions]
            pred_test_s3 = tf.math.argmax(sum_all, axis=-1)           # [1, H, W]

            _, h, w, c = img_test.get_shape().as_list()
            base_id = int(np.floor(self.num_try / 2.))
            self.img_test = tf.slice(img_test, begin=[base_id, 0, 0, 0], size=[1, h, w, c])
            self.pred_cls_test = pred_test_s3
        else:
            self.img_test = img_test
            self.pred_cls_test = tf.math.argmax(pred_test, axis=-1)

    def _best_metrics_record(self):
        self.best_mIoU_tfph = tf.compat.v1.placeholder(tf.float32, name='best_mIoU')
        self.best_acc_tfph = tf.compat.v1.placeholder(tf.float32, name='best_acc')
        self.best_precision_tfph = tf.compat.v1.placeholder(tf.float32, name='best_precision')
        self.best_recall_tfph = tf.compat.v1.placeholder(tf.float32, name='best_recall')
        self.best_f1_score_tfph = tf.compat.v1.placeholder(tf.float32, name='best_f1_score')

        # Best mIoU variable
        self.best_mIoU = tf.compat.v1.get_variable(name='best_mIoU', dtype=tf.float32, initializer=tf.constant(0.),
                                                   trainable=False)
        self.assign_best_mIoU = tf.assign(self.best_mIoU, value=self.best_mIoU_tfph)

        # Best acciracy variable
        self.best_acc = tf.compat.v1.get_variable(name='best_acc', dtype=tf.float32, initializer=tf.constant(0.),
                                                  trainable=False)
        self.assign_best_acc = tf.assign(self.best_acc, value=self.best_acc_tfph)

        # Best precision variable
        self.best_precision = tf.compat.v1.get_variable(name='best_precision', dtype=tf.float32, initializer=tf.constant(0.),
                                                       trainable=False)
        self.assign_best_precision = tf.assign(self.best_precision, value=self.best_precision_tfph)

        # Best recall variable
        self.best_recall = tf.compat.v1.get_variable(name='best_recall', dtype=tf.float32, initializer=tf.constant(0.),
                                                     trainable=False)
        self.assign_best_recall = tf.assign(self.best_recall, value=self.best_recall_tfph)

        # Best f1_score variable
        self.best_f1_score = tf.compat.v1.get_variable(name='best_f1_score', dtype=tf.float32, initializer=tf.constant(0.),
                                                       trainable=False)
        self.assign_best_f1_score = tf.assign(self.best_f1_score, value=self.best_f1_score_tfph)

    def init_optimizer(self, loss, variables, name=None):
        with tf.compat.v1.variable_scope(name):
            global_step = tf.Variable(0., dtype=tf.float32, trainable=False)
            start_learning_rate = self.lr
            end_learning_rate = self.lr * 0.01
            start_decay_step = self.start_decay_step
            decay_steps = self.decay_steps

            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                     tf.compat.v1.train.polynomial_decay(start_learning_rate,
                                                               global_step - start_decay_step,
                                                               decay_steps, end_learning_rate, power=1.0),
                                     start_learning_rate))
            self.tb_lr = tf.compat.v1.summary.scalar('learning_rate', learning_rate)

            learn_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(
                loss, global_step=global_step, var_list=variables)

        return learn_step

    def _init_tensorboard(self):
        self.tb_total = tf.compat.v1.summary.scalar('Loss/total_loss', self.total_loss)
        self.tb_data = tf.compat.v1.summary.scalar('Loss/data_loss', self.data_loss)
        self.tb_adv = tf.compat.v1.summary.scalar('Loss/gen_loss', self.gen_loss)
        self.tb_dice = tf.compat.v1.summary.scalar('Loss/dice_loss', self.dice_loss)
        self.tb_dis = tf.compat.v1.summary.scalar('Loss/dis_loss', self.dis_loss)
        self.summary_op = tf.summary.merge(
            inputs=[self.tb_total, self.tb_data, self.tb_adv, self.tb_dice, self.tb_dis, self.tb_lr])

        self.tb_mIoU = tf.compat.v1.summary.scalar('Acc/mIoU', self.mIoU_metric)
        self.tb_accuracy = tf.compat.v1.summary.scalar('Acc/accuracy', self.accuracy_metric)
        self.tb_precision = tf.compat.v1.summary.scalar('Acc/precision', self.precision_metric)
        self.tb_recall = tf.compat.v1.summary.scalar('Acc/recall', self.recall_metric)
        self.tb_f1_score = tf.compat.v1.summary.scalar('Acc/f1_score', self.f1_score_metric)
        self.metric_summary_op = tf.compat.v1.summary.merge(inputs=[self.tb_mIoU, self.tb_accuracy,
                                                          self.tb_precision, self.tb_recall, self.tb_f1_score])

    @staticmethod
    def normalize(data):
        return data / 127.5 - 1.0

    def convert_one_hot(self, data):
        shape = data.get_shape().as_list()
        data = tf.dtypes.cast(data, dtype=tf.uint8)
        data = tf.one_hot(data, depth=self.num_classes, axis=-1, dtype=tf.float32, name='one_hot')
        data = tf.reshape(data, shape=[*shape[:3], self.num_classes])

        return data

class Generator(object):
    def __init__(self, name=None, conv_dims=None, padding='SAME', logger=None, resize_factor=0.5, norm='instance',
                 num_classes=4, _ops=None):
        self.name = name
        self.conv_dims = conv_dims
        self.padding = padding
        self.logger = logger
        self.resize_factor = resize_factor
        self.norm = norm
        self.num_classes = num_classes
        self._ops = _ops
        self.reuse = False

    def __call__(self, input_img, train_mode=True, keep_rate=0.5):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            # This part is for compatible between input size [640, 400] and [320, 200]
            if self.resize_factor == 1.0:
                # Stage 0
                tf_utils.print_activations(input_img, logger=self.logger)
                s0_conv1 = tf_utils.conv2d(x=input_img, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=self.padding, initializer='He', name='s0_conv1', logger=self.logger)
                s0_conv1 = tf_utils.relu(s0_conv1, name='relu_s0_conv1', logger=self.logger)

                s0_conv2 = tf_utils.conv2d(x=s0_conv1, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=self.padding, initializer='He', name='s0_conv2', logger=self.logger)
                s0_conv2 = tf_utils.norm(s0_conv2, name='s0_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=train_mode, logger=self.logger)
                s0_conv2 = tf_utils.relu(s0_conv2, name='relu_s0_conv2', logger=self.logger)

                # Stage 1
                s1_maxpool = tf_utils.max_pool(x=s0_conv2, name='s1_maxpool2d', logger=self.logger)

                s1_conv1 = tf_utils.conv2d(x=s1_maxpool, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=self.padding, initializer='He', name='s1_conv1', logger=self.logger)
                s1_conv1 = tf_utils.norm(s1_conv1, name='s1_norm0', _type=self.norm, _ops=self._ops,
                                         is_train=train_mode, logger=self.logger)
                s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1', logger=self.logger)

                s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=self.padding, initializer='He', name='s1_conv2', logger=self.logger)
                s1_conv2 = tf_utils.norm(s1_conv2, name='s1_norm1', _type='batch', _ops=self._ops,
                                         is_train=train_mode, logger=self.logger)
                s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2', logger=self.logger)
            else:
                # Stage 1
                tf_utils.print_activations(input_img, logger=self.logger)
                s1_conv1 = tf_utils.conv2d(x=input_img, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=self.padding, initializer='He', name='s1_conv1', logger=self.logger)
                s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1', logger=self.logger)

                s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=self.padding, initializer='He', name='s1_conv2', logger=self.logger)
                s1_conv2 = tf_utils.norm(s1_conv2, name='s1_norm1', _type=self.norm, _ops=self._ops,
                                         is_train=train_mode, logger=self.logger)
                s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2', logger=self.logger)

            # Stage 2
            s2_maxpool = tf_utils.max_pool(x=s1_conv2, name='s2_maxpool2d', logger=self.logger)
            s2_conv1 = tf_utils.conv2d(x=s2_maxpool, output_dim=self.conv_dims[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s2_conv1', logger=self.logger)
            s2_conv1 = tf_utils.norm(s2_conv1, name='s2_norm0', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s2_conv1 = tf_utils.relu(s2_conv1, name='relu_s2_conv1', logger=self.logger)

            s2_conv2 = tf_utils.conv2d(x=s2_conv1, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s2_conv2', logger=self.logger)
            s2_conv2 = tf_utils.norm(s2_conv2, name='s2_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s2_conv2 = tf_utils.relu(s2_conv2, name='relu_s2_conv2', logger=self.logger)

            # Stage 3
            s3_maxpool = tf_utils.max_pool(x=s2_conv2, name='s3_maxpool2d', logger=self.logger)
            s3_conv1 = tf_utils.conv2d(x=s3_maxpool, output_dim=self.conv_dims[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s3_conv1', logger=self.logger)
            s3_conv1 = tf_utils.norm(s3_conv1, name='s3_norm0', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s3_conv1 = tf_utils.relu(s3_conv1, name='relu_s3_conv1', logger=self.logger)

            s3_conv2 = tf_utils.conv2d(x=s3_conv1, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s3_conv2', logger=self.logger)
            s3_conv2 = tf_utils.norm(s3_conv2, name='s3_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s3_conv2 = tf_utils.relu(s3_conv2, name='relu_s3_conv2', logger=self.logger)

            # Stage 4
            s4_maxpool = tf_utils.max_pool(x=s3_conv2, name='s4_maxpool2d', logger=self.logger)
            s4_conv1 = tf_utils.conv2d(x=s4_maxpool, output_dim=self.conv_dims[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s4_conv1', logger=self.logger)
            s4_conv1 = tf_utils.norm(s4_conv1, name='s4_norm0', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s4_conv1 = tf_utils.relu(s4_conv1, name='relu_s4_conv1', logger=self.logger)

            s4_conv2 = tf_utils.conv2d(x=s4_conv1, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s4_conv2', logger=self.logger)
            s4_conv2 = tf_utils.norm(s4_conv2, name='s4_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s4_conv2 = tf_utils.relu(s4_conv2, name='relu_s4_conv2', logger=self.logger)
            s4_conv2_drop = tf_utils.dropout(x=s4_conv2, keep_prob=keep_rate, name='s4_dropout',
                                             logger=self.logger)

            # Stage 5
            s5_maxpool = tf_utils.max_pool(x=s4_conv2_drop, name='s5_maxpool2d', logger=self.logger)
            s5_conv1 = tf_utils.conv2d(x=s5_maxpool, output_dim=self.conv_dims[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s5_conv1', logger=self.logger)
            s5_conv1 = tf_utils.norm(s5_conv1, name='s5_norm0', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s5_conv1 = tf_utils.relu(s5_conv1, name='relu_s5_conv1', logger=self.logger)

            s5_conv2 = tf_utils.conv2d(x=s5_conv1, output_dim=self.conv_dims[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s5_conv2', logger=self.logger)
            s5_conv2 = tf_utils.norm(s5_conv2, name='s5_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s5_conv2 = tf_utils.relu(s5_conv2, name='relu_s5_conv2', logger=self.logger)
            s5_conv2_drop = tf_utils.dropout(x=s5_conv2, keep_prob=keep_rate, name='s5_dropout',
                                             logger=self.logger)

            # Stage 6
            s6_deconv1 = tf_utils.deconv2d(x=s5_conv2_drop, output_dim=self.conv_dims[10], k_h=2, k_w=2,
                                           initializer='He', name='s6_deconv1', logger=self.logger)
            s6_deconv1 = tf_utils.norm(s6_deconv1, name='s6_norm0', _type=self.norm, _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            s6_deconv1 = tf_utils.relu(s6_deconv1, name='relu_s6_deconv1', logger=self.logger)
            # Cropping
            w1 = s4_conv2_drop.get_shape().as_list()[2]
            w2 = s6_deconv1.get_shape().as_list()[2] - s4_conv2_drop.get_shape().as_list()[2]
            s6_deconv1_split, _ = tf.split(s6_deconv1, num_or_size_splits=[w1, w2], axis=2, name='axis2_split')
            tf_utils.print_activations(s6_deconv1_split, logger=self.logger)
            # Concat
            s6_concat = tf_utils.concat(values=[s6_deconv1_split, s4_conv2_drop], axis=3, name='s6_axis3_concat',
                                        logger=self.logger)

            s6_conv2 = tf_utils.conv2d(x=s6_concat, output_dim=self.conv_dims[11], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s6_conv2', logger=self.logger)
            s6_conv2 = tf_utils.norm(s6_conv2, name='s6_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s6_conv2 = tf_utils.relu(s6_conv2, name='relu_s6_conv2', logger=self.logger)

            s6_conv3 = tf_utils.conv2d(x=s6_conv2, output_dim=self.conv_dims[12], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s6_conv3', logger=self.logger)
            s6_conv3 = tf_utils.norm(s6_conv3, name='s6_norm2', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s6_conv3 = tf_utils.relu(s6_conv3, name='relu_s6_conv3', logger=self.logger)

            # Stage 7
            s7_deconv1 = tf_utils.deconv2d(x=s6_conv3, output_dim=self.conv_dims[13], k_h=2, k_w=2, initializer='He',
                                           name='s7_deconv1', logger=self.logger)
            s7_deconv1 = tf_utils.norm(s7_deconv1, name='s7_norm0', _type=self.norm, _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            s7_deconv1 = tf_utils.relu(s7_deconv1, name='relu_s7_deconv1', logger=self.logger)
            # Concat
            s7_concat = tf_utils.concat(values=[s7_deconv1, s3_conv2], axis=3, name='s7_axis3_concat',
                                        logger=self.logger)

            s7_conv2 = tf_utils.conv2d(x=s7_concat, output_dim=self.conv_dims[14], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s7_conv2', logger=self.logger)
            s7_conv2 = tf_utils.norm(s7_conv2, name='s7_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s7_conv2 = tf_utils.relu(s7_conv2, name='relu_s7_conv2', logger=self.logger)

            s7_conv3 = tf_utils.conv2d(x=s7_conv2, output_dim=self.conv_dims[15], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s7_conv3', logger=self.logger)
            s7_conv3 = tf_utils.norm(s7_conv3, name='s7_norm2', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s7_conv3 = tf_utils.relu(s7_conv3, name='relu_s7_conv3', logger=self.logger)

            # Stage 8
            s8_deconv1 = tf_utils.deconv2d(x=s7_conv3, output_dim=self.conv_dims[16], k_h=2, k_w=2, initializer='He',
                                           name='s8_deconv1', logger=self.logger)
            s8_deconv1 = tf_utils.norm(s8_deconv1, name='s8_norm0', _type=self.norm, _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            s8_deconv1 = tf_utils.relu(s8_deconv1, name='relu_s8_deconv1', logger=self.logger)
            # Concat
            s8_concat = tf_utils.concat(values=[s8_deconv1,s2_conv2], axis=3, name='s8_axis3_concat',
                                        logger=self.logger)

            s8_conv2 = tf_utils.conv2d(x=s8_concat, output_dim=self.conv_dims[17], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s8_conv2', logger=self.logger)
            s8_conv2 = tf_utils.norm(s8_conv2, name='s8_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s8_conv2 = tf_utils.relu(s8_conv2, name='relu_s8_conv2', logger=self.logger)

            s8_conv3 = tf_utils.conv2d(x=s8_conv2, output_dim=self.conv_dims[18], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s8_conv3', logger=self.logger)
            s8_conv3 = tf_utils.norm(s8_conv3, name='s8_norm2', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s8_conv3 = tf_utils.relu(s8_conv3, name='relu_conv3', logger=self.logger)

            # Stage 9
            s9_deconv1 = tf_utils.deconv2d(x=s8_conv3, output_dim=self.conv_dims[19], k_h=2, k_w=2,
                                           initializer='He', name='s9_deconv1', logger=self.logger)
            s9_deconv1 = tf_utils.norm(s9_deconv1, name='s9_norm0', _type=self.norm, _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            s9_deconv1 = tf_utils.relu(s9_deconv1, name='relu_s9_deconv1', logger=self.logger)
            # Concat
            s9_concat = tf_utils.concat(values=[s9_deconv1, s1_conv2], axis=3, name='s9_axis3_concat',
                                        logger=self.logger)

            s9_conv2 = tf_utils.conv2d(x=s9_concat, output_dim=self.conv_dims[20], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s9_conv2', logger=self.logger)
            s9_conv2 = tf_utils.norm(s9_conv2, name='s9_norm1', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s9_conv2 = tf_utils.relu(s9_conv2, name='relu_s9_conv2', logger=self.logger)

            s9_conv3 = tf_utils.conv2d(x=s9_conv2, output_dim=self.conv_dims[21], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=self.padding, initializer='He', name='s9_conv3', logger=self.logger)
            s9_conv3 = tf_utils.norm(s9_conv3, name='s9_norm2', _type=self.norm, _ops=self._ops,
                                     is_train=train_mode, logger=self.logger)
            s9_conv3 = tf_utils.relu(s9_conv3, name='relu_s9_conv3', logger=self.logger)

            if self.resize_factor == 1.0:
                s10_deconv1 = tf_utils.deconv2d(x=s9_conv3, output_dim=self.conv_dims[-1], k_h=2, k_w=2,
                                                initializer='He', name='s10_deconv1', logger=self.logger)
                s10_deconv1 = tf_utils.norm(s10_deconv1, name='s10_norm0', _type=self.norm, _ops=self._ops,
                                            is_train=train_mode, logger=self.logger)
                s10_deconv1 = tf_utils.relu(s10_deconv1, name='relu_s10_deconv1', logger=self.logger)
                # Concat
                s10_concat = tf_utils.concat(values=[s10_deconv1, s0_conv2], axis=3, name='s10_axis3_concat',
                                             logger=self.logger)

                s10_conv2 = tf_utils.conv2d(s10_concat, output_dim=self.conv_dims[-1], k_h=3, k_w=3, d_h=1, d_w=1,
                                            padding=self.padding, initializer='He', name='s10_conv2', logger=self.logger)
                s10_conv2 = tf_utils.norm(s10_conv2, name='s10_norm1', _type=self.norm, _ops=self._ops,
                                          is_train=train_mode, logger=self.logger)
                s10_conv2 = tf_utils.relu(s10_conv2, name='relu_s10_conv2', logger=self.logger)

                s10_conv3 = tf_utils.conv2d(x=s10_conv2, output_dim=self.conv_dims[-1], k_h=3, k_w=3, d_h=1, d_w=1,
                                            padding=self.padding, initializer='He', name='s10_conv3', logger=self.logger)
                s10_conv3 = tf_utils.norm(s10_conv3, name='s10_norm2', _type=self.norm, _ops=self._ops,
                                          is_train=train_mode, logger=self.logger)
                s10_conv3 = tf_utils.relu(s10_conv3, name='relu_s10_conv3', logger=self.logger)

                output = tf_utils.conv2d(s10_conv3, output_dim=self.num_classes, k_h=1, k_w=1, d_h=1, d_w=1,
                                         padding=self.padding, initializer='He', name='output', logger=self.logger)
            else:
                output = tf_utils.conv2d(s9_conv3, output_dim=self.num_classes, k_h=1, k_w=1, d_h=1, d_w=1,
                                         padding=self.padding, initializer='He', name='output', logger=self.logger)

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class Discriminator(object):
    def __init__(self, name=None, dis_c=None, norm='instance', logger=None, _ops=None):
        self.name = name
        self.conv_dims = dis_c
        self.norm = norm
        self.logger = logger
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x, logger=self.logger)

            # H1: (640, 200) -> (320, 200)
            h0_conv2d = tf_utils.conv2d(x, output_dim=self.conv_dims[0], initializer='He', logger=self.logger,
                                        name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv2d, logger=self.logger, name='h0_lrelu')

            # H2: (320, 200) -> (160, 100)
            h1_conv2d = tf_utils.conv2d(h0_lrelu, output_dim=self.conv_dims[1], initializer='He', logger=self.logger,
                                        name='h1_conv2d')
            h1_norm = tf_utils.norm(h1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h1_norm')
            h1_lrelu = tf_utils.lrelu(h1_norm, logger=self.logger, name='h1_lrelu')

            # H3: (160, 100) -> (80, 50)
            h2_conv2d = tf_utils.conv2d(h1_lrelu, output_dim=self.conv_dims[2], initializer='He', logger=self.logger,
                                        name='h2_conv2d')
            h2_norm = tf_utils.norm(h2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h2_norm')
            h2_lrelu = tf_utils.lrelu(h2_norm, logger=self.logger, name='h2_lrelu')

            # H4: (80, 50) -> (40, 25)
            h3_conv2d = tf_utils.conv2d(h2_lrelu, output_dim=self.conv_dims[3], initializer='He', logger=self.logger,
                                        name='h3_conv2d')
            h3_norm = tf_utils.norm(h3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h3_norm')
            h3_lrelu = tf_utils.lrelu(h3_norm, logger=self.logger, name='h3_lrelu')

            # H5: (40, 25) -> (20, 13)
            h4_conv2d = tf_utils.conv2d(h3_lrelu, output_dim=self.conv_dims[4], initializer='He', logger=self.logger,
                                        name='h4_conv2d')
            h4_norm = tf_utils.norm(h4_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h4_norm')
            h4_lrelu = tf_utils.lrelu(h4_norm, logger=self.logger, name='h4_lrelu')

            # H6: (20, 13) -> (20, 13)
            output = tf_utils.conv2d(h4_lrelu, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                     initializer='He', logger=self.logger, name='output_conv2d')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output