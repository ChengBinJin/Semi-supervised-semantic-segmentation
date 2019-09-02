# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import sys
import time
import numpy as np
import tensorflow as tf
import utils as utils


class Solver(object):
    def __init__(self, model, data, is_train=False, multi_test=False):
        self.model = model
        self.data = data
        self.is_train = is_train
        self.multi_test = False if self.is_train else multi_test
        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        feed = {
            self.model.rate_tfph: 0.5,  # rate: 1 - keep_prob
            self.model.train_mode_tfph: True
        }

        # Update discriminator and generator
        self.sess.run(self.model.dis_optim, feed_dict=feed)
        self.sess.run(self.model.gen_optim, feed_dict=feed)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, total_loss, data_loss, gen_loss, dice_loss, dis_loss, summary = self.sess.run(
            [self.model.gen_optim, self.model.total_loss, self.model.data_loss, self.model.gen_loss,
             self.model.dice_loss, self.model.dis_loss, self.model.summary_op], feed_dict=feed)

        return total_loss, data_loss, gen_loss, dice_loss, dis_loss, summary

    def eval(self, tb_writer=None, iter_time=None, save_dir=None, is_debug=False):
        if self.multi_test:
            run_ops = [self.model.mIoU_metric_update,
                       self.model.accuracy_metric_update,
                       self.model.precision_metric_update,
                       self.model.recall_metric_update,
                       self.model.per_class_accuracy_metric_update,
                       self.model.img_val,
                       self.model.pred_cls_val,
                       self.model.seg_img_val,
                       self.model.img_name_val,
                       self.model.user_id_val,
                       self.model.img_val_s1,
                       self.model.img_val_s2,
                       self.model.pred_val_s1,
                       self.model.pred_val_s2,
                       self.model.seg_img_val_s1,
                       self.model.seg_img_val_s2]
        else:
            run_ops = [self.model.mIoU_metric_update,
                       self.model.accuracy_metric_update,
                       self.model.precision_metric_update,
                       self.model.recall_metric_update,
                       self.model.per_class_accuracy_metric_update,
                       self.model.img_val,
                       self.model.pred_cls_val,
                       self.model.seg_img_val,
                       self.model.img_name_val,
                       self.model.user_id_val]

        feed = {
            self.model.rate_tfph: 0.,           # rate: 1 - keep_prob
            self.model.train_mode_tfph: False
        }

        # Initialize/reset the running variables
        self.sess.run(self.model.running_vars_initializer)

        per_cla_acc_mat = None
        for iter_time in range(self.data.num_val_imgs):
            img_s1, img_s2, pred_s1, pred_s2, seg_img_s1, seg_img_s2 = None, None, None, None, None, None

            if self.multi_test:
                _, _, _, _, per_cla_acc_mat, img, pred_cls, seg_img, img_name, user_id, \
                img_s1, img_s2, pred_s1, pred_s2, seg_img_s1, seg_img_s2 = self.sess.run(run_ops, feed_dict=feed)
            else:
                _, _, _, _, per_cla_acc_mat, img, pred_cls, seg_img, img_name, user_id = \
                    self.sess.run(run_ops, feed_dict=feed)

            if iter_time % 100 == 0:
                msg  = "\r - Evaluating progress: {:.2f}%".format((iter_time/self.data.num_val_imgs)*100.)

                # Print it.
                sys.stdout.write(msg)
                sys.stdout.flush()

            ############################################################################################################
            if not self.is_train:
                # Save images
                utils.save_imgs(img_stores=[img, pred_cls, seg_img],
                                save_dir=save_dir,
                                img_name=img_name.astype('U26'),
                                is_vertical=False)

            if not self.is_train and is_debug and self.multi_test:
                # # Step 1: save rotated images
                pred_cls_s1 = np.argmax(pred_s1, axis=-1)  # predict class using argmax function
                utils.save_imgs(img_stores=[img_s1, pred_cls_s1, seg_img_s1],
                                save_dir=os.path.join(save_dir, 'debug'),
                                name_append='step1_',
                                img_name=img_name.astype('U26'),
                                is_vertical=True)

                # Step 2: save inverse-roated images
                pred_cls_s2 = np.argmax(pred_s2, axis=-1)  # predict class using argmax function
                utils.save_imgs(img_stores=[img_s2, pred_cls_s2, seg_img_s2],
                                save_dir=os.path.join(save_dir, 'debug'),
                                name_append='step2_',
                                img_name=img_name.astype('U26'),
                                is_vertical=True)

                # Step 3: Save comparison image that includes img, single_pred, multi-test_pred, gt
                utils.save_imgs(img_stores=[img, np.expand_dims(pred_cls_s1[5], axis=0), pred_cls, seg_img],
                                save_dir=os.path.join(save_dir, 'debug'),
                                name_append='step3_',
                                img_name=img_name.astype('U26'),
                                is_vertical=False)
            ############################################################################################################

        # Calculate the mIoU
        mIoU, accuracy, precision, recall, f1_score,  metric_summary_op = self.sess.run([
            self.model.mIoU_metric,
            self.model.accuracy_metric,
            self.model.precision_metric,
            self.model.recall_metric,
            self.model.f1_score_metric,
            self.model.metric_summary_op])

        if self.is_train:
            # Write to tensorboard
            tb_writer.add_summary(metric_summary_op, iter_time)
            tb_writer.flush()

        mIoU *= 100.
        accuracy *= 100.
        precision *= 100.
        recall *= 100.
        f1_score *= 100.
        per_cla_acc_mat *= 100.

        return mIoU, accuracy, per_cla_acc_mat, precision, recall, f1_score

    def test_test(self, save_dir, is_debug=True):
        print('Number of iterations: {}'.format(self.data.num_test_imgs))

        if self.multi_test:
            run_ops = [self.model.img_test,
                       self.model.pred_test_s1,
                       self.model.pred_cls_test,
                       self.model.img_name_test,
                       self.model.user_id_test]
        else:
            run_ops = [self.model.img_test,
                       self.model.pred_cls_test,
                       self.model.img_name_test,
                       self.model.user_id_test]

        feed = {
            self.model.rate_tfph: 0.,  # rate: 1 - keep_prob
            self.model.train_mode_tfph: False
        }

        # Time check
        total_time = 0.

        pred_s1 = None
        for iter_time in range(self.data.num_test_imgs):
            tic = time.time()  # tic

            if self.multi_test:
                img, pred_s1, pred_cls, img_name, user_id = self.sess.run(run_ops, feed_dict=feed)
            else:
                img, pred_cls, img_name, user_id = self.sess.run(run_ops, feed_dict=feed)

            toc = time.time()  # toc
            total_time += toc - tic

            # Debug for multi-test
            if self.multi_test and is_debug:
                pred_cls_s1 = np.argmax(pred_s1, axis=-1)  # predict class using argmax function

                # Save images
                utils.save_imgs(img_stores=[img, np.expand_dims(pred_cls_s1[5], axis=0), pred_cls],
                                save_dir=os.path.join(save_dir, 'debug'),
                                img_name=img_name.astype('U26'),
                                is_vertical=False)

            # Save images
            utils.save_imgs(img_stores=[img, pred_cls],
                            save_dir=save_dir,
                            img_name=img_name.astype('U26'),
                            is_vertical=False)

            # Write as npy format
            utils.save_npy(data=pred_cls,
                           save_dir=save_dir,
                           file_name=img_name.astype('U26'))

            if iter_time % 100 == 0:
                print("- Evaluating progress: {:.2f}%".format((iter_time/self.data.num_test_imgs)*100.))

        msg = "Average processing time: {:.2f} msec. for one image"
        print(msg.format(total_time / self.data.num_test_imgs * 1000.))

    def sample(self, iter_time, save_dir, num_imgs=4):
        feed = {
            self.model.rate_tfph: 0.5,  # rate: 1 - keep_prob
            self.model.train_mode_tfph: True
        }

        img, pred_cls, seg_img = self.sess.run([self.model.img_train, self.model.pred_cls_train, self.model.seg_img_train],
                                             feed_dict=feed)

        # if batch_size is bigger than num_imgs, we just show num_imgs
        num_imgs = np.minimum(num_imgs, img.shape[0])

        # Save imgs
        utils.save_imgs(img_stores=[img[:num_imgs], pred_cls[:num_imgs], seg_img[:num_imgs]],
                        iter_time=iter_time,
                        save_dir=save_dir,
                        is_vertical=True)

    def set_best_mIoU(self, best_mIoU):
        self.sess.run(self.model.assign_best_mIoU, feed_dict={self.model.best_mIoU_tfph: best_mIoU})

    def get_best_mIoU(self):
        return self.sess.run(self.model.best_mIoU)

    def set_best_acc(self, best_acc):
        self.sess.run(self.model.assign_best_acc, feed_dict={self.model.best_acc_tfph: best_acc})

    def get_best_acc(self):
        return self.sess.run(self.model.best_acc)

    def set_best_precision(self, best_precision):
        self.sess.run(self.model.assign_best_precision, feed_dict={self.model.best_precision_tfph: best_precision})

    def get_best_precision(self):
        return self.sess.run(self.model.best_precision)

    def set_best_recall(self, best_recall):
        self.sess.run(self.model.assign_best_recall, feed_dict={self.model.best_recall_tfph: best_recall})

    def get_best_recall(self):
        return self.sess.run(self.model.best_recall)

    def set_best_f1_score(self, best_f1_score):
        self.sess.run(self.model.assign_best_f1_score, feed_dict={self.model.best_f1_score_tfph: best_f1_score})

    def get_best_f1_score(self):
        return self.sess.run(self.model.best_f1_score)
