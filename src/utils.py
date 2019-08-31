# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import os
import logging


def make_folders(is_train=True, cur_time=None, subfolder=None):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))
    sample_dir = os.path.join('../sample', subfolder, '{}'.format(cur_time))
    val_dir, test_dir = None, None

    if is_train:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
    else:
        val_dir = os.path.join('../val', subfolder, '{}'.format(cur_time))
        test_dir = os.path.join('../test', subfolder, '{}'.format(cur_time))

        if not os.path.isdir(val_dir):
            os.makedirs(val_dir)

        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    return model_dir, log_dir, sample_dir, val_dir, test_dir


def init_logger(logger, log_dir, name, is_train):
    logger.propagate = False  # solve print log multiple times problem
    file_handler, stream_handler = None, None

    if is_train:
        formatter = logging.Formatter(' - %(message)s')

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

    return logger, file_handler, stream_handler


def print_main_parameters(logger, flags, is_train=False):
    if is_train:
        logger.info('gpu_index: \t\t\t{}'.format(flags.gpu_index))
        logger.info('dataset: \t\t\t{}'.format(flags.dataset))
        logger.info('method: \t\t\t{}'.format(flags.method))
        logger.info('multi_test: \t\t\t{}'.format(flags.multi_test))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t\t{}'.format(flags.resize_factor))
        logger.info('use_dice_loss: \t\t{}'.format(flags.use_dice_loss))
        logger.info('use_batch_norm: \t\t{}'.format(flags.use_batch_norm))
        logger.info('lambda_one: \t\t\t{}'.format(flags.lambda_one))
        logger.info('lambda_two: \t\t\t{}'.format(flags.lambda_two))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learing_rate: \t\t{}'.format(flags.learning_rate))
        logger.info('weight_decay: \t\t{}'.format(flags.weight_decay))
        logger.info('iters: \t\t\t{}'.format(flags.iters))
        logger.info('print_freq: \t\t\t{}'.format(flags.print_freq))
        logger.info('sample_freq: \t\t{}'.format(flags.sample_freq))
        logger.info('eval_freq: \t\t\t{}'.format(flags.eval_freq))
        logger.info('load_model: \t\t\t{}'.format(flags.load_model))
    else:
        print('-- gpu_index: \t\t{}'.format(flags.gpu_index))
        print('-- dataset: \t\t{}'.format(flags.dataset))
        print('-- method: \t\t{}'.format(flags.method))
        print('-- multi_test: \t\t\t{}'.format(flags.multi_test))
        print('-- batch_size: \t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t\t{}'.format(flags.resize_factor))
        print('-- use_dice_loss: \t\t{}'.format(flags.use_dice_loss))
        print('-- use_batch_norm: \t\t{}'.format(flags.use_batch_norm))
        print('-- lambda_one: \t\t\t{}'.format(flags.lambda_one))
        print('-- lambda_two: \t\t\t{}'.format(flags.lambda_two))
        print('-- is_train: \t\t{}'.format(flags.is_train))
        print('-- learing_rate: \t{}'.format(flags.learning_rate))
        print('-- weight_decay: \t{}'.format(flags.weight_decay))
        print('-- iters: \t\t{}'.format(flags.iters))
        print('-- print_freq: \t\t{}'.format(flags.print_freq))
        print('-- sample_freq: \t{}'.format(flags.sample_freq))
        print('-- eval_freq: \t\t{}'.format(flags.eval_freq))
        print('-- load_model: \t\t{}'.format(flags.load_model))