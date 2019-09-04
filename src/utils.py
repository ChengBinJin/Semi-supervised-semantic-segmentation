# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import logging
import cv2
import numpy as np


def make_folders(is_train=True, cur_time=None, subfolder=''):
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
        logger.info('lambda_one: \t\t\t{}'.format(flags.lambda_one))
        logger.info('lambda_two: \t\t\t{}'.format(flags.lambda_two))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learing_rate: \t\t{}'.format(flags.learning_rate))
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
        print('-- lambda_one: \t\t\t{}'.format(flags.lambda_one))
        print('-- lambda_two: \t\t\t{}'.format(flags.lambda_two))
        print('-- is_train: \t\t{}'.format(flags.is_train))
        print('-- learing_rate: \t{}'.format(flags.learning_rate))
        print('-- iters: \t\t{}'.format(flags.iters))
        print('-- print_freq: \t\t{}'.format(flags.print_freq))
        print('-- sample_freq: \t{}'.format(flags.sample_freq))
        print('-- eval_freq: \t\t{}'.format(flags.eval_freq))
        print('-- load_model: \t\t{}'.format(flags.load_model))


def save_imgs(img_stores, iter_time=None, save_dir=None, margin=5, img_name=None, name_append='', is_vertical=True):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    num_categories = len(img_stores)
    for i in range(num_categories):
        if img_stores[i].shape[-1] == 1:
            img_stores[i] = np.squeeze(img_stores[i], axis=-1)

    num_imgs, h, w = img_stores[0].shape

    if is_vertical:
        canvas = np.zeros((num_categories * h + (num_categories + 1) * margin,
                           num_imgs * w + (num_imgs + 1) * margin, 3), dtype=np.uint8)

        for i in range(num_imgs):
            for j in range(num_categories):
                if j == 0:      # gray-scale image
                    canvas[(j + 1) * margin + j * h:(j + 1) * margin + (j + 1) * h,
                    (i + 1) * margin + i * w:(i + 1) * (margin + w), :] = \
                        np.dstack((img_stores[j][i], img_stores[j][i], img_stores[j][i]))
                else:           # label maps
                    canvas[(j+1)*margin+j*h:(j+1)*margin+(j+1)*h, (i+1)*margin+i*w:(i+1)*(margin+w), :] = \
                        convert_color_label(img_stores[j][i])
    else:
        canvas = np.zeros((num_imgs * h + (num_imgs + 1) * margin,
                           num_categories * w + (num_categories + 1) * margin, 3), dtype=np.uint8)

        for i in range(num_imgs):
            for j in range(num_categories):
                if j == 0:              # gray-scale image
                    canvas[(i+1)*margin+i*h:(i+1)*(margin+h), (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = \
                        np.dstack((img_stores[j][i], img_stores[j][i], img_stores[j][i]))
                elif j == 1:            # soft-argmax maps
                    canvas[(i + 1) * margin + i * h:(i + 1) * (margin + h),
                    (j + 1) * margin + j * w:(j + 1) * margin + (j + 1) * w, :] = \
                        convert_color_label(np.round(img_stores[j][i]))
                elif j == 2:            # label maps
                    canvas[(i+1)*margin+i*h:(i+1)*(margin+h), (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = \
                        convert_color_label(img_stores[j][i])
                else:
                    img = img_stores[j][i]
                    img = inverse_transform_seg(img, n_classes=4)
                    canvas[(i+1)*margin+i*h:(i+1)*(margin+h), (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = \
                        convert_color_label(img)


    if img_name is None:
        cv2.imwrite(os.path.join(save_dir, str(iter_time).zfill(6) + '.png'), canvas)
    else:
        cv2.imwrite(os.path.join(save_dir, name_append+img_name[0]), canvas)

def inverse_transform_seg(img, n_classes):
    img = (img + 1.) * 127.5 * ((n_classes - 1) / 255.)
    img = np.round(img).astype(np.uint8)
    return img


def convert_color_label(img):
    yellow = [102, 255, 255]
    green = [102, 204, 0]
    cyan = [153, 153, 0]
    violet = [102, 0, 102]

    # 0: background - violet
    # 1: sclera - cyan
    # 2: iris - green
    # 3: pupil - yellow
    img_rgb = np.zeros([*img.shape, 3], dtype=np.uint8)
    for i, color in enumerate([violet, cyan, green, yellow]):
        img_rgb[img == i] = color

    return img_rgb


def save_npy(data, save_dir, file_name, size=(640, 400)):
    save_dir = os.path.join(save_dir, 'npy')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Extract image number from [000002342342_U23.png]
    file_name = file_name[0].split('_')[0]

    # Convert [1, H, W] to [H, W]
    data = np.squeeze(data)

    # Resize from [H/2, W/2] to [H, W]
    if data.shape[0:2] != size:
        data = cv2.resize(data, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    # Convert data type from int32 to uint8
    data = data.astype(np.uint8)

    # Save data in npy format by requirement
    np.save(os.path.join(save_dir, file_name), data)