# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import logging
import utils as utils


class OpenEDS(object):
    def __init__(self, name='OpenEDS', track='Semantic_Segmentation_Dataset', is_train=True, resized_factor=0.5,
                 log_dir=None):
        self.name = name
        self.track = track

        self.num_train_imgs = 8916
        self.num_val_imgs = 2403
        self.num_test_imgs = 1440

        self.num_train_persons = 95
        self.num_val_persons = 28
        self.num_test_persons = 29
        self.num_classes = 4

        self.decode_img_shape = (int(640 * resized_factor), int(400 * 2 * resized_factor), 1)
        self.single_img_shape = (int(640 * resized_factor), int(400 * resized_factor), 1)

        # TFrecord path
        self.train_path = '../../Data/OpenEDS/{}/train_ori/train.tfrecords'.format(self.track)
        self.val_path = '../../Data/OpenEDS/{}/validation/validation.tfrecords'.format(self.track)
        self.test_path = '../../Data/OpenEDS/{}/test/test.tfrecords'.format(self.track)
        self.overfitting_path = '../../Data/OpenEDS/{}/overfitting/overfitting.tfrecords'.format(self.track)

        if is_train:
            self.logger = logging.getLogger(__name__)   # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, log_dir=log_dir, is_train=is_train, name='dataset')

            self.logger.info('Dataset name: \t\t{}'.format(self.name))
            self.logger.info('Dataset track: \t\t{}'.format(self.track))
            self.logger.info('Num. of training imgs: \t{}'.format(self.num_train_imgs))
            self.logger.info('Num. of validation imgs: \t{}'.format(self.num_val_imgs))
            self.logger.info('Num. of test imgs: \t\t{}'.format(self.num_test_imgs))
            self.logger.info('Num. of training persons: \t{}'.format(self.num_train_persons))
            self.logger.info('Num. of validation persons: \t{}'.format(self.num_val_persons))
            self.logger.info('Num. of test persons: \t{}'.format(self.num_test_persons))
            self.logger.info('Num. of classes: \t\t{}'.format(self.num_classes))
            self.logger.info('Decode image shape: \t\t{}'.format(self.decode_img_shape))
            self.logger.info('Single img shape: \t\t{}'.format(self.single_img_shape))
            self.logger.info('Training TFrecord path: \t{}'.format(self.train_path))
            self.logger.info('Validation TFrecord path: \t{}'.format(self.val_path))
            self.logger.info('Test TFrecord path: \t\t{}'.format(self.test_path))
            self.logger.info('Overfitting TFrecord path: \t{}'.format(self.overfitting_path))

    def __call__(self, is_train=True):
        if is_train:
            return self.train_path, self.val_path, self.overfitting_path
        else:
            return self.test_path, self.val_path, None


def Dataset(name, track='Semantic_Segmentation_Dataset', is_train=True, resized_factor=0.5, log_dir=None):
    if name == 'OpenEDS' and track == 'Semantic_Segmentation_Dataset':
        return OpenEDS(name=name, track=track, is_train=is_train, resized_factor=resized_factor, log_dir=log_dir)
    else:
        raise NotImplementedError
