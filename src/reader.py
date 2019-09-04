# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import math
import tensorflow as tf

class Reader(object):
    def __init__(self, tfrecords_file, decode_img_shape=(320, 400, 1), img_shape=(320, 200, 1), batch_size=1,
                 min_queue_examples=200, num_threads=8, name='DataReader'):
        self.tfrecords_file = tfrecords_file
        self.decode_img_shape = decode_img_shape
        self.img_shape = img_shape

        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name

        # For data augmentations
        self.resize_factor = 1.1
        self.max_delta = 255 * 0.1  # 'max_delta' must be in the interval [0, 0.5]
        self.rotate_angle = 10.
        self._graph()

    def _graph(self):
        with tf.compat.v1.variable_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.io.parse_single_example(serialized_example, features={
                'image/file_name': tf.io.FixedLenFeature([], tf.string),
                'image/user_id': tf.io.FixedLenFeature([], tf.string),
                'image/encoded_image': tf.io.FixedLenFeature([], tf.string)})

            image_buffer = features['image/encoded_image']
            self.user_id_buffer = features['image/user_id']
            self.image_name_buffer = features['image/file_name']
            image = tf.image.decode_jpeg(image_buffer, channels=self.img_shape[2])

            # Resize to 2D
            # image = tf.image.resize(image, size=(self.decode_img_shape[0], self.decode_img_shape[1]))
            image = tf.cast(tf.image.resize(image, size=(self.decode_img_shape[0], self.decode_img_shape[1]),
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), dtype=tf.float32)

            # Split to two images
            self.img_ori, self.seg_img_ori = tf.split(image, num_or_size_splits=[self.img_shape[1], self.img_shape[1]],
                                                      axis=1)

    def shuffle_batch(self):
        img, seg_img = self.preprocess(self.img_ori, self.seg_img_ori)

        return tf.train.shuffle_batch(tensors=[img, seg_img],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)

    def batch(self, multi_test=False):
        if multi_test:
            img, seg_img = self.multi_test_process(self.img_ori, self.seg_img_ori)
        else:
            img, seg_img = self.img_ori, self.seg_img_ori

        return tf.train.batch(tensors=[img, seg_img, self.image_name_buffer, self.user_id_buffer],
                              batch_size=self.batch_size,
                              num_threads=self.num_threads,
                              capacity=self.min_queue_examples + 3 * self.batch_size,
                              allow_smaller_final_batch=True)

    def multi_test_process(self, img_ori, seg_img_ori):
        imgs, seg_imgs = list(), list()

        for (img_ori_, seg_img_ori_) in [(img_ori, seg_img_ori),
                                         (tf.image.flip_left_right(img_ori), tf.image.flip_left_right(seg_img_ori))]:
            for degree in range(-10, 11, 2):
                img, seg_img = self.fixed_rotation(img_ori_, seg_img_ori_, degree)
                imgs.append(img), seg_imgs.append(seg_img)

        return imgs, seg_imgs

    def preprocess(self, img_ori, seg_img_ori):
        # Data augmentation
        img_trans, seg_img_trans = self.random_translation(img_ori, seg_img_ori)      # Random translation
        img_flip, seg_img_flip = self.random_flip(img_trans, seg_img_trans)           # Random left-right flip
        img_brit, seg_img_brit = self.random_brightness(img_flip, seg_img_flip)       # Random brightness
        img_rotate, seg_img_rotate = self.random_rotation(img_brit, seg_img_brit)     # Random rotation

        return img_rotate, seg_img_rotate

    def random_translation(self, img, seg_img):
        # Step 1: Resized to the bigger image
        img = tf.image.resize(images=img,
                              size=(int(self.resize_factor * self.img_shape[0]),
                                    int(self.resize_factor * self.img_shape[1])),
                              method=tf.image.ResizeMethod.BICUBIC)
        seg_img = tf.image.resize(images=seg_img,
                                  size=(int(self.resize_factor * self.img_shape[0]),
                                        int(self.resize_factor * self.img_shape[1])),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Step 2: Concat two images according to the depth axis
        combined = tf.concat(values=[img, seg_img], axis=2)
        depth = combined.get_shape().as_list()[-1]

        # Step 3: Random crop
        combined = tf.image.random_crop(value=combined, size=[*self.img_shape[0:2], depth])

        # Step 4: Clip value in the range of v_min and v_max
        combined = tf.clip_by_value(t=combined, clip_value_min=0., clip_value_max=255.)

        # Step 5: Split into two images
        img, seg_img = tf.split(combined, num_or_size_splits=[self.img_shape[2], self.img_shape[2]], axis=2)

        return img, seg_img

    def random_flip(self, img, seg_img):
        # Step 1: Concat two images according to the depth axis
        combined = tf.concat(values=[img, seg_img], axis=2)

        # Step 2: Random flip
        combined = tf.image.random_flip_left_right(image=combined)

        # Step 3: Split into two images
        img, seg_img = tf.split(combined, num_or_size_splits=[self.img_shape[2], self.img_shape[2]], axis=2)

        return img, seg_img

    def random_brightness(self, img, seg_img):
        # Step 1: Random brightness
        img = tf.image.random_brightness(img, max_delta=self.max_delta)

        # Step 2: Clip value in the range of v_min and v_max
        img = tf.clip_by_value(t=img, clip_value_min=0., clip_value_max=255.)

        return img, seg_img

    def fixed_rotation(self, img, seg_img, degree):
        # Step 1: Concat two images according to the depth axis
        combined = tf.concat(values=[img, seg_img], axis=2)

        # Step 2: from degree to radian
        radian = degree * math.pi / 180.

        # Step 3: Rotate image
        combined = tf.contrib.image.rotate(images=combined, angles=radian, interpolation='NEAREST')

        # Step 4: Split into two images
        img, seg_img = tf.split(combined, num_or_size_splits=[self.img_shape[2], self.img_shape[2]], axis=2)

        return img, seg_img

    def random_rotation(self, img, seg_img):
        # Step 1: Concat two images according to the depth axis
        combined = tf.concat(values=[img, seg_img], axis=2)

        # Step 2: Select a random angle
        radian_min = -self.rotate_angle * math.pi / 180.
        radian_max = self.rotate_angle * math.pi / 180.
        random_angle = tf.random.uniform(shape=[1], minval=radian_min, maxval=radian_max)

        # Step 3: Rotate image
        combined = tf.contrib.image.rotate(images=combined, angles=random_angle, interpolation='NEAREST')

        # Step 4: Split into two images
        img, seg_img = tf.split(combined, num_or_size_splits=[self.img_shape[2], self.img_shape[2]], axis=2)

        return img, seg_img

    def test_random_translation(self, num_imgs):
        imgs, seg_imgs = list(), list()
        for i in range(num_imgs):
            img, seg_img = self.random_translation(self.img_ori, self.seg_img_ori)
            imgs.append(img), seg_imgs.append(seg_img)

        return tf.train.shuffle_batch(tensors=[imgs, seg_imgs, self.img_ori, self.seg_img_ori],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)

    def test_random_flip(self, num_imgs):
        imgs, seg_imgs = list(), list()
        for i in range(num_imgs):
            img, segImg = self.random_flip(self.img_ori, self.seg_img_ori)
            imgs.append(img), seg_imgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, seg_imgs, self.img_ori, self.seg_img_ori],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)

    def test_random_brightness(self, numImgs):
        imgs, seg_imgs = list(), list()
        for i in range(numImgs):
            img, segImg = self.random_brightness(self.img_ori, self.seg_img_ori)
            imgs.append(img), seg_imgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, seg_imgs, self.img_ori, self.seg_img_ori],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)

    def test_random_rotation(self, numImgs):
        imgs, seg_imgs = list(), list()
        for i in range(numImgs):
            img, segImg = self.random_rotation(self.img_ori, self.seg_img_ori)
            imgs.append(img), seg_imgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, seg_imgs, self.img_ori, self.seg_img_ori],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)

    def test_multi_test(self):
        imgs, seg_imgs = self.multi_test_process(self.img_ori, self.seg_img_ori)

        return tf.train.shuffle_batch(tensors=[imgs, seg_imgs],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)