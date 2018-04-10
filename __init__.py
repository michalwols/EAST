import matplotlib

matplotlib.use('TkAgg')

import os
import cv2
import numpy as np
import collections
import tensorflow as tf
from .eval import resize_image, sort_poly, detect
from . import model


class EAST:
  def __init__(self, checkpoint=os.path.dirname(os.path.abspath(__file__)) +
                                '/east_icdar2015_resnet_v1_50_rbox'):
    self.checkpoint = checkpoint

    if checkpoint:
      self.load_model()

  def load_model(self):
    try:
      ckpt_state = tf.train.get_checkpoint_state(self.checkpoint)
      model_path = os.path.join(self.checkpoint,
                                os.path.basename(
                                  ckpt_state.model_checkpoint_path))
    except AttributeError as e:
      import warnings
      warnings.warn("Couldn't find model checkpoint, needs to be downloaded "
                    "first")
      raise

    self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                       name='input_images')
    self.global_step = tf.get_variable('global_step', [],
                                       initializer=tf.constant_initializer(0),
                                       trainable=False)

    self.f_score, self.f_geometry = model.model(self.input_images,
                                                is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997,
                                                          self.global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    saver.restore(self.sess, model_path)

  def predict(self, img, min_score=.6, min_box_score=.1, nms_threshold=.2,
              min_box_size=3):
    timer = collections.OrderedDict([
      ('net', 0),
      ('restore', 0),
      ('nms', 0)
    ])

    if isinstance(img, str):
      img = cv2.imread(img, 1)

    im_resized, (ratio_h, ratio_w) = resize_image(img)

    score, geometry = self.sess.run(
      [self.f_score, self.f_geometry],
      feed_dict={self.input_images: [im_resized[:, :, ::-1]]})

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer,
                          score_map_thresh=min_score,
                          box_thresh=min_box_score, nms_thres=nms_threshold)

    print(timer)

    scores = None
    if boxes is not None:
      scores = boxes[:, 8].reshape(-1)
      boxes = boxes[:, :8].reshape((-1, 4, 2))
      boxes[:, :, 0] /= ratio_w
      boxes[:, :, 1] /= ratio_h

    text_lines = []
    if boxes is not None:
      text_lines = []
      for box, score in zip(boxes, scores):
        box = sort_poly(box.astype(np.int32))
        if np.linalg.norm(box[0] - box[1]) < min_box_size or np.linalg.norm(
            box[3] - box[0]) < min_box_size:
          continue
        tl = collections.OrderedDict(zip(
          ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
          map(float, box.flatten())))
        tl['score'] = float(score)
        text_lines.append(tl)
    return text_lines, boxes, scores

  def draw(self, img, text_lines=None):
    if not text_lines:
      text_lines = self.predict(img)

    for t in text_lines:
      d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                    t['y2'], t['x3'], t['y3']], dtype='int32')
      d = d.reshape(-1, 2)
      cv2.polylines(img, [d], isClosed=True, color=(255, 255, 0))
    return img
