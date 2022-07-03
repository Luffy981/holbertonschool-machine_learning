#!/usr/bin/env python3
"""
YOLO(You Only Look Once !!!)
"""
from numpy.core.fromnumeric import searchsorted
import tensorflow.keras as K
import tensorflow.keras.backend as backend
from numpy import concatenate as cat
import tensorflow as tf
import numpy as np
import cv2


class Yolo:
    """
    YOLO V3
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Args:
            model_path: is the path to where a Darknet Keras model is stored
            classes_path: is the path to where the list of class names used for
                  the Darknet model, listed in order of index, can be found
            class_t: representing the box score threshold for the initial
                     filtering step
            nms_t: representing the IOU threshold for non-max suppression
            anchors:(outputs,anchor_boxes,2)containing all of the anchor boxs
                outputs: is the number of outputs (predictions)
                         made by the Darknet model
                anchor_boxes: number of anchor boxes used for each predic
                2 => [anchor_box_width, anchor_box_height]
        """
        # Loading model with Keras
        self.model = K.models.load_model(model_path)
        # Character   Meaning
        # 'r'   open for reading (default)
        # 'w'   open for writing, truncating the file first
        # 'x'   open for exclusive creation, failing if the file already exists
        # 'a'   open for writing, appending to the end of the file if it exists
        # 'b'   binary mode
        # 't'   text mode (default)
        # '+'   open a disk file for updating (reading and writing)
        # '
        # U'   universal newlines mode (deprecated)
        with open(classes_path, 'rt') as fd:
            self.class_names = fd.read().rstrip('\n').split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, number):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-number))

    def process_outputs(self, outputs, image_size):
        """
           Args:
             outputs: numpy.ndarray - contains predictions from model
               for single image.
             image_size: numpy.ndarray - images original
               size (image_height, image_width)
           Return:
              tuple - (boxes, box_confidence, box_class_probs)
              boxes: numpy.ndarray - (grid_height, grid_width, anchorboxes, 4)
                 4 => (x1, y1, x2, y2)
              box_confidence: numpy.ndarray - shape
                (grid_height, grid_width, anchor_boxes, 1)
              box_class_probs: numpy.ndarray - shape
                (grid_height, grid_width, anchor_boxes, classes)
                contains class probabilities for each output
        """
        IH, IW = image_size[0], image_size[1]
        boxes = [output[..., :4] for output in outputs]
        box_confidence, class_probs = [], []
        cornersX, cornersY = [], []

        for output in outputs:
            # Organize grid cells
            gridH, gridW, anchors = output.shape[:3]
            cx = np.arange(gridW).reshape(1, gridW)
            cx = np.repeat(cx, gridH, axis=0)
            cy = np.arange(gridW).reshape(1, gridW)
            cy = np.repeat(cy, gridH, axis=0).T

            cornersX.append(np.repeat(cx[..., np.newaxis], anchors, axis=2))
            cornersY.append(np.repeat(cy[..., np.newaxis], anchors, axis=2))
            # box confidence and class probability activations
            box_confidence.append(self.sigmoid(output[..., 4:5]))
            class_probs.append(self.sigmoid(output[..., 5:]))

        inputW = self.model.input.shape[1].value
        inputH = self.model.input.shape[2].value

        # Predicted boundary box
        for x, box in enumerate(boxes):
            bx = ((self.sigmoid(box[..., 0])+cornersX[x])/outputs[x].shape[1])
            by = ((self.sigmoid(box[..., 1])+cornersY[x])/outputs[x].shape[0])
            bw = ((np.exp(box[..., 2])*self.anchors[x, :, 0])/inputW)
            bh = ((np.exp(box[..., 3])*self.anchors[x, :, 1])/inputH)

            # x1
            box[..., 0] = (bx - (bw * 0.5))*IW
            # y1
            box[..., 1] = (by - (bh * 0.5))*IH
            # x2
            box[..., 2] = (bx + (bw * 0.5))*IW
            # y2
            box[..., 3] = (by + (bh * 0.5))*IH

        return (boxes, box_confidence, class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
           Args:
            boxes: numpy.ndarrays - (grid_height, grid_width, anchor_boxes, 4)
              containing the processed boundary boxes for
              each output, respectively
            box_confidences: a list of numpy.ndarrays of shape
              (grid_height, grid_width, anchor_boxes, 1) containing
              the processed box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape
              (grid_height, grid_width, anchor_boxes, classes) containing the
              processed box class probabilities for each output, respectively
           Return:
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing
              all of the filtered bounding boxes
            box_classes: a numpy.ndarray of shape (?,) containing the class
              number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
              for each box in filtered_boxes, respectively
        """
        best_boxes, scores, classes = None, None, None
        for x in range(len(boxes)):
            box_score = box_confidences[x] * box_class_probs[x]
            box_class = np.argmax(box_score, axis=-1)
            box_score = np.amax(box_score, axis=-1)
            mask = box_score >= self.class_t

            if best_boxes is None:
                best_boxes = boxes[x][mask]
                scores = box_score[mask]
                classes = box_class[mask]
            else:
                best_boxes = cat((best_boxes, boxes[x][mask]), axis=0)
                scores = cat((scores, box_score[mask]), axis=0)
                classes = cat((classes, box_class[mask]), axis=0)

        return (best_boxes, classes, scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
           Args:
            filtered_boxes: a numpy.ndarray of shape (?, 4)
              containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
              number for the class that filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
              for each box in filtered_boxes, respectively
           Return:
            Tuple (box_predictions, predicted_box_class, predicted_box_scores)
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of
              the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing
              the class number for box_predictions ordered by
              class and box score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing
              the box scores for box_predictions ordered by class and
              box score, respectively
        """
        # Non max suppression
        idx = tf.image.non_max_suppression(
            filtered_boxes, box_scores, box_scores.shape[0],
            iou_threshold=self.nms_t
        )
        run = backend.eval
        sup_boxes = run(tf.gather(filtered_boxes, idx))
        sup_scores = run(tf.gather(box_scores, idx))
        sup_classes = run(tf.gather(box_classes, idx))

        # Sort by class
        srt = sup_classes.argsort()
        sup_classes = sup_classes[srt]
        sup_scores = sup_scores[srt]
        sup_boxes = sup_boxes[srt, :]

        # Get indexes for sorting by score within
        # within each each group pre sorted by class
        idxs = []
        for x in range(81):
            idx_chunk = np.where(sup_classes == x)
            if idx_chunk[0].shape[0] > 0:
                idxs.append(np.amax(idx_chunk))
        prev = 0

        for x in idxs:
            # ordered slice of box scores
            slice = (-sup_scores[prev:x+1]).argsort()
            sup_scores[prev:x+1] = (sup_scores[prev:x+1])[slice]
            sup_boxes[prev:x+1, :] = (sup_boxes[prev:x+1, :])[slice]
            prev = x+1

        return (sup_boxes, sup_classes, sup_scores)

    @staticmethod
    def load_images(folder_path):
        """
           Args:
            folder_path: a string representing the path to the folder
              holding all the images to load
           Return:
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        import os
        image_paths = os.listdir(folder_path)
        images, new_paths = [], []

        for path in image_paths:
            images.append(cv2.imread(folder_path+"/"+path))
            new_paths.append(folder_path+"/"+path)

        return images, new_paths
