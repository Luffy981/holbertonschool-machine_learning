#!/usr/bin/env python3
"""
YOLO(You Only Look Once !!!)
"""
import numpy as np
import tensorflow.keras as K


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
