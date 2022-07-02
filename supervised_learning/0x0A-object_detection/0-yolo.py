#!/usr/bin/env python3
"""
YOLO(You Only Look Once !!!)
"""
import numpy as n
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
