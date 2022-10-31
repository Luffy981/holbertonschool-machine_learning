#!/usr/bin/env python3
"""
   Module contains:
   Class Yolo
"""


from numpy.core.defchararray import index
from numpy.core.fromnumeric import searchsorted
import tensorflow.keras as K
import tensorflow.keras.backend as backend
from numpy import concatenate as cat
import tensorflow as tf
import numpy as np
import cv2


class Yolo():
    """
       Yolo v3 class for performing object detection.

       Public Instance Attributes
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
           Init method for instanciating Yolo class.

           Args:
             model_path: path to Darknet keras model
             classes_path: path to list of class names for
               darknet model
             class_t: float representing box score for initial
               filtering step
             nms_t: float representing IOU threshold for non-max
               supression
             anchors: numpy.ndarray - shape (outputs, anchor_boxes, 2)
               containing all anchor boxes
                 outputs: number of predictions made
                 anchor_boxes: number of anchor boxes for each pred.
                 2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'rt') as fd:
            self.class_names = fd.read().rstrip('\n').split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, arr):
        """sigmoid activation function"""
        return 1 / (1+np.exp(-1*arr))

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

            cornersX.append(
                np.repeat(cx[..., np.newaxis], anchors, axis=2)
                )
            cornersY.append(
                np.repeat(cy[..., np.newaxis], anchors, axis=2)
                )
            # box confidence and class probability activations
            box_confidence.append(self.sigmoid(output[..., 4:5]))
            class_probs.append(self.sigmoid(output[..., 5:]))

        inputW = self.model.input.shape[1].value
        inputH = self.model.input.shape[2].value

        # Predicted boundary box
        for x, box in enumerate(boxes):
            bx = (
                (self.sigmoid(box[..., 0])+cornersX[x])/outputs[x].shape[1]
                )
            by = (
                (self.sigmoid(box[..., 1])+cornersY[x])/outputs[x].shape[0]
                )
            bw = (
                (np.exp(box[..., 2])*self.anchors[x, :, 0])/inputW
                )
            bh = (
                (np.exp(box[..., 3])*self.anchors[x, :, 1])/inputH
                )

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

    def preprocess_images(self, images):
        """
           Args:
            images: a list of images as numpy.ndarrays

           Return:
            images: a numpy.ndarray of shape (ni, input_h, input_w, 3)
              containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model
                  Note - this can vary by model
                input_w: the input width for the Darknet model
                  Note - this can vary by model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2) containing
              the original height and width of the images
                2 => (image_height, image_width)
        """
        inputW = self.model.input.shape[1].value
        inputH = self.model.input.shape[2].value
        pimages, image_sizes = None, None
        channels = None
        dims = (inputH, inputW)
        Q = cv2.INTER_CUBIC

        for img in images:
            for ch in range(3):
                # resize each channel individually
                if channels is None:
                    channels = cv2.resize(
                        img[:, :, ch], dims, interpolation=Q
                    )
                    channels = channels[..., np.newaxis]
                else:
                    resized = cv2.resize(
                        img[:, :, ch], dims, interpolation=Q
                    )
                    resized = resized[..., np.newaxis]
                    channels = np.concatenate((channels, resized), axis=2)

            # Stack resized images
            if pimages is None:
                pimages = (channels.copy())[np.newaxis, ...]
                image_sizes = np.array([[img.shape[0], img.shape[1]]])
            else:
                add_resized = (channels.copy())[np.newaxis, ...]
                pimages = np.concatenate((pimages, add_resized), axis=0)
                image_sizes = np.concatenate((
                    image_sizes,
                    np.array([[img.shape[0], img.shape[1]]])
                    ), axis=0)
            channels = None

        pimages = cv2.normalize(
            pimages, None, alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        return (pimages, image_sizes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
           Displays the image with all boundary boxes,
             class names, and box scores.

           Args:
            image: a numpy.ndarray containing an unprocessed image
            boxes: a numpy.ndarray containing the boundary
              boxes for the image
            box_classes: a numpy.ndarray containing the class indices
              for each box
            box_scores: a numpy.ndarray containing the box scores
              for each box
            file_name: the file path where the original image is stored

           Return:
            None
        """
        BLUE = (255, 0, 0)
        RED = (0, 0, 200)

        for i, box in enumerate(boxes):
            x, y = round(box[0]), round(box[1])
            xWidth, yHeight = round(box[2]), round(box[3])

            idd = box_classes[i]
            score = " {:0.2f}".format(box_scores[i])
            label = self.class_names[idd]+score
            cv2.rectangle(
                image, (int(x), int(y)),
                (int(xWidth), int(yHeight)), BLUE, 2
            )
            cv2.putText(
                image, label, (int(x-5), int(y-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, thickness=1,
                lineType=cv2.LINE_AA
                )

        # cv2.imshow(file_name, image)
        wait = input("Press the s button to save image: ")
        if wait == "s":
            import os
            if os.path.isdir("./detections") is False:
                os.mkdir("./detections")
            cv2.imwrite("./detections/{}".format(file_name), image)

        cv2.destroyAllWindows()
