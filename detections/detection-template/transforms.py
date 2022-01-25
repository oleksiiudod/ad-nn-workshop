"""
A set of consistent transformation classes for data pre-processing and augmentation. 
"""

### Imports
import numpy as np
import torch
import cv2
import copy


class AffineTransform(object):
    """
    Applies an affine transformation on an image

    :param output_size:
        tuple specifying the target size of the image
    """

    def __init__(self, zoom=0.3):
        self.zoom = zoom

    def __call__(self, sample):
        """
        :param sample:
            :img:
                tuple specifying the target size of image
            :annotations:
                dictionary containing annotations for given data type
            :data_type:
                string indicating whether the input data is detection or segmentation
        """
        # Reproducability
        # np.random.seed(0)

        # Unpack sample
        img, annotations, data_type = copy.deepcopy(sample)
        assert data_type in ["detection", "segmentation"]

        # Establish Scales (random)
        h, w, c = img.shape
        scale_x = np.random.uniform() * self.zoom + 1.0  # hard-coded
        scale_y = np.random.uniform() * self.zoom + 1.0
        max_offx = (scale_x - 1.0) * w
        max_offy = (scale_y - 1.0) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)
        flip = np.random.uniform() > 0.5

        scales = (scale_x, scale_y)
        offsets = (offx, offy)

        ### Transform image
        img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
        img = img[offy : (offy + h), offx : (offx + w)]
        if flip:
            img = cv2.flip(img, 1)

        ### Detection
        if data_type == "detection":
            # Convert to Float array
            annotations["gt_boxes"] = np.asarray(
                annotations["gt_boxes"], dtype=np.float
            )

            # Apply same transform on bounding boxes and angles
            annotations["gt_boxes"] = self._offset_boxes(
                annotations["gt_boxes"], img.shape, scales, offsets, flip
            )
            annotations["gt_angles"] = self._flip_angles(annotations["gt_angles"], flip)

            # Revert to Int array
            annotations["gt_boxes"] = np.asarray(annotations["gt_boxes"], dtype=np.int)

        ### Segmentation
        else:
            # Apply same transform on segmentation annotations
            seg = annotations["gt_segmentation"]

            # Accommodate small image
            ratio_w = img.shape[1] / seg.shape[1]
            ratio_h = img.shape[0] / seg.shape[0]

            # Scale to segmentation img size  # Not throughly test
            offy = int(offy / ratio_w)
            offx = int(offx / ratio_h)
            w = int(w / ratio_w)
            h = int(h / ratio_h)

            seg = cv2.resize(seg.astype("float32"), (0, 0), fx=scale_x, fy=scale_y)
            seg = np.asarray(seg, dtype=np.int)

            # except:
            #     print("offy", offy)
            #     print("offx", offx)
            #     print("w", w)
            #     print("h", h)
            #     print("seg.shape", seg.shape)
            #     print(seg)

            # finally:
            #     seg = cv2.resize(seg, (0, 0), fx=scale_x, fy=scale_y)

            seg = seg[offy : (offy + h), offx : (offx + w)]
            if flip:
                seg = cv2.flip(seg, 1)

            annotations["gt_segmentation"] = seg

        return [img, annotations, data_type]

    # ----- Helper Internal Methods -----#
    def _offset_boxes(self, boxes, im_shape, scales, offs, flip):
        """Applies the offset to the bounding boxes"""
        if len(boxes) == 0:
            return boxes
        boxes[:, 0::2] *= scales[0]
        boxes[:, 1::2] *= scales[1]
        boxes[:, 0::2] -= offs[0]
        boxes[:, 1::2] -= offs[1]
        boxes = self._clip_boxes(boxes, im_shape)
        if flip:
            boxes_x = np.copy(boxes[:, 0])
            boxes[:, 0] = im_shape[1] - boxes[:, 2]
            boxes[:, 2] = im_shape[1] - boxes_x
        return boxes

    def _flip_angles(self, gt_angles, flip):
        """Flips the angles"""
        flipped_angles = gt_angles.copy()
        if flip:
            # angles are [sin_v, cos_v] -> [y, x] (reversed order of coordinates)
            flipped_angles[:, 1] = -flipped_angles[:, 1]  # negate x value
        return flipped_angles

    def _clip_boxes(self, boxes, im_shape):
        """Clip boxes to image boundaries."""
        if boxes.shape[0] == 0:
            return boxes
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    # --------------- /end ---------------#


class Resize(object):
    """
    Rescale the image and annotations in a sample to a given size.

    :param output_size:
        tuple specifying the target size of the image
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (np.ndarray))
        self.w, self.h = output_size

    def __call__(self, sample):
        """
        :param sample:
            :img:
                tuple specifying the target size of image
            :annotations:
                dictionary containing annotations for given data type
            :data_type:
                string indicating whether the input data is detection or segmentation
        """
        # Reproducability
        # np.random.seed(0)

        # Unpack sample
        img, annotations, data_type = copy.deepcopy(sample)
        assert data_type in ["detection", "segmentation"]

        ### Check if resize needed
        if img.shape[1] != self.w or img.shape[0] != self.h:

            ### Detection
            if data_type == "detection":
                # Convert to Float array
                annotations["gt_boxes"] = np.asarray(
                    annotations["gt_boxes"], dtype=np.float
                )

                # Resize boxes
                annotations["gt_boxes"][:, 0::2] *= float(self.w) / img.shape[1]
                annotations["gt_boxes"][:, 1::2] *= float(self.h) / img.shape[0]

                # Revert to Int array
                annotations["gt_boxes"] = np.asarray(
                    annotations["gt_boxes"], dtype=np.int64
                )

            ## Segmentation
            else:
                # print(annotations["gt_segmentation"].shape[0:2])
                assert annotations["gt_segmentation"].shape[0:2] == img.shape[0:2]
                # Resize segmentation
                annotations["gt_segmentation"] = cv2.resize(
                    annotations["gt_segmentation"],
                    (self.w, self.h),
                    interpolation=cv2.INTER_CUBIC,
                )

            ### Resize image
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        return [img, annotations, data_type]


class Recolor(object):
    """
    Apply random recoloring of the image for data augmentation

    :param brightness_scale
        float describing the extend of the colour perturbation
    """

    def __init__(self, brightness_scale=0.1):
        self.brightness_scale = brightness_scale

    def __call__(self, sample):
        """
        :param sample:
            :img:
                tuple specifying the target size of image
            :annotations:
                dictionary containing annotations for given data type
            :data_type:
                string indicating whether the input data is detection or segmentation
        """
        # Reproducability
        # np.random.seed(0)

        # Unpack sample
        img, annotations, data_type = copy.deepcopy(sample)
        assert data_type in ["detection", "segmentation"]

        ### Prepare colors
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (1.0 / 255) * img.astype(np.float)

        ### Parameters
        t = np.random.uniform(-1, 1, 3)
        up = np.random.uniform(-1, 1)

        ### Apply change
        img *= 1 + t * self.brightness_scale
        mx = 1 + self.brightness_scale
        img = img / mx

        return [img, annotations, data_type]


class ToOutput(object):
    """
    Converts to Tensor output. Should be final step in transformation pipeline.
    :param output_size:
        tuple specifying the target size of the output
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (np.ndarray))
        self.output_size = output_size

    def __call__(self, sample):
        """
        :param sample:
            :img:
                tuple specifying the target size of image
            :annotations:
                dictionary containing annotations for given data type
            :data_type:
                string indicating whether the input data is detection or segmentation
        """

        # Reproducability
        # np.random.seed(0)

        # Unpack sample
        img, annotations, data_type = copy.deepcopy(sample)
        assert data_type in ["detection", "segmentation"]

        ### Prepare colors  # TO FIX (a more logical place for this)
        if np.max(img) > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (1.0 / 255) * img.astype(np.float)

        # Planar format
        im_planar = img.transpose(2, 0, 1)

        ### Into tensor format
        img = torch.from_numpy(im_planar).type("torch.FloatTensor")

        ### Format Segmentation
        if data_type == "segmentation":
            # print(annotations["gt_segmentation"].shape)

            if np.all(annotations["gt_segmentation"].shape != self.output_size):
                annotations["gt_segmentation"] = self._mask_resize(
                    annotations["gt_segmentation"], self.output_size
                )

        else:
            annotations["gt_boxes"] = np.asarray(
                annotations["gt_boxes"], dtype=np.int64
            )

        return [img, annotations, data_type]

    def _mask_resize(self, img, res_wh):
        """Resizes the segmentation mask to desired size"""
        res_wh = (*res_wh,)
        max_label = np.max(img)
        planes = [
            cv2.resize(
                (255 * (img == v)).astype(np.uint8),
                res_wh,
                interpolation=cv2.INTER_AREA,
            )
            for v in range(max_label + 1)
        ]
        stacked_planes = np.stack(planes)
        argmax = np.argmax(stacked_planes, axis=0)
        argmax = argmax.astype(np.uint8)
        return argmax


class PreprocessTransform(object):
    """
    Rescales the image and annotations during preprocessing.

    :param input put_size:
        tuple specifying the size of the image input to NN

    :param output_size:
        tuple specifying the size of the image output of NN
    """

    def __init__(self, input_size, output_size):
        assert isinstance(output_size, (np.ndarray))
        assert isinstance(input_size, (np.ndarray))
        self.w, self.h = input_size
        self.output_size = output_size

    def __call__(self, sample):
        """
        :param sample:
            :img:
                tuple specifying the target size of image
            :annotations:
                dictionary containing annotations for given data type
            :data_type:
                string indicating whether the input data is detection or segmentation
        """
        # Reproducability
        np.random.seed(0)

        # Unpack sample
        img, annotations, data_type = copy.deepcopy(sample)
        assert data_type in ["detection", "segmentation"]

        ### Check if resize needed
        if img.shape[1] != self.w or img.shape[0] != self.h:

            ### Detection
            if data_type == "detection":
                pass

            ## Segmentation
            else:
                # print(annotations["gt_segmentation"].shape[0:2])
                assert annotations["gt_segmentation"].shape[0:2] == img.shape[0:2]

                # Resize segmenation to input size
                annotations["gt_segmentation"] = np.asarray(
                    annotations["gt_segmentation"], dtype=np.int
                )  # Ensure int
                annotations["gt_segmentation"] = self._mask_resize(
                    annotations["gt_segmentation"], (self.w, self.h)
                )

                # # Resize segmentation to input (TO REMOVE)
                # annotations["gt_segmentation"] = cv2.resize(
                #     annotations["gt_segmentation"],
                #     (self.w, self.h),
                #     interpolation=cv2.INTER_CUBIC,
                # )

                # # Resize segmenation to output
                # annotations["gt_segmentation"] = np.asarray(
                #     annotations["gt_segmentation"], dtype=np.int
                # )  # Ensure int
                # annotations["gt_segmentation"] = self._mask_resize(
                #     annotations["gt_segmentation"], self.output_size
                # )

            ### Resize image
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

        return [img, annotations, data_type]

    def _mask_resize(self, img, res_wh):
        """Resizes the segmentation mask to desired size"""
        res_wh = (*res_wh,)
        max_label = np.max(img)
        planes = [
            cv2.resize(
                (255 * (img == v)).astype(np.uint8),
                res_wh,
                interpolation=cv2.INTER_AREA,
            )
            for v in range(max_label + 1)
        ]
        stacked_planes = np.stack(planes)
        argmax = np.argmax(stacked_planes, axis=0)
        argmax = argmax.astype(np.uint8)
        return argmax


class TransformTemplate(object):
    """
    Template for creating new transforms in the pipeline

    :param output_size:
        tuple specifying the target size of the output
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        """
        :param sample:
            :img:
                tuple specifying the target size of image
            :annotations:
                dictionary containing annotations for given data type
            :data_type:
                string indicating whether the input data is detection or segmentation
        """

        # Reproducability
        # np.random.seed(0)

        # Unpack sample
        img, annotations, data_type = copy.deepcopy(sample)
        assert data_type in ["detection", "segmentation"]

        ### DO SOMETHING ###

        return [img, annotations, data_type]
