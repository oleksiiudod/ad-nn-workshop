import torch.utils.data as data
from torchvision import transforms

import numpy as np
import pandas as pd
from os.path import join, isfile
import cv2


class ObjectDetectionDataset(data.Dataset):
    """ 
    Dataset wrapper class used to access data
    
    :param data_dir:
        string of the path to local input raw data folder

    :param stage:
        string of either train or val specifying which part of data to load
    """

    def __init__(self, data_dir, stage, img_shape=[1280, 128], out_shape=[160, 16]):
        super().__init__()

        self.stage = stage
        self.transform = transforms.Compose([transforms.ToTensor(),])
        self.data_list = DataExtractor(data_dir, stage, img_shape, out_shape)

    def __getitem__(self, index):
        """ Get item based on index """

        # Datapoint as numpy array
        img, anno = self.data_list[index]

        # Apply transforms
        img_tensor = self.transform(img)
        anno["centers"] = self.transform(anno["centers"])
        
        return img_tensor, anno

    def __len__(self):
        return len(self.data_list)


class DataExtractor:
    """ Used to interact with the custom folder structure and annotations in our dataset """

    def __init__(self, data_root, stage, img_shape, out_shape):

        # Root folder for data
        self.data_root = data_root

        # Indicates train/val/test
        assert stage in ["train", "val"]
        self.stage = stage

        # Record image iinput/output shapes
        self.img_w, self.img_h = img_shape
        self.out_w, self.out_h = out_shape

        # Load the list of images for relevant stage
        list_path = join(data_root, self.stage + ".list")

        # Get the individual paths to each image
        self.image_paths_list = []
        with open(list_path) as f:
            for line in f.readlines():
                filename = line.strip()
                image_path = join("images", filename + ".jpg")
                anno_path = join("labels", filename + ".txt")

                # Check that image and annotation both exist
                if not isfile(join(data_root, image_path)) and not isfile(
                    join(data_root, image_path)
                ):
                    continue

                # Append into list of paths
                self.image_paths_list.append([image_path, anno_path])

    def load_image(self, path):
        img_path = join(self.data_root, path)
        
        # Load image using OpenCV (relatively slow)
        img = cv2.imread(img_path)
        return img

    def load_annotations(self, path):
        anno_path = join(self.data_root, path)

        # Annotation row format in file: label, cx, cy, w, h, a1, a2,
        anno_array = pd.read_csv(
            anno_path, header=None, delimiter=r"\s+", dtype=float
        )._values

        # Get labels
        labels = anno_array[:, 0]
        filter = np.where(labels != [6])[0] # This gets rid of 

        # Create map
        centers = np.zeros((self.out_h, self.out_w))

        for i in filter:
            cx_out = int(anno_array[i,1] * self.out_w)
            cy_out = int(anno_array[i,2] * self.out_h)
            centers[cy_out][cx_out] = 1

        # Convert bbox to absolute coordinates
        cx_abs = anno_array[:, 1] * self.img_w
        cy_abs = anno_array[:, 2] * self.img_h
        w_abs = anno_array[:, 3] * self.img_w
        h_abs = anno_array[:, 4] * self.img_h

        # Standard bbox format uses top-left coordinates: [x1,x2,w,h]
        x1 = cx_abs - (w_abs / 2)
        y1 = cy_abs - (h_abs / 2)

        # Numpy is annoying with shapes :)
        bboxes = np.concatenate(
            [
                x1.reshape(-1, 1),
                y1.reshape(-1, 1),
                w_abs.reshape(-1, 1),
                h_abs.reshape(-1, 1),
            ],
            axis=1,
        )

        return {"bboxes": bboxes[filter], "labels": labels[filter], "centers": centers}

    def __getitem__(self, index):
        img_path, anno_path = self.image_paths_list[index]
        img = self.load_image(img_path)
        anno = self.load_annotations(anno_path)

        return img, anno

    def __len__(self):
        return len(self.image_paths_list)
