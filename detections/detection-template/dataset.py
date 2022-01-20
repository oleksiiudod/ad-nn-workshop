import torch
import torch.utils.data as data

import numpy as np
import os
import os.path.join as join
import os.path.isfile as isfile


class Dataset(data.Dataset):
    """ Dataset wrapper class """

    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data = np.loadtxt(data_dir)

    def __getitem__(self, index):
        """ Get item based on index """
        data_point = self.data[index]
        # Might want to include some transforms here
        return data_point

    def __len__(self):
        return self.total_size




class DetectionsToList:

    def __init__(self, data_root, stage):
        
        # Root folder for data
        self.data_root = data_root
        
        # Indicates train/val/test
        assert stage in ["train", "val", "test"]
        self.stage = stage

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
                if not isfile(join(data_root, image_path)) and not isfile(join(data_root, image_path)):
                    continue

                self.image_paths_list.append([image_path, anno_path])


    def load_annotations(self):
        return 
    