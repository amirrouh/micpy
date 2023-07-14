from pathlib import Path
import cv2
import numpy as np


class IO:
    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path

    def import_training_data(self):
        images = list(map(str, self.images_path.rglob("*.tif")))
        images.sort()
        labels = list(map(str, self.labels_path.rglob("*.tif")))
        labels.sort()

        self.names = []
        self.images_arr = []
        self.labels_arr = []
        for img in images:
            name = Path(img).name
            stem = Path(img).stem

            for lbl in labels:
                if stem in lbl:
                    self.names.append(name)
                    self.images_arr.append(cv2.imread(img, 0))
                    self.labels_arr.append(cv2.imread(lbl, 0))

        self.images_arr = np.array(self.images_arr)
        self.labels_arr = np.array(self.labels_arr)

        return self.images_arr, self.labels_arr


    def convert_bindary_labels_to_grayscale(self):
        self.gray_scale_labels_arr = []
        for i, lbl in enumerate(self.labels_arr):
            lbl[lbl != 0] = 255
            self.gray_scale_labels_arr.append(lbl)
        self.gray_scale_labels_arr = np.array(self.gray_scale_labels_arr)
        return self.gray_scale_labels_arr
            print(lbl)
            cv2.imwrite(output_path /self.names[i], lbl)
