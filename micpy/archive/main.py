import cv2
from pathlib import Path

from matplotlib import pyplot as plt

from config import original_images, data_path, sample_labels
from core import grid_layover

# grid_layover(original_images[150], data_path)

for img in original_images:
    if "6754_46579_RSFA_50x0052" in Path(img).name:
        print(img)


for lbl in sample_labels:
    if "6754_46579_RSFA_50x0052" in Path(lbl).name:
        print(lbl)


