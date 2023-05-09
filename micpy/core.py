from pathlib import Path
from os.path import join

from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage import morphology
import cv2
from PIL import Image
from tqdm import tqdm


def read(img_path, gray_scale=True):
    img = cv2.imread(img_path, 0)
    return img
    

def left_right_hist_equalization(img, kernel=3, min_threshold=10, max_threshold=255):
    """ left to right histogram equalization for a single image based on average intensity in the n left column and right columns"""
    img_adjusted = img.copy()
    left_arr = img[:,:kernel]
    right_arr = img[:, -kernel:]
    left_avg = np.average(left_arr)
    right_avg = np.average(right_arr)
    avg = (left_avg  + right_avg / 2)
    for c in range(img.shape[1]):
        col = img[:, c]
        col_filtered = col[(col >= min_threshold) & (col <= max_threshold)]
        col_avg = np.average(col_filtered)
        if  col_avg < avg:
            col = col + np.abs(col_avg - avg)
        else:
            col = col - np.abs(col_avg - avg)
        img_adjusted[:, c] = col
    return img_adjusted


def set_average_intensity_single(img, avg_intensity, min_threshold=10, max_threshold=255):
    """ left to right histogram equalization for a single image based on average intensity in the n left column and right columns"""
    img_adjusted = img.copy()

    for c in range(img.shape[1]):
        col = img[:, c]
        col_filtered = col[(col >= min_threshold) & (col <= max_threshold)]
        col_avg = np.average(col_filtered)
        if  col_avg < avg_intensity:
            col = col + np.abs(col_avg - avg_intensity)
        else:
            col = col - np.abs(col_avg - avg_intensity)
        img_adjusted[:, c] = col
    return img_adjusted


def set_average_intensity_multiple(images, outout_dir, min_threshold=0, max_threshold=255):
    """ changes the average intensity of multiple images to the average intensity of the images

    Parameters
    ----------
    images : list
        list of input image paths as list of strings
    outout_dir : str
        output directory
    min_threshold : int, optional
        minimum threshold to be included in calculation of the average inrensity, by default 10
    max_threshold : int, optional
        maximum threshold to be included in calculation of the average inrensity, by default 255
    """
    avg_intensity_all = []
    print("Calculating the average intensoty of all the images")
    for i in tqdm(images):
        img = read(i)
        avg_intensity_all.append(np.average(img))
    average_intensity = np.average(avg_intensity_all)
    print("Adjusting all the intensities to the average intensity")
    for i in tqdm(images):
        img_org = read(i)
        img_new = set_average_intensity_single(img_org, average_intensity)
        name = Path(i).name
        cv2.imwrite(join(outout_dir, name), img_new)


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def hist_equalization(img):
    img_eq = exposure.equalize_hist(img)
    img_eq = convert(img_eq, 0, 255, np.uint8)
    return img_eq


def clahe(img):
    clashe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clashe.apply(img)
    return img_clahe

    def detect_contour_all(self):
        """
        Calculates the contours based on the thresholded image.
        Return:
            mask (numpy array): Background image all 255.
            contours (list): List of 2D np arrays showing contour points
        """
        _, threshold = self.threshold()
        try:
            _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(threshold.shape, np.uint8)
        cv2.drawContours(mask, contours, -1, (255, 0, 0), 3)
        return mask, contours
    
    
def detect_contour_filtered(image_arr, eccentricity_min, eccentricity_max, area_min, area_max):
    """
    Calculates the contours based on the thresholded image.
    Return:
        mask (numpy array): Background image all 255.
        contours (list): List of 2D np arrays showing contour points
    """
    try:
        _, contours, _ = cv2.findContours(image_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(image_arr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for c in contours:
        if len(c) >= 5:
            ellipse = cv2.fitEllipseAMS(c)

            (x, y), (a, b), angle = ellipse

            eccentricity = np.sqrt(abs(a ** 2 - b ** 2)) / max(a, b)
            area = np.pi * a * b

            if (eccentricity <= eccentricity_max) and (eccentricity >= eccentricity_min):
                if (area <= area_max) and (area >= area_min):
                    filtered_contours.append(c)

    mask = np.zeros(image_arr.shape, np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, (255, 0, 0), 3)
    return mask, filtered_contours


def remove_noise(img, min_size=3, connectivity=2):
    binarized = np.where(img>0.1, 1, 0)
    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=min_size, connectivity=connectivity).astype(int)
    # black out pixels
    mask_x, mask_y = np.where(processed == 0)
    img[mask_x, mask_y] = 0
    return img


def calc_tot_avg_intensity(sem_dir):
    """ calculates the avergae intensoty of all the sem images """


def threshold(img, min_threshold, max_threshold):
    """
    Calculates thresholding results.

    Return:
        img (numpy array): Original image
        threshold (numpy array): Thresholded image
    """
    img_th = cv2.imread(img, 0)
    img_th = cv2.medianBlur(img_th, 5)
    _, threshold = cv2.threshold(img_th, min_threshold, max_threshold, cv2.THRESH_BINARY)
    return threshold


def detect_contour(thresholded_arr, eccentricity_min, eccentricity_max, contour_length_min, contour_length_max):
    """
    Calculates the contours based on the thresholded image.
    Return:
        mask (numpy array): Background image all 255.
        contours (list): List of 2D np arrays showing contour points
    """
    try:
        _, contours, _ = cv2.findContours(thresholded_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thresholded_arr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for c in contours:
        if len(c) >= 5:
            ellipse = cv2.fitEllipseAMS(c)

            (x, y), (a, b), angle = ellipse

            eccentricity = np.sqrt(abs(a ** 2 - b ** 2)) / max(a, b)
            area = np.pi * a * b

            if (eccentricity <= eccentricity_max) and (eccentricity >= eccentricity_min):
                if (len(c)>contour_length_min) and len(c) < contour_length_max:
                    filtered_contours.append(c)

    mask = np.zeros(thresholded_arr.shape, np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, (255, 0, 0), 3)
    return mask, filtered_contours


def combine(columns, space, images, output_image_path):
    """ Combines (pathches) multiple image to form a single image
    Parameters
    ----------
    columns : int
        Number of columns in patching pattern
    space : float
        spacing between each slice when patching (in pixels)
    images : list
        list of str representing images paths
    output_image_path : str
        absolute path for the combined (resulting) image
    """
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([Image.open(image).width for image in images])
    height_max = max([Image.open(image).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(tqdm(images)):
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(output_image_path)