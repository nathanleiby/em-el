#!/usr/bin/env python
# coding: utf-8

import os

from datetime import datetime
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import img_as_float
import cv2

def image_show(image, nrows=1, ncols=1, cmap='gray', figsize=(16,16), **kwargs):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax.imshow(image, cmap='gray') # TODO: how to make this write a file?
    ax.axis('off')
    return fig, ax

# ## Try different points transform for card identification
#
# Based on blog post here: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#
# This does a perspective transform using the cv2 getPerspectiveTransform and warpPerspective functions

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


RATIO_THRESHOLD = 0.015
BLUR = 15

def find_cards_in_image(image, show=True, ratio_threshold=RATIO_THRESHOLD):
    # Pre-processing: Convert frame to standard size, 1024x768
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, BLUR)
    edges = cv2.Canny(gray, 10, 25)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = image.shape[0]* image.shape[1]

    possible_cards = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(box)
        ratio = area / size
        if ratio_threshold and ratio < ratio_threshold: # Any contour large enough is a candidate
            continue
        # Mark this box as possible card
        else:
            possible_cards.append(contour)
    if show:
        # draw segment outline in green
        image_show(cv2.drawContours(image, possible_cards, -1, color=(0,255,0), thickness=3))
    return possible_cards


IMAGES_DIRECTORY = 'images/many-cards'
def find_individual_card_images_from_picture(image_filename, card_size_ratio=RATIO_THRESHOLD, save_fig=False, show=False):
    filename = os.path.join(IMAGES_DIRECTORY, image_filename)
    loaded_file = io.imread(filename)
    possible_cards = find_cards_in_image(loaded_file.copy(), show=show, ratio_threshold=card_size_ratio)
    print("\tFound {} images in image".format(len(possible_cards)))
    warped_cards = []
    for possible_card in possible_cards:
        corner_points = cv2.boxPoints(cv2.minAreaRect(possible_card))
        warped = four_point_transform(loaded_file, corner_points)

        if show:
            image_show(warped)
        if save_fig:
            filename = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f') + ".png"
            dir_path = os.path.join('images', 'candidate_images', os.path.splitext(image_filename)[0])
            try:
                os.makedirs(dir_path)
            except:
                pass
            full_path = os.path.join(dir_path, filename)
            print("\tSaving {} ...".format(full_path))
            plt.imsave(full_path, warped)
        plt.close()
        warped_cards.append(warped)
    return warped


if __name__ == "__main__":
    image_files = os.listdir(IMAGES_DIRECTORY)
    for image_filename in image_files:
        print(image_filename)
        # find_individual_card_images_from_picture(image_filename, show=True)
        # It looks like we can get more of the cards in if we relax the threshold requirement
        # find_individual_card_images_from_picture(image_filename, card_size_ratio=0.004)
        find_individual_card_images_from_picture(image_filename, card_size_ratio=0.004, save_fig=True)