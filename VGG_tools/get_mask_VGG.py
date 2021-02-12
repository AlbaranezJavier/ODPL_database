import cv2
import numpy as np
import json
import os


def get_fill_convex(img, points):
    # triangle = numpy.array([[50, 30], [40, 80], [10, 90]], numpy.int32)
    cv2.fillConvexPoly(img, points, 1)
    cv2.imshow("prueba", img)
    cv2.waitKey(0)


def get_points(d, h, w):
    with open(d) as f:
        data = json.load(f)
    #image
    for key in data.keys():
        img_c0 = np.zeros((h, w))
        img_c1 = np.zeros((h, w))
        images = data[key]["filename"]
        #detection
        for key_det in data[key]["regions"].keys():
            p_x = data[key]["regions"][key_det]["shape_attributes"]["all_points_x"]
            p_y = data[key]["regions"][key_det]["shape_attributes"]["all_points_y"]
            tag = data[key]["regions"][key_det]["region_attributes"]["label"]
            # merge p_x and p_y
            # if tag == c0 get_fill_convex(img_c0, points)
            # else...
        # save images
        cv2.imwrite(d.split('.')[0]+"_c0.png", img_c0)
        cv2.imwrite(d.split('.')[0]+"_c1.png", img_c1)
        print(images)
    return images


if __name__ == "__main__":
    d = "labels_VGG.json"
    points = get_points(d)
    # get_fill_convex()
